"""ANOVA family and variance-homogeneity tests."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..charts.base import Param, Role
from .base import Result, analysis, clean
from .common import ALPHA, CONF, GROUP, Y, groups_of, need, sig_note

CAT = "Variance & ANOVA"


def anova_oneway(arrs):
    """Classical one-way ANOVA table values from a list of arrays."""
    allv = np.concatenate(arrs)
    N, k = allv.size, len(arrs)
    gm = allv.mean()
    ssb = sum(a.size * (a.mean() - gm) ** 2 for a in arrs)
    ssw = sum(((a - a.mean()) ** 2).sum() for a in arrs)
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw if msw > 0 else np.inf
    return dict(ssb=ssb, ssw=ssw, dfb=dfb, dfw=dfw, msb=msb, msw=msw, F=F, p=float(stats.f.sf(F, dfb, dfw)), N=N, k=k,
                eta2=ssb / (ssb + ssw), omega2=(ssb - dfb * msw) / (ssb + ssw + msw))


@analysis("anova_one", "One-way ANOVA", CAT, [Y, GROUP], [Param("posthoc", "Post-hoc Tukey HSD", "bool", True), ALPHA],
          code={"stata": "oneway {y} {group}, tabulate", "r": "fit <- aov({y_r} ~ factor({group_r}), data = df); summary(fit); TukeyHSD(fit)",
                "python": "import scipy.stats as st; st.f_oneway(*[v.dropna() for _,v in df.groupby('{group}')['{y}']])"},
          reference="F = MSB/MSW, df = (k-1, N-k). η² = SSB/SST; ω² = (SSB-(k-1)MSW)/(SST+MSW).")
def anova_one(d, posthoc=True, alpha=0.05):
    gs = groups_of(d["y"], d["group"], min_n=1)
    need(len(gs) >= 2, "need at least 2 groups")
    arrs = list(gs.values())
    need(all(a.size >= 1 for a in arrs) and sum(a.size for a in arrs) > len(arrs), "need more observations than groups")
    t = anova_oneway(arrs)
    ref = stats.f_oneway(*arrs)
    r = Result(f"One-way ANOVA: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", "H0: all group means are equal")
    r.table("Group means", ["group", "n", "mean", "std.dev"], [[n, a.size, a.mean(), a.std(ddof=1) if a.size > 1 else np.nan] for n, a in gs.items()])
    r.table("ANOVA table", ["source", "SS", "df", "MS", "F", "p-value"],
            [["between groups", t["ssb"], t["dfb"], t["msb"], t["F"], t["p"]], ["within groups", t["ssw"], t["dfw"], t["msw"], "", ""],
             ["total", t["ssb"] + t["ssw"], t["dfb"] + t["dfw"], "", "", ""]])
    r.kv("Effect size", [("η² (eta squared)", t["eta2"]), ("ω² (omega squared)", t["omega2"])])
    r.text("Conclusion", sig_note(t["p"], alpha))
    if posthoc and len(arrs) > 2:
        try:
            res = stats.tukey_hsd(*arrs)
            names = list(gs)
            rows = []
            ci = res.confidence_interval(1 - alpha)
            for i in range(len(arrs)):
                for j in range(i + 1, len(arrs)):
                    rows.append([f"{names[i]} − {names[j]}", arrs[i].mean() - arrs[j].mean(), ci.low[i, j], ci.high[i, j], float(res.pvalue[i, j]),
                                 "*" if res.pvalue[i, j] < alpha else ""])
            r.table(f"Tukey HSD pairwise comparisons ({1 - alpha:.0%} family-wise CI)", ["pair", "difference", "lower", "upper", "adj. p", "sig"], rows)
        except Exception as e:  # pragma: no cover
            r.note(f"Tukey HSD unavailable: {e}")
    r.values.update(dict(F=t["F"], p=t["p"], dfb=t["dfb"], dfw=t["dfw"], eta2=t["eta2"], scipy_F=float(ref.statistic), scipy_p=float(ref.pvalue)))
    return r


@analysis("anova_welch", "Welch's ANOVA (unequal variances)", CAT, [Y, GROUP],
          code={"stata": "oneway {y} {group}   /* Welch: use  robvar / or: anova ... , vce(robust) */", "r": "oneway.test({y_r} ~ factor({group_r}), data = df, var.equal = FALSE)",
                "python": "import pingouin as pg; pg.welch_anova(df, dv='{y}', between='{group}')   # or statsmodels.stats.oneway.anova_oneway(..., use_var='unequal')"},
          reference="Welch (1951): F* = [Σw_i(x̄_i - x̄_w)²/(k-1)] / [1 + 2(k-2)/(k²-1)·Σ(1-w_i/W)²/(n_i-1)], df2 = (k²-1)/(3Σ(1-w_i/W)²/(n_i-1)).")
def anova_welch(d):
    gs = groups_of(d["y"], d["group"])
    need(len(gs) >= 2 and all(a.size >= 2 for a in gs.values()), "need ≥2 groups with ≥2 values each")
    arrs = list(gs.values())
    k = len(arrs)
    n = np.array([a.size for a in arrs], float)
    m = np.array([a.mean() for a in arrs])
    v = np.array([a.var(ddof=1) for a in arrs])
    need(np.all(v > 0), "a group is constant")
    w = n / v
    W = w.sum()
    mw = (w * m).sum() / W
    lam = (((1 - w / W) ** 2) / (n - 1)).sum()
    F = ((w * (m - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    df2 = (k ** 2 - 1) / (3 * lam)
    p = float(stats.f.sf(F, k - 1, df2))
    r = Result(f"Welch's ANOVA: {d.name('y')} by {d.name('group')}")
    r.table("Group means", ["group", "n", "mean", "variance"], [[nm, a.size, a.mean(), a.var(ddof=1)] for nm, a in gs.items()])
    r.kv("Test", [("Welch F", F), ("df1", k - 1), ("df2", df2), ("p-value", p)])
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(F=F, df1=k - 1, df2=df2, p=p))
    return r


@analysis("levene", "Levene / Brown–Forsythe test (equal variances)", CAT, [Y, GROUP],
          [Param("center", "Centre", "choice", "median", choices=("median", "mean", "trimmed"))],
          code={"stata": "robvar {y}, by({group})", "r": "# install.packages('car'); car::leveneTest({y_r} ~ factor({group_r}), data = df, center = {center})",
                "python": "import scipy.stats as st; st.levene(*[v.dropna() for _,v in df.groupby('{group}')['{y}']], center='{center}')"},
          reference="W = ANOVA F on |x_ij - centre_i|. centre = median is the Brown–Forsythe variant (robust).")
def levene(d, center="median"):
    gs = groups_of(d["y"], d["group"])
    need(len(gs) >= 2, "need at least 2 groups")
    res = stats.levene(*gs.values(), center=center)
    r = Result(f"Levene's test ({center}-centred): {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", "H0: all groups have equal variances")
    r.table("Group variances", ["group", "n", "variance", "std.dev"], [[n, a.size, a.var(ddof=1), a.std(ddof=1)] for n, a in gs.items()])
    r.kv("Test", [("W statistic", float(res.statistic)), ("df1", len(gs) - 1), ("df2", sum(a.size for a in gs.values()) - len(gs)), ("p-value", float(res.pvalue))])
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(W=float(res.statistic), p=float(res.pvalue)))
    return r


@analysis("bartlett", "Bartlett's test (equal variances)", CAT, [Y, GROUP],
          code={"stata": "sdtest {y}, by({group})  /* 2 groups;  for k groups: robvar */", "r": "bartlett.test({y_r} ~ factor({group_r}), data = df)",
                "python": "import scipy.stats as st; st.bartlett(*[v.dropna() for _,v in df.groupby('{group}')['{y}']])"},
          reference="Bartlett (1937): T = [(N-k)ln s_p² - Σ(n_i-1)ln s_i²] / [1 + (Σ1/(n_i-1) - 1/(N-k))/(3(k-1))] ~ χ²(k-1). Sensitive to non-normality.")
def bartlett(d):
    gs = groups_of(d["y"], d["group"])
    need(len(gs) >= 2 and all(a.size >= 2 for a in gs.values()), "need ≥2 groups with ≥2 values each")
    res = stats.bartlett(*gs.values())
    r = Result(f"Bartlett's test: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", "H0: all groups have equal variances (assumes normality)")
    r.table("Group variances", ["group", "n", "variance"], [[n, a.size, a.var(ddof=1)] for n, a in gs.items()])
    r.kv("Test", [("χ² statistic", float(res.statistic)), ("df", len(gs) - 1), ("p-value", float(res.pvalue))])
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(chi2=float(res.statistic), p=float(res.pvalue)))
    return r


@analysis("anova_two", "Two-way ANOVA (with interaction)", CAT, [Y, Role("a", "categorical", "Factor A"), Role("b", "categorical", "Factor B")],
          [Param("interaction", "Include A×B interaction", "bool", True), Param("ss_type", "Sums of squares", "choice", "II", choices=("I", "II", "III"))],
          code={"stata": "anova {y} {a}##{b}", "r": "fit <- aov({y_r} ~ factor({a_r}) {inter_r} factor({b_r}), data = df); summary(fit)   # type I SS; car::Anova(fit, type = 2) for type II",
                "python": "import statsmodels.api as sm, statsmodels.formula.api as smf; sm.stats.anova_lm(smf.ols('{y} ~ C({a}) {inter_py} C({b})', df).fit(), typ={ss_num})"},
          reference="statsmodels OLS + anova_lm. Type II/III results coincide with Stata's `anova` / R car::Anova for balanced designs; "
                    "type III uses sum-to-zero (effect) coding.")
def anova_two(d, interaction=True, ss_type="II"):
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    df = pd.DataFrame({"y": np.asarray(d["y"], dtype=float), "a": np.asarray(d["a"], dtype=object).astype(str), "b": np.asarray(d["b"], dtype=object).astype(str)}).dropna()
    need(df["a"].nunique() >= 2 and df["b"].nunique() >= 2, "each factor needs at least 2 levels")
    need(len(df) > df["a"].nunique() * df["b"].nunique(), "not enough observations for the full model")
    inter = "*" if interaction else "+"
    contr = "C(a, Sum) {} C(b, Sum)".format(inter) if ss_type == "III" else "C(a) {} C(b)".format(inter)
    fit = smf.ols(f"y ~ {contr}", df).fit()
    tab = sm.stats.anova_lm(fit, typ={"I": 1, "II": 2, "III": 3}[ss_type])
    r = Result(f"Two-way ANOVA (type {ss_type} SS): {d.name('y')} ~ {d.name('a')} {inter} {d.name('b')}")
    rows = []
    for idx, row in tab.iterrows():
        nm = str(idx).replace("C(a, Sum)", d.name("a")).replace("C(b, Sum)", d.name("b")).replace("C(a)", d.name("a")).replace("C(b)", d.name("b")).replace(":", " × ")
        if nm == "Intercept" and ss_type == "III":
            continue
        rows.append([nm, row["sum_sq"], row["df"], row["sum_sq"] / row["df"] if row["df"] else "", row.get("F", ""), row.get("PR(>F)", "")])
    r.table("ANOVA table", ["source", "SS", "df", "MS", "F", "p-value"], rows)
    cells = df.groupby(["a", "b"])["y"].agg(["count", "mean", "std"]).reset_index()
    r.table("Cell means", [d.name("a"), d.name("b"), "n", "mean", "std.dev"], cells.values.tolist())
    r.kv("Model fit", [("R²", float(fit.rsquared)), ("adj. R²", float(fit.rsquared_adj)), ("residual std.err", float(np.sqrt(fit.mse_resid)))])
    if cells["count"].nunique() > 1:
        r.note("Unbalanced design: type I SS depend on the order of the factors; prefer type II (no interaction) or III.")
    r.values.update(dict(table=tab, r2=float(fit.rsquared)))
    return r


@analysis("tukey_hsd", "Tukey HSD pairwise comparisons", CAT, [Y, GROUP], [ALPHA],
          code={"stata": "pwmean {y}, over({group}) mcompare(tukey) effects", "r": "TukeyHSD(aov({y_r} ~ factor({group_r}), data = df))",
                "python": "import scipy.stats as st; st.tukey_hsd(*[v.dropna() for _,v in df.groupby('{group}')['{y}']])"},
          reference="Studentised-range based simultaneous intervals (Tukey–Kramer for unequal n) via scipy.stats.tukey_hsd.")
def tukey_hsd(d, alpha=0.05):
    gs = groups_of(d["y"], d["group"])
    need(len(gs) >= 3, "need at least 3 groups (use a t-test for 2)")
    arrs = list(gs.values())
    res = stats.tukey_hsd(*arrs)
    ci = res.confidence_interval(1 - alpha)
    names = list(gs)
    rows = []
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            rows.append([f"{names[i]} − {names[j]}", arrs[i].mean() - arrs[j].mean(), ci.low[i, j], ci.high[i, j], float(res.pvalue[i, j]), "*" if res.pvalue[i, j] < alpha else ""])
    r = Result(f"Tukey HSD: {d.name('y')} by {d.name('group')}")
    r.table(f"Pairwise differences ({1 - alpha:.0%} family-wise)", ["pair", "difference", "lower", "upper", "adj. p", "sig"], rows)
    r.values.update(dict(p=np.array(res.pvalue)))
    return r


@analysis("anova_rm", "Repeated-measures ANOVA (one within factor)", CAT, [Role("xs", "numeric_multi", "Measurements (one column per condition; rows = subjects)")],
          code={"stata": "* reshape long first, then:  anova y subject condition, repeated(condition)", "r": "# reshape to long, then: summary(aov(y ~ cond + Error(subject/cond), data = long))",
                "python": "from statsmodels.stats.anova import AnovaRM  # on the long-format data"},
          reference="SS_cond / MS_error with error = subjects × conditions; Greenhouse–Geisser ε reported for the sphericity violation.")
def anova_rm(d):
    names = d.label_list("xs")
    X = np.column_stack([np.asarray(c, dtype=float) for c in d["xs"]])
    X = X[np.all(np.isfinite(X), axis=1)]
    n, k = X.shape
    need(k >= 2 and n >= 3, "need at least 2 conditions and 3 complete subjects")
    gm = X.mean()
    ss_cond = n * ((X.mean(0) - gm) ** 2).sum()
    ss_subj = k * ((X.mean(1) - gm) ** 2).sum()
    ss_tot = ((X - gm) ** 2).sum()
    ss_err = ss_tot - ss_cond - ss_subj
    df_c, df_e = k - 1, (k - 1) * (n - 1)
    F = (ss_cond / df_c) / (ss_err / df_e)
    p = float(stats.f.sf(F, df_c, df_e))
    # Greenhouse–Geisser epsilon from the covariance matrix
    S = np.cov(X, rowvar=False)
    Sc = S - S.mean(0)[None, :] - S.mean(1)[:, None] + S.mean()
    eps = float(np.trace(Sc) ** 2 / ((k - 1) * (Sc ** 2).sum())) if k > 2 else 1.0
    pgg = float(stats.f.sf(F, df_c * eps, df_e * eps))
    r = Result("Repeated-measures ANOVA")
    r.table("Condition means", ["condition", "mean", "std.dev"], [[names[j], X[:, j].mean(), X[:, j].std(ddof=1)] for j in range(k)])
    r.table("ANOVA table", ["source", "SS", "df", "MS", "F", "p-value"],
            [["conditions", ss_cond, df_c, ss_cond / df_c, F, p], ["subjects", ss_subj, n - 1, ss_subj / (n - 1), "", ""], ["error", ss_err, df_e, ss_err / df_e, "", ""]])
    r.kv("Sphericity-corrected", [("Greenhouse–Geisser ε", eps), ("p-value (GG-corrected)", pgg), ("partial η²", ss_cond / (ss_cond + ss_err))])
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(F=F, p=p, eps=eps, p_gg=pgg))
    return r
