"""Parametric location / proportion tests (Wikipedia: one-sample t, unpaired t, Welch, paired t, F, Z, permutation)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .base import Result, analysis, clean, clean_pair
from .common import (ALPHA, ALT, CONF, GROUP, X, Y, alt_label, ci_t, need, p_from_t, p_from_z, sig_note, two_groups)
from ..charts.base import Param, Role

CAT = "Means & location (parametric)"


def _group_table(r, items):
    rows = []
    for name, a in items:
        rows.append([name, a.size, a.mean(), a.std(ddof=1) if a.size > 1 else np.nan,
                     a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else np.nan])
    r.table("Sample statistics", ["group", "n", "mean", "std.dev", "std.err"], rows)


def welch_df(v1, n1, v2, n2):
    a, b = v1 / n1, v2 / n2
    return (a + b) ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))


@analysis("t_one", "One-sample t-test", CAT, [X], [Param("mu0", "Hypothesised mean μ0", "float", 0.0), ALT, CONF],
          code={"stata": "ttest {x} == {mu0}", "r": "t.test(df${x}, mu = {mu0}, alternative = \"{alternative}\", conf.level = {conf})",
                "python": "import scipy.stats as st; st.ttest_1samp(df['{x}'].dropna(), {mu0}, alternative='{alternative}')"},
          reference="t = (x̄ - μ0)/(s/√n), df = n-1.")
def t_one(d, mu0=0.0, alternative="two-sided", conf=0.95):
    x = clean(d["x"])
    need(x.size > 1, "need at least 2 values")
    n, m, s = x.size, x.mean(), x.std(ddof=1)
    need(s > 0, "the variable is constant")
    se = s / np.sqrt(n)
    t = (m - mu0) / se
    p = p_from_t(t, n - 1, alternative)
    lo, hi = ci_t(m, se, n - 1, conf, alternative)
    r = Result(f"One-sample t-test: {d.name('x')}")
    r.text("Hypotheses", alt_label(alternative, "mean", f"{mu0:g}"))
    _group_table(r, [(d.name("x"), x)])
    r.kv("Test", [("t statistic", t), ("df", n - 1), ("p-value", p), (f"{conf:.0%} CI lower", lo), (f"{conf:.0%} CI upper", hi),
                  ("Cohen's d", (m - mu0) / s)])
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(t=t, df=n - 1, p=p, lo=lo, hi=hi, mean=m, d=(m - mu0) / s))
    return r


def _two_sample(d, equal_var, alternative, conf, title, mu_diff=0.0):
    a, x, b, y = two_groups(d, title)
    need(x.size > 1 and y.size > 1, "each group needs at least 2 values")
    n1, n2 = x.size, y.size
    m1, m2, v1, v2 = x.mean(), y.mean(), x.var(ddof=1), y.var(ddof=1)
    diff = m1 - m2 - mu_diff
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        df = welch_df(v1, n1, v2, n2)
        se = np.sqrt(v1 / n1 + v2 / n2)
    need(se > 0, "both groups are constant")
    t = diff / se
    p = p_from_t(t, df, alternative)
    lo, hi = ci_t(m1 - m2, se, df, conf, alternative)
    r = Result(f"{title}: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", alt_label(alternative, f"mean({a}) - mean({b})", f"{mu_diff:g}"))
    _group_table(r, [(a, x), (b, y)])
    sp = np.sqrt(sp2) if equal_var else np.sqrt((v1 + v2) / 2)
    r.kv("Test", [("difference in means", m1 - m2), ("std.err of difference", se), ("t statistic", t), ("df", df), ("p-value", p),
                  (f"{conf:.0%} CI lower", lo), (f"{conf:.0%} CI upper", hi), ("Cohen's d", (m1 - m2) / sp)])
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(t=t, df=df, p=p, lo=lo, hi=hi, diff=m1 - m2, se=se))
    return r


@analysis("t_two", "Two-sample t-test (pooled variance)", CAT, [Y, GROUP],
          [Param("mu_diff", "Hypothesised difference", "float", 0.0), ALT, CONF],
          code={"stata": "ttest {y}, by({group})", "r": "t.test({y_r} ~ {group_r}, data = df, var.equal = TRUE, alternative = \"{alternative}\", conf.level = {conf})",
                "python": "import scipy.stats as st; g=[v.dropna() for _,v in df.groupby('{group}')['{y}']]; st.ttest_ind(g[0], g[1], equal_var=True, alternative='{alternative}')"},
          reference="Student's t with pooled variance, df = n1+n2-2. Assumes equal variances (see F-test / Levene).")
def t_two(d, mu_diff=0.0, alternative="two-sided", conf=0.95):
    return _two_sample(d, True, alternative, conf, "Two-sample t-test (pooled variance)", mu_diff)


@analysis("t_welch", "Welch's t-test (unequal variances)", CAT, [Y, GROUP],
          [Param("mu_diff", "Hypothesised difference", "float", 0.0), ALT, CONF],
          code={"stata": "ttest {y}, by({group}) unequal welch", "r": "t.test({y_r} ~ {group_r}, data = df, var.equal = FALSE, alternative = \"{alternative}\", conf.level = {conf})",
                "python": "import scipy.stats as st; g=[v.dropna() for _,v in df.groupby('{group}')['{y}']]; st.ttest_ind(g[0], g[1], equal_var=False, alternative='{alternative}')"},
          reference="Welch–Satterthwaite df = (s1²/n1+s2²/n2)² / [(s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1)].")
def t_welch(d, mu_diff=0.0, alternative="two-sided", conf=0.95):
    return _two_sample(d, False, alternative, conf, "Welch's t-test", mu_diff)


@analysis("t_paired", "Paired t-test", CAT, [Role("x", "numeric", "Before / first measure"), Role("y", "numeric", "After / second measure")],
          [Param("mu_diff", "Hypothesised mean difference", "float", 0.0), ALT, CONF],
          code={"stata": "ttest {x} == {y}", "r": "t.test(df${x}, df${y}, paired = TRUE, mu = {mu_diff}, alternative = \"{alternative}\", conf.level = {conf})",
                "python": "import scipy.stats as st; st.ttest_rel(df['{x}'], df['{y}'], nan_policy='omit', alternative='{alternative}')"},
          reference="One-sample t on the differences x - y, df = n-1.")
def t_paired(d, mu_diff=0.0, alternative="two-sided", conf=0.95):
    x, y = clean_pair(d["x"], d["y"])
    need(x.size > 1, "need at least 2 complete pairs")
    diff = x - y
    n, m, s = diff.size, diff.mean(), diff.std(ddof=1)
    need(s > 0, "all differences are identical")
    se = s / np.sqrt(n)
    t = (m - mu_diff) / se
    p = p_from_t(t, n - 1, alternative)
    lo, hi = ci_t(m, se, n - 1, conf, alternative)
    r = Result(f"Paired t-test: {d.name('x')} − {d.name('y')}")
    r.text("Hypotheses", alt_label(alternative, "mean difference", f"{mu_diff:g}"))
    _group_table(r, [(d.name("x"), x), (d.name("y"), y), ("difference", diff)])
    r.kv("Test", [("t statistic", t), ("df", n - 1), ("p-value", p), (f"{conf:.0%} CI lower", lo), (f"{conf:.0%} CI upper", hi),
                  ("Cohen's d (paired)", m / s), ("correlation of pairs", float(np.corrcoef(x, y)[0, 1]))])
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(t=t, df=n - 1, p=p, lo=lo, hi=hi, diff=m))
    return r


@analysis("f_test", "F-test for equality of two variances", CAT, [Y, GROUP], [ALT, CONF],
          code={"stata": "sdtest {y}, by({group})", "r": "var.test({y_r} ~ {group_r}, data = df, alternative = \"{alternative}\", conf.level = {conf})",
                "python": "import scipy.stats as st; g=[v.dropna() for _,v in df.groupby('{group}')['{y}']]; F=g[0].var()/g[1].var(); F, 2*min(st.f.cdf(F,len(g[0])-1,len(g[1])-1), st.f.sf(F,len(g[0])-1,len(g[1])-1))"},
          reference="F = s1²/s2² ~ F(n1-1, n2-1) under normality; two-sided p = 2·min(P(F≤f), P(F≥f)).")
def f_test(d, alternative="two-sided", conf=0.95):
    a, x, b, y = two_groups(d, "F-test")
    need(x.size > 1 and y.size > 1, "each group needs at least 2 values")
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    need(v2 > 0, "second group is constant")
    d1, d2 = x.size - 1, y.size - 1
    F = v1 / v2
    cdf = stats.f.cdf(F, d1, d2)
    p = {"two-sided": min(1.0, 2 * min(cdf, 1 - cdf)), "less": cdf, "greater": 1 - cdf}[alternative]
    if alternative == "two-sided":
        lo, hi = F / stats.f.ppf(1 - (1 - conf) / 2, d1, d2), F / stats.f.ppf((1 - conf) / 2, d1, d2)
    elif alternative == "less":
        lo, hi = 0.0, F / stats.f.ppf(1 - conf, d1, d2)
    else:
        lo, hi = F / stats.f.ppf(conf, d1, d2), np.inf
    r = Result(f"F-test for equal variances: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", alt_label(alternative, f"var({a})/var({b})", "1"))
    r.kv("Test", [("variance " + a, v1), ("variance " + b, v2), ("F statistic", F), ("df numerator", d1), ("df denominator", d2),
                  ("p-value", p), (f"{conf:.0%} CI ratio lower", lo), (f"{conf:.0%} CI ratio upper", hi)])
    r.note("The F-test is very sensitive to non-normality; consider Levene / Brown–Forsythe instead.")
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(F=F, df1=d1, df2=d2, p=p, lo=lo, hi=hi))
    return r


@analysis("z_one", "Z-test for one mean (σ known)", CAT, [X],
          [Param("mu0", "Hypothesised mean μ0", "float", 0.0), Param("sigma", "Known population σ", "float", 1.0, min=1e-12), ALT, CONF],
          code={"stata": "* Stata: ztest is not built in; z = (mean - {mu0}) / ({sigma}/sqrt(N))\nsummarize {x}\ndisplay (r(mean)-{mu0})/({sigma}/sqrt(r(N)))",
                "r": "z <- (mean(df${x}, na.rm=TRUE) - {mu0}) / ({sigma}/sqrt(sum(!is.na(df${x})))); c(z = z, p = 2*pnorm(-abs(z)))",
                "python": "from statsmodels.stats.weightstats import ztest; ztest(df['{x}'].dropna(), value={mu0}, alternative='two-sided')"},
          reference="z = (x̄ - μ0)/(σ/√n) ~ N(0,1).")
def z_one(d, mu0=0.0, sigma=1.0, alternative="two-sided", conf=0.95):
    x = clean(d["x"])
    need(x.size > 0, "no data")
    n, m = x.size, x.mean()
    se = sigma / np.sqrt(n)
    z = (m - mu0) / se
    p = p_from_z(z, alternative)
    q = stats.norm.ppf(1 - (1 - conf) / 2 if alternative == "two-sided" else conf)
    lo, hi = (m - q * se, m + q * se) if alternative == "two-sided" else ((-np.inf, m + q * se) if alternative == "less" else (m - q * se, np.inf))
    r = Result(f"Z-test for a mean: {d.name('x')}")
    r.text("Hypotheses", alt_label(alternative, "mean", f"{mu0:g}") + f"   (σ = {sigma:g} known)")
    r.kv("Test", [("n", n), ("sample mean", m), ("std.err σ/√n", se), ("z statistic", z), ("p-value", p),
                  (f"{conf:.0%} CI lower", lo), (f"{conf:.0%} CI upper", hi)])
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(z=z, p=p, lo=lo, hi=hi))
    return r


@analysis("z_two", "Z-test for two means (σ known)", CAT, [Y, GROUP],
          [Param("sigma1", "Known σ of group 1", "float", 1.0, min=1e-12), Param("sigma2", "Known σ of group 2", "float", 1.0, min=1e-12),
           Param("mu_diff", "Hypothesised difference", "float", 0.0), ALT, CONF],
          code={"stata": "* two-sample z-test: z = (m1-m2-{mu_diff}) / sqrt({sigma1}^2/n1 + {sigma2}^2/n2)\ntabstat {y}, by({group}) stat(mean n)",
                "r": "g <- split(df${y}, df${group}); z <- (mean(g[[1]], na.rm=TRUE) - mean(g[[2]], na.rm=TRUE) - {mu_diff}) / sqrt({sigma1}^2/sum(!is.na(g[[1]])) + {sigma2}^2/sum(!is.na(g[[2]]))); c(z = z, p = 2*pnorm(-abs(z)))",
                "python": "from statsmodels.stats.weightstats import ztest; g=[v.dropna() for _,v in df.groupby('{group}')['{y}']]; ztest(g[0], g[1], value={mu_diff})"},
          reference="z = (x̄1 - x̄2 - Δ0)/√(σ1²/n1 + σ2²/n2).")
def z_two(d, sigma1=1.0, sigma2=1.0, mu_diff=0.0, alternative="two-sided", conf=0.95):
    a, x, b, y = two_groups(d, "Two-sample z-test")
    se = np.sqrt(sigma1 ** 2 / x.size + sigma2 ** 2 / y.size)
    diff = x.mean() - y.mean()
    z = (diff - mu_diff) / se
    p = p_from_z(z, alternative)
    q = stats.norm.ppf(1 - (1 - conf) / 2)
    r = Result(f"Two-sample Z-test: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", alt_label(alternative, f"mean({a}) - mean({b})", f"{mu_diff:g}") + f"   (σ1 = {sigma1:g}, σ2 = {sigma2:g})")
    _group_table(r, [(a, x), (b, y)])
    r.kv("Test", [("difference", diff), ("std.err", se), ("z statistic", z), ("p-value", p), (f"{conf:.0%} CI lower", diff - q * se), (f"{conf:.0%} CI upper", diff + q * se)])
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(z=z, p=p, diff=diff))
    return r


def _binary(x, success):
    """Boolean success array + n, ignoring missing."""
    a = np.asarray(x, dtype=object)
    ok = np.array([not (v is None or (isinstance(v, float) and np.isnan(v))) for v in a.tolist()], dtype=bool)
    a = a[ok]
    if success in (None, ""):
        u = sorted(set(map(str, a.tolist())))
        need(len(u) <= 2, "variable has more than 2 distinct values; give the 'success' value")
        success = u[-1] if len(u) == 2 else u[0]
    s = np.array([str(v) == str(success) or _same_num(v, success) for v in a.tolist()], dtype=bool)
    return s, str(success)


def _same_num(v, s):
    try:
        return float(v) == float(s)
    except (TypeError, ValueError):
        return False


@analysis("prop_one", "One-sample proportion test (z and exact)", "Means & location (parametric)", [Role("x", "any", "Binary variable")],
          [Param("success", "Value counted as success (blank = last level)", "text", ""), Param("p0", "Hypothesised proportion p0", "float", 0.5, min=0.0001, max=0.9999),
           ALT, CONF],
          code={"stata": "prtest {x} == {p0}", "r": "prop.test(sum(df${x} == {success_r}), sum(!is.na(df${x})), p = {p0}, alternative = \"{alternative}\", correct = FALSE); binom.test(sum(df${x} == {success_r}), sum(!is.na(df${x})), p = {p0})",
                "python": "from statsmodels.stats.proportion import proportions_ztest; proportions_ztest((df['{x}']=={success_r}).sum(), df['{x}'].count(), {p0})"},
          reference="z = (p̂-p0)/√(p0(1-p0)/n); exact binomial p-value and Clopper–Pearson / Wilson intervals.")
def prop_one(d, success="", p0=0.5, alternative="two-sided", conf=0.95):
    s, suc = _binary(d["x"], success)
    n, k = s.size, int(s.sum())
    need(n > 0, "no data")
    ph = k / n
    se0 = np.sqrt(p0 * (1 - p0) / n)
    z = (ph - p0) / se0
    pz = p_from_z(z, alternative)
    bt = stats.binomtest(k, n, p0, alternative=alternative)
    cp = bt.proportion_ci(conf, method="exact")
    wil = stats.binomtest(k, n, p0).proportion_ci(conf, method="wilson")
    r = Result(f"One-sample proportion test: {d.name('x')} = {suc}")
    r.text("Hypotheses", alt_label(alternative, "proportion", f"{p0:g}"))
    r.kv("Data", [("successes", k), ("n", n), ("sample proportion", ph)])
    r.table("Tests", ["method", "statistic", "p-value"], [["z-test (null SE)", z, pz], ["exact binomial", "", bt.pvalue]])
    r.table("Confidence intervals", ["method", "lower", "upper"], [["Clopper–Pearson (exact)", cp.low, cp.high], ["Wilson score", wil.low, wil.high]])
    r.text("Conclusion", sig_note(bt.pvalue) + "  (exact test)")
    r.values.update(dict(z=z, p_z=pz, p_exact=bt.pvalue, k=k, n=n, cp=(cp.low, cp.high), wilson=(wil.low, wil.high)))
    return r


@analysis("prop_two", "Two-sample proportion test", "Means & location (parametric)", [Role("x", "any", "Binary variable"), GROUP],
          [Param("success", "Value counted as success (blank = last level)", "text", ""), ALT, CONF],
          code={"stata": "prtest {x}, by({group})", "r": "t <- table(df${group}, df${x} == {success_r}); prop.test(t[, \"TRUE\"], rowSums(t), alternative = \"{alternative}\", conf.level = {conf}, correct = FALSE)",
                "python": "from statsmodels.stats.proportion import proportions_ztest; t=pd.crosstab(df['{group}'], df['{x}']=={success_r}); proportions_ztest(t[True].values, t.sum(axis=1).values)"},
          reference="Pooled z = (p̂1-p̂2)/√(p̂(1-p̂)(1/n1+1/n2)); CI uses the unpooled (Wald) standard error.")
def prop_two(d, success="", alternative="two-sided", conf=0.95):
    s, suc = _binary(d["x"], success)
    g = np.asarray(d["group"], dtype=object)
    ok = np.array([not (v is None or (isinstance(v, float) and np.isnan(v))) for v in np.asarray(d["x"], dtype=object).tolist()])
    g = np.array([str(v) for v in g[ok].tolist()], dtype=object)
    lv = list(dict.fromkeys(g.tolist()))
    need(len(lv) == 2, f"grouping variable must have exactly 2 groups (found {len(lv)})")
    k = [int(s[g == l].sum()) for l in lv]
    n = [int((g == l).sum()) for l in lv]
    p1, p2 = k[0] / n[0], k[1] / n[1]
    pp = (k[0] + k[1]) / (n[0] + n[1])
    se_p = np.sqrt(pp * (1 - pp) * (1 / n[0] + 1 / n[1]))
    need(se_p > 0, "all observations are in the same outcome")
    z = (p1 - p2) / se_p
    pval = p_from_z(z, alternative)
    se_u = np.sqrt(p1 * (1 - p1) / n[0] + p2 * (1 - p2) / n[1])
    q = stats.norm.ppf(1 - (1 - conf) / 2)
    r = Result(f"Two-sample proportion test: {d.name('x')} = {suc} by {d.name('group')}")
    r.text("Hypotheses", alt_label(alternative, f"p({lv[0]}) - p({lv[1]})", "0"))
    r.table("Data", ["group", "successes", "n", "proportion"], [[lv[0], k[0], n[0], p1], [lv[1], k[1], n[1], p2]])
    r.kv("Test", [("difference", p1 - p2), ("pooled proportion", pp), ("z statistic", z), ("p-value", pval),
                  (f"{conf:.0%} CI lower", p1 - p2 - q * se_u), (f"{conf:.0%} CI upper", p1 - p2 + q * se_u)])
    r.text("Conclusion", sig_note(pval))
    r.values.update(dict(z=z, p=pval, diff=p1 - p2))
    return r


@analysis("permutation", "Permutation test (two groups)", CAT, [Y, GROUP],
          [Param("stat", "Statistic", "choice", "mean", choices=("mean", "median")), Param("n_resamples", "Resamples", "int", 20000, min=100, max=5_000_000),
           Param("seed", "Random seed", "int", 12345), ALT],
          code={"stata": "permute {group} diff = (r(mu_1)-r(mu_2)), reps({n_resamples}) seed({seed}): ttest {y}, by({group})",
                "r": "set.seed({seed}); g <- split(df${y}, df${group}); obs <- {stat}(g[[1]], na.rm=TRUE) - {stat}(g[[2]], na.rm=TRUE); pool <- na.omit(df${y}); n1 <- length(na.omit(g[[1]])); sims <- replicate({n_resamples}, {{ i <- sample(length(pool), n1); {stat}(pool[i]) - {stat}(pool[-i]) }}); mean(abs(sims) >= abs(obs))",
                "python": "import scipy.stats as st, numpy as np; g=[v.dropna().values for _,v in df.groupby('{group}')['{y}']]; st.permutation_test(g, lambda a,b: np.{stat}(a)-np.{stat}(b), n_resamples={n_resamples}, random_state={seed}, alternative='{alternative}')"},
          reference="Monte-Carlo permutation distribution of the group difference (exact when all permutations are enumerated).")
def permutation(d, stat="mean", n_resamples=20000, seed=12345, alternative="two-sided"):
    a, x, b, y = two_groups(d, "Permutation test")
    f = np.mean if stat == "mean" else np.median
    res = stats.permutation_test((x, y), lambda u, v: f(u) - f(v), permutation_type="independent",
                                 n_resamples=int(n_resamples), alternative=alternative, random_state=np.random.default_rng(int(seed)))
    obs = float(f(x) - f(y))
    r = Result(f"Permutation test of difference in {stat}s: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", "H0: the two groups come from the same distribution (labels exchangeable)")
    _group_table(r, [(a, x), (b, y)])
    r.kv("Test", [(f"observed difference in {stat}", obs), ("p-value", float(res.pvalue)), ("resamples", int(n_resamples)), ("seed", int(seed))])
    r.note("P-value is Monte-Carlo (add-one corrected) unless the number of distinct permutations is below the resample count, "
           "in which case it is exact. Re-running with the same seed reproduces it exactly.")
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(p=float(res.pvalue), obs=obs))
    return r
