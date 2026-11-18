"""Summary statistics, z-scores, confidence intervals, frequency tables, correlation."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .base import Param, Result, Role, analysis, clean, clean_pair, fmt_p
from .common import ALT, CONF, GROUP, X, XS, Y, need

CAT = "Summary & descriptive"
COR = "Correlation & association"


def describe_array(x: np.ndarray) -> dict:
    """The one canonical set of descriptive statistics (sample sd uses n-1)."""
    n = x.size
    if n == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    mean = float(np.mean(x))
    vals, cnt = np.unique(x, return_counts=True)
    return {"n": n, "mean": mean, "sd": sd, "var": sd ** 2 if n > 1 else float("nan"),
            "se": sd / np.sqrt(n) if n > 1 else float("nan"), "min": float(x.min()), "q1": float(q1),
            "median": float(med), "q3": float(q3), "max": float(x.max()), "range": float(x.max() - x.min()),
            "iqr": float(q3 - q1), "sum": float(x.sum()),
            "cv": sd / mean if n > 1 and mean != 0 else float("nan"),
            "skew": float(stats.skew(x, bias=False)) if n > 2 else float("nan"),
            "kurt": float(stats.kurtosis(x, bias=False)) if n > 3 else float("nan"),   # excess, as Stata/R e1071 type 2
            "mode": float(vals[np.argmax(cnt)]), "mad": float(np.median(np.abs(x - med)))}


@analysis("summary", "Summary statistics", CAT, [XS, Role("group", "categorical", "By group", required=False)],
          [Param("trim", "Also show 5% trimmed mean", "bool", False)],
          code={"stata": "summarize {xs}, detail", "r": "summary(df[, c({xs_q})]); sapply(df[, c({xs_q})], sd, na.rm = TRUE)",
                "python": "df[[{xs_q}]].describe().T"},
          reference="Sample sd and variance use n-1; skewness/kurtosis are the bias-corrected (type 2) estimators, "
                    "kurtosis is excess kurtosis.")
def summary(d, trim=False):
    """Mean, standard deviation, quartiles, skewness... for each selected variable."""
    r = Result("Summary statistics")
    names = d.label_list("xs")
    cols = d["xs"] if isinstance(d["xs"], list) else [d["xs"]]
    keys = ["n", "mean", "sd", "se", "min", "q1", "median", "q3", "max", "skew", "kurt"]
    heads = ["variable", "n", "mean", "std.dev", "std.err", "min", "Q1", "median", "Q3", "max", "skew", "ex.kurt"]
    if d.has("group"):
        g = np.asarray(d["group"], dtype=object).astype(str)
        for nm, c in zip(names, cols):
            c = np.asarray(c, dtype=float)
            rows = []
            for lv in dict.fromkeys(g.tolist()):
                s = describe_array(clean(c[g == lv]))
                rows.append([lv] + [s.get(k, float("nan")) for k in keys])
                r.values[(nm, lv)] = s
            r.table(f"{nm} by {d.name('group')}", ["group"] + heads[1:], rows)
    else:
        rows = []
        for nm, c in zip(names, cols):
            x = clean(c)
            s = describe_array(x)
            r.values[nm] = s
            rows.append([nm] + [s.get(k, float("nan")) for k in keys])
        r.table("", heads, rows)
        extra = []
        for nm, c in zip(names, cols):
            x = clean(c)
            s = r.values[nm]
            m = len(np.asarray(c)) - x.size
            extra.append([nm, m, s.get("var"), s.get("iqr"), s.get("cv"), s.get("mad"), s.get("mode")]
                         + ([float(stats.trim_mean(x, 0.05))] if trim else []))
        r.table("More", ["variable", "missing", "variance", "IQR", "CV", "MAD", "mode"] + (["5% trim mean"] if trim else []), extra)
    return r


@analysis("zscores", "Z-scores (standardise)", CAT, [X],
          [Param("flag", "Flag |z| ≥", "float", 3.0, min=0.5, max=10), Param("ddof", "Std.dev. denominator", "choice", "sample (n-1)",
                                                                              choices=("sample (n-1)", "population (n)"))],
          code={"stata": "egen z_{x} = std({x})", "r": "z_{x} <- as.numeric(scale(df${x}))",
                "python": "z_{x} = (df['{x}'] - df['{x}'].mean()) / df['{x}'].std(ddof=1)"},
          reference="z = (x - mean) / sd.")
def zscores(d, flag=3.0, ddof="sample (n-1)"):
    """z = (x − mean)/sd for every observation; flags unusually large |z|."""
    x_all = np.asarray(d["x"], dtype=float)
    x = clean(x_all)
    need(x.size > 1, "need at least 2 non-missing values")
    dd = 1 if ddof.startswith("sample") else 0
    m, sd = x.mean(), x.std(ddof=dd)
    need(sd > 0, "the variable is constant (sd = 0)")
    z = (x_all - m) / sd
    r = Result(f"Z-scores of {d.name('x')}")
    r.kv("Parameters", [("n", x.size), ("mean", m), ("std.dev", sd), ("denominator", f"n-{dd}" if dd else "n")])
    big = np.flatnonzero(np.abs(np.nan_to_num(z)) >= flag)
    r.text(f"Observations with |z| ≥ {flag}",
           "none" if big.size == 0 else "\n".join(f"  row {i + 1:>8,}  value {x_all[i]:>14.6g}  z = {z[i]:>8.3f}" for i in big[:200]))
    r.table("Distribution of z", ["min", "Q1", "median", "Q3", "max"],
            [list(np.nanpercentile(z, [0, 25, 50, 75, 100]))])
    r.values.update({"z": z, "mean": m, "sd": sd, "flagged": big})
    return r


@analysis("ci_mean", "Confidence interval for a mean", CAT, [X], [CONF],
          code={"stata": "ci means {x}, level({conf_pct})", "r": "t.test(df${x}, conf.level = {conf})$conf.int",
                "python": "import scipy.stats as st; st.t.interval({conf}, len(df['{x}'].dropna())-1, loc=df['{x}'].mean(), scale=st.sem(df['{x}'], nan_policy='omit'))"},
          reference="mean ± t(1-α/2, n-1)·s/√n")
def ci_mean(d, conf=0.95):
    x = clean(d["x"])
    need(x.size > 1, "need at least 2 values")
    n, m, s = x.size, x.mean(), x.std(ddof=1)
    se = s / np.sqrt(n)
    q = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
    r = Result(f"{conf:.0%} confidence interval for the mean of {d.name('x')}")
    r.kv("", [("n", n), ("mean", m), ("std.dev", s), ("std.err", se), ("t critical", q), ("lower", m - q * se), ("upper", m + q * se)])
    r.values.update(dict(mean=m, lower=m - q * se, upper=m + q * se, se=se, n=n))
    return r


@analysis("ci_var", "Confidence interval for variance / std.dev.", CAT, [X], [CONF],
          code={"stata": "* Stata: sdtest {x} == 1   (reports the sd confidence interval)", "r": "n <- sum(!is.na(df${x})); v <- var(df${x}, na.rm=TRUE); c((n-1)*v/qchisq(c(1-(1-{conf})/2, (1-{conf})/2), n-1))",
                "python": "n=df['{x}'].count(); v=df['{x}'].var(); import scipy.stats as st; [(n-1)*v/st.chi2.ppf(q,n-1) for q in ((1+{conf})/2,(1-{conf})/2)]"},
          reference="(n-1)s²/χ²(1-α/2, n-1) ≤ σ² ≤ (n-1)s²/χ²(α/2, n-1)")
def ci_var(d, conf=0.95):
    x = clean(d["x"])
    need(x.size > 1, "need at least 2 values")
    n, v = x.size, x.var(ddof=1)
    lo = (n - 1) * v / stats.chi2.ppf(1 - (1 - conf) / 2, n - 1)
    hi = (n - 1) * v / stats.chi2.ppf((1 - conf) / 2, n - 1)
    r = Result(f"{conf:.0%} confidence interval for the variance of {d.name('x')}")
    r.table("", ["", "estimate", "lower", "upper"], [["variance", v, lo, hi], ["std.dev", np.sqrt(v), np.sqrt(lo), np.sqrt(hi)]])
    r.values.update(dict(var_lo=lo, var_hi=hi, var=v))
    return r


@analysis("frequency", "Frequency table", CAT, [Role("x", "any", "Variable")],
          [Param("max_levels", "Max levels shown", "int", 100, min=1, max=100000)],
          code={"stata": "tabulate {x}, missing", "r": "table(df${x}, useNA = \"ifany\")", "python": "df['{x}'].value_counts(dropna=False)"})
def frequency(d, max_levels=100):
    x = np.asarray(d["x"], dtype=object)
    keys = np.array(["(missing)" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in x.tolist()], dtype=object)
    vals, cnt = np.unique(keys, return_counts=True)
    o = np.argsort(-cnt, kind="stable")
    vals, cnt = vals[o], cnt[o]
    tot = cnt.sum()
    rows, cum = [], 0
    for v, c in list(zip(vals, cnt))[:max_levels]:
        cum += c
        rows.append([v, int(c), 100 * c / tot, 100 * cum / tot])
    r = Result(f"Frequency table of {d.name('x')}")
    r.table("", ["value", "count", "percent", "cumulative %"], rows, decimals=2)
    if len(vals) > max_levels:
        r.note(f"showing the {max_levels} most frequent of {len(vals):,} distinct values")
    r.values.update(dict(levels=vals.tolist(), counts=cnt.tolist()))
    return r


@analysis("correlation", "Correlation matrix (r, p-values)", COR, [XS],
          [Param("method", "Method", "choice", "pearson", choices=("pearson", "spearman", "kendall")),
           Param("use", "Missing values", "choice", "pairwise", choices=("pairwise", "listwise"))],
          code={"stata": "pwcorr {xs}, sig", "r": "cor(df[, c({xs_q})], use = \"pairwise.complete.obs\", method = \"{method}\")",
                "python": "df[[{xs_q}]].corr(method='{method}')"},
          reference="Pearson r with t = r√(n-2)/√(1-r²), df = n-2; Spearman via scipy.stats.spearmanr; Kendall τ-b via scipy.stats.kendalltau.")
def correlation(d, method="pearson", use="pairwise"):
    names = d.label_list("xs")
    cols = [np.asarray(c, dtype=float) for c in d["xs"]]
    k = len(cols)
    need(k >= 2, "select at least two variables")
    if use == "listwise":
        ok = np.all([np.isfinite(c) for c in cols], axis=0)
        cols = [c[ok] for c in cols]
    R = np.eye(k)
    P = np.zeros((k, k))
    N = np.zeros((k, k), dtype=int)
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    for i in range(k):
        for j in range(k):
            a, b = clean_pair(cols[i], cols[j])
            N[i, j] = a.size
            if i == j:
                R[i, j], P[i, j] = 1.0, 0.0
            elif j < i:
                R[i, j], P[i, j] = R[j, i], P[j, i]
            elif a.size > 2 and np.ptp(a) > 0 and np.ptp(b) > 0:
                res = fn(a, b)
                R[i, j], P[i, j] = float(res[0]), float(res[1])
            else:
                R[i, j], P[i, j] = np.nan, np.nan
    r = Result(f"{method.capitalize()} correlation matrix")
    r.table("Correlation coefficients", [""] + names, [[names[i]] + list(R[i]) for i in range(k)])
    r.table("Two-sided p-values", [""] + names, [[names[i]] + [fmt_p(p) if i != j else "" for j, p in enumerate(P[i])] for i in range(k)])
    r.table("Pairwise n", [""] + names, [[names[i]] + list(N[i]) for i in range(k)])
    r.values.update(dict(r=R, p=P, n=N))
    return r


@analysis("corr_test", "Correlation test (one pair, with CI)", COR, [Role("x", "numeric", "X"), Role("y", "numeric", "Y")],
          [Param("method", "Method", "choice", "pearson", choices=("pearson", "spearman", "kendall")), ALT, CONF],
          code={"stata": "pwcorr {x} {y}, sig" , "r": "cor.test(df${x}, df${y}, method = \"{method}\", alternative = \"{alternative}\", conf.level = {conf})",
                "python": "import scipy.stats as st; st.{method}r(df['{x}'], df['{y}'])" },
          reference="Pearson: Fisher z-transform CI, z' = atanh(r), se = 1/√(n-3).")
def corr_test(d, method="pearson", alternative="two-sided", conf=0.95):
    x, y = clean_pair(d["x"], d["y"])
    need(x.size > 3, "need at least 4 complete pairs")
    n = x.size
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    if method == "pearson":
        res = stats.pearsonr(x, y, alternative=alternative)
    elif method == "spearman":
        res = stats.spearmanr(x, y, alternative=alternative)
    else:
        res = stats.kendalltau(x, y, alternative=alternative)
    rho, p = float(res[0]), float(res[1])
    r = Result(f"{method.capitalize()} correlation: {d.name('x')} vs {d.name('y')}")
    pairs = [("n", n), ("coefficient", rho), ("p-value", p)]
    lo = hi = float("nan")
    if method != "kendall" and abs(rho) < 1:
        z = np.arctanh(rho)
        se = 1 / np.sqrt(n - 3) if method == "pearson" else np.sqrt((1 + rho ** 2 / 2) / (n - 3))   # Fieller et al. 1957 for Spearman
        q = stats.norm.ppf(1 - (1 - conf) / 2)
        lo, hi = float(np.tanh(z - q * se)), float(np.tanh(z + q * se))
        pairs += [(f"{conf:.0%} CI lower", lo), (f"{conf:.0%} CI upper", hi)]
    if method == "pearson":
        t = rho * np.sqrt((n - 2) / (1 - rho ** 2)) if abs(rho) < 1 else float("inf")
        pairs += [("t statistic", t), ("df", n - 2), ("R²", rho ** 2)]
    r.kv("", pairs)
    r.values.update(dict(r=rho, p=p, n=n, lo=lo, hi=hi))
    return r


@analysis("covariance", "Covariance matrix", COR, [XS],
          code={"stata": "correlate {xs}, covariance", "r": "cov(df[, c({xs_q})], use = \"pairwise.complete.obs\")", "python": "df[[{xs_q}]].cov()"})
def covariance(d):
    names = d.label_list("xs")
    cols = [np.asarray(c, dtype=float) for c in d["xs"]]
    k = len(cols)
    C = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            a, b = clean_pair(cols[i], cols[j])
            if a.size > 1:
                C[i, j] = np.cov(a, b, ddof=1)[0, 1]
    r = Result("Covariance matrix (n-1 denominator, pairwise complete)")
    r.table("", [""] + names, [[names[i]] + list(C[i]) for i in range(k)])
    r.values["cov"] = C
    return r
