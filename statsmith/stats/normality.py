"""Normality and distribution-fit tests."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..charts.base import Param, Role
from .base import Result, analysis, clean
from .common import X, need, sig_note

CAT = "Normality & distribution fit"


def shapiro_francia(x):
    """Shapiro–Francia W' and p-value (Royston 1993 approximation, valid for 5 ≤ n ≤ 5000).

    W' is the squared correlation between the ordered sample and the expected normal order statistics
    approximated by Blom scores m_i = Φ⁻¹((i - 3/8)/(n + 1/4)).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    m = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    w = float(np.corrcoef(x, m)[0, 1] ** 2)
    u = np.log(n)
    v = np.log(u)
    mu = -1.2725 + 1.0521 * (v - u)
    sig = 1.0308 - 0.26758 * (v + 2 / u)
    z = (np.log(1 - w) - mu) / sig
    return w, float(stats.norm.sf(z)), float(z)


def lilliefors_p(D: float, n: int) -> float:
    """Lilliefors (1967) p-value for the normal case.

    Uses the Dallal & Wilkinson (1986) analytic approximation (as in R's nortest::lillie.test), accurate for p < 0.1 and
    refined with their Table-based correction for larger p.
    """
    Kd = D
    nd = n
    if nd > 100:
        Kd = D * (nd / 100) ** 0.49
        nd = 100
    p = np.exp(-7.01256 * Kd ** 2 * (nd + 2.78019) + 2.99587 * Kd * np.sqrt(nd + 2.78019) - 0.122119 + 0.974598 / np.sqrt(nd) + 1.67997 / nd)
    if p > 0.1:
        KK = (np.sqrt(nd) - 0.01 + 0.85 / np.sqrt(nd)) * D
        if KK <= 0.302:
            p = 1.0
        elif KK <= 0.5:
            p = 2.76773 - 19.828315 * KK + 80.709644 * KK ** 2 - 138.55152 * KK ** 3 + 81.218052 * KK ** 4
        elif KK <= 0.9:
            p = -4.901232 + 40.662806 * KK - 97.490286 * KK ** 2 + 94.029866 * KK ** 3 - 32.355711 * KK ** 4
        elif KK <= 1.31:
            p = 6.198765 - 19.558097 * KK + 23.186922 * KK ** 2 - 12.234627 * KK ** 3 + 2.423045 * KK ** 4
        else:
            p = 0.0
    return float(min(max(p, 0.0), 1.0))


def lilliefors_stat(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    z = (x - x.mean()) / x.std(ddof=1)
    cdf = stats.norm.cdf(z)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - cdf), np.max(cdf - (i - 1) / n)))


def _norm_report(d, title, stat_name, stat, p, extra=()):
    r = Result(f"{title}: {d.name('x')}")
    r.text("Hypotheses", "H0: the data are drawn from a normal distribution")
    x = clean(d["x"])
    r.kv("Test", [("n", x.size), (stat_name, stat), ("p-value", p)] + list(extra))
    r.text("Conclusion", sig_note(p) + "  → " + ("evidence against normality" if p < 0.05 else "no evidence against normality"))
    return r


@analysis("shapiro_wilk", "Shapiro–Wilk normality test", CAT, [X],
          code={"stata": "swilk {x}", "r": "shapiro.test(df${x})", "python": "import scipy.stats as st; st.shapiro(df['{x}'].dropna())"},
          reference="Shapiro & Wilk (1965); algorithm AS R94 (Royston 1995) as implemented in scipy. Valid for 3 ≤ n ≤ 5000.")
def shapiro_wilk(d):
    x = clean(d["x"])
    need(3 <= x.size, "need at least 3 values")
    need(np.ptp(x) > 0, "the variable is constant")
    res = stats.shapiro(x)
    r = _norm_report(d, "Shapiro–Wilk test", "W statistic", float(res.statistic), float(res.pvalue))
    if x.size > 5000:
        r.note("n > 5000: the p-value of Shapiro–Wilk is unreliable for very large samples; use a Q-Q plot / effect sizes instead.")
    r.values.update(dict(W=float(res.statistic), p=float(res.pvalue)))
    return r


@analysis("shapiro_francia", "Shapiro–Francia normality test", CAT, [X],
          code={"stata": "sfrancia {x}", "r": "# install.packages('nortest'); nortest::sf.test(df${x})", "python": "# statsmith.stats.normality.shapiro_francia(x)"},
          reference="Shapiro & Francia (1972): squared correlation of the ordered data with Blom normal scores; p via Royston (1993) normal approximation, 5 ≤ n ≤ 5000.")
def shapiro_francia_test(d):
    x = clean(d["x"])
    need(5 <= x.size <= 5000, "Shapiro–Francia is valid for 5 ≤ n ≤ 5000")
    need(np.ptp(x) > 0, "the variable is constant")
    w, p, z = shapiro_francia(x)
    r = _norm_report(d, "Shapiro–Francia test", "W' statistic", w, p, [("z (Royston)", z)])
    r.values.update(dict(W=w, p=p, z=z))
    return r


@analysis("ks_normal", "Kolmogorov–Smirnov test vs a fully specified distribution", CAT, [X],
          [Param("dist", "Distribution", "choice", "norm", choices=("norm", "uniform", "expon", "lognorm", "gamma", "t", "chi2")),
           Param("loc", "Location / mean / lower bound", "float", 0.0), Param("scale", "Scale / sd / width (0 = estimate from data for norm)", "float", 1.0),
           Param("shape", "Shape / df (if applicable)", "float", 1.0),
           Param("method", "p-value method", "choice", "auto", choices=("auto", "exact", "asymp"))],
          code={"stata": "ksmirnov {x} = normal(({x} - {loc})/{scale})", "r": "ks.test(df${x}, \"p{dist}\", {loc}, {scale})",
                "python": "import scipy.stats as st; st.kstest(df['{x}'].dropna(), '{dist}', args=({loc}, {scale}))"},
          reference="D = sup|F_n(x) - F0(x)|. Parameters must be specified IN ADVANCE: if they are estimated from the same data the p-value is "
                    "conservative — use Lilliefors' test for normality with estimated mean/sd.")
def ks_normal(d, dist="norm", loc=0.0, scale=1.0, shape=1.0, method="auto"):
    x = clean(d["x"])
    need(x.size >= 2, "need at least 2 values")
    args = {"norm": (loc, scale), "uniform": (loc, scale), "expon": (loc, scale), "lognorm": (shape, loc, scale), "gamma": (shape, loc, scale),
            "t": (shape, loc, scale), "chi2": (shape, loc, scale)}[dist]
    if dist == "norm" and scale <= 0:
        args = (x.mean(), x.std(ddof=1))
    res = stats.kstest(x, dist, args=args, method=method)
    r = Result(f"One-sample Kolmogorov–Smirnov test: {d.name('x')} vs {dist}{tuple(round(float(a), 4) for a in args)}")
    r.text("Hypotheses", "H0: the data follow the specified distribution (parameters known in advance)")
    r.kv("Test", [("n", x.size), ("D statistic", float(res.statistic)), ("p-value", float(res.pvalue))])
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(D=float(res.statistic), p=float(res.pvalue)))
    return r


@analysis("lilliefors", "Lilliefors normality test", CAT, [X],
          [Param("n_mc", "Monte-Carlo samples for p (0 = analytic approx.)", "int", 0, min=0, max=1_000_000), Param("seed", "Random seed", "int", 12345)],
          code={"stata": "* Lilliefors: use R/Python", "r": "# install.packages('nortest'); nortest::lillie.test(df${x})",
                "python": "from statsmodels.stats.diagnostic import lilliefors; lilliefors(df['{x}'].dropna(), dist='norm')"},
          reference="Lilliefors (1967): KS statistic with mean and sd estimated from the data; p-value by Dallal & Wilkinson (1986) approximation "
                    "(as R nortest) or by Monte-Carlo simulation of the null distribution.")
def lilliefors(d, n_mc=0, seed=12345):
    x = clean(d["x"])
    need(x.size >= 5, "need at least 5 values")
    need(np.ptp(x) > 0, "the variable is constant")
    D = lilliefors_stat(x)
    p = lilliefors_p(D, x.size)
    extra = []
    if n_mc:
        rng = np.random.default_rng(int(seed))
        sims = np.array([lilliefors_stat(rng.standard_normal(x.size)) for _ in range(int(n_mc))])
        pm = float((np.sum(sims >= D) + 1) / (n_mc + 1))
        extra.append(("p-value (Monte-Carlo)", pm))
        p = pm
    r = _norm_report(d, "Lilliefors test", "D statistic", D, p, extra)
    r.values.update(dict(D=D, p=p))
    return r


@analysis("anderson_darling", "Anderson–Darling normality test", CAT, [X],
          code={"stata": "* Anderson-Darling: use R/Python", "r": "# install.packages('nortest'); nortest::ad.test(df${x})",
                "python": "from statsmodels.stats.diagnostic import normal_ad; normal_ad(df['{x}'].dropna())"},
          reference="A²* = A²(1 + 0.75/n + 2.25/n²); p-value by D'Agostino & Stephens (1986) piecewise formula (R nortest::ad.test).")
def anderson_darling(d):
    from statsmodels.stats.diagnostic import normal_ad
    x = clean(d["x"])
    need(x.size >= 8, "need at least 8 values")
    need(np.ptp(x) > 0, "the variable is constant")
    A2, p = normal_ad(x)                      # statsmodels returns the raw A^2 (mean & sd estimated, sd with n-1)
    n = x.size
    A2s = A2 * (1 + 0.75 / n + 2.25 / n ** 2)  # Stephens' small-sample modification A^2*
    r = _norm_report(d, "Anderson–Darling test", "A² (raw)", float(A2), float(p), [("modified A²*", float(A2s))])
    r.values.update(dict(A2=float(A2), A2star=float(A2s), p=float(p)))
    return r


@analysis("jarque_bera", "Jarque–Bera normality test", CAT, [X],
          code={"stata": "sktest {x}", "r": "# install.packages('tseries'); tseries::jarque.bera.test(df${x})", "python": "import scipy.stats as st; st.jarque_bera(df['{x}'].dropna())"},
          reference="JB = n/6·(S² + K²/4) ~ χ²(2), S = skewness (biased), K = excess kurtosis (biased).")
def jarque_bera(d):
    x = clean(d["x"])
    need(x.size >= 8, "need at least 8 values")
    res = stats.jarque_bera(x)
    r = _norm_report(d, "Jarque–Bera test", "JB statistic", float(res.statistic), float(res.pvalue),
                     [("skewness", float(stats.skew(x))), ("excess kurtosis", float(stats.kurtosis(x)))])
    r.values.update(dict(JB=float(res.statistic), p=float(res.pvalue)))
    return r


@analysis("dagostino", "D'Agostino–Pearson omnibus normality test", CAT, [X],
          code={"stata": "sktest {x}", "r": "# install.packages('moments'); moments::agostino.test(df${x}); moments::anscombe.test(df${x})", "python": "import scipy.stats as st; st.normaltest(df['{x}'].dropna())"},
          reference="K² = Z(skewness)² + Z(kurtosis)² ~ χ²(2) (D'Agostino 1971, D'Agostino & Pearson 1973), n ≥ 20 recommended.")
def dagostino(d):
    x = clean(d["x"])
    need(x.size >= 8, "need at least 8 values (n ≥ 20 recommended)")
    res = stats.normaltest(x)
    r = _norm_report(d, "D'Agostino–Pearson K² test", "K² statistic", float(res.statistic), float(res.pvalue),
                     [("z(skewness)", float(stats.skewtest(x).statistic)), ("z(kurtosis)", float(stats.kurtosistest(x).statistic))])
    r.values.update(dict(K2=float(res.statistic), p=float(res.pvalue)))
    return r
