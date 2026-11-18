"""Time-series, trend, randomness, outlier and process-quality analyses. Rows are assumed to be in time order."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..charts.base import Param, Role
from .base import Result, analysis, clean
from .common import ALPHA, CONF, X, need

CAT = "Time series & quality"
LAGS = Param("lags", "Number of lags", "int", 20, min=1, max=500)


def series(d, role="x"):
    x = np.asarray(d[role], dtype=float)
    x = x[np.isfinite(x)]
    return x


def acf_values(x, nlags):
    x = x - x.mean()
    n = len(x)
    c0 = float(x @ x)
    return np.array([1.0] + [float(x[:-k] @ x[k:]) / c0 for k in range(1, nlags + 1)])


# =========================================================================== ACF / PACF + Ljung-Box
@analysis("acf_pacf", "Autocorrelation (ACF) & partial autocorrelation (PACF)", CAT, [X], [LAGS],
          code={"stata": "ac {x}, lags({lags})\npac {x}, lags({lags})", "r": "acf(df${x_r}, lag.max = {lags}, plot = FALSE); pacf(df${x_r}, lag.max = {lags}, plot = FALSE)",
                "python": "from statsmodels.tsa.stattools import acf, pacf; acf(df['{x}'].dropna(), nlags={lags}); pacf(df['{x}'].dropna(), nlags={lags})"},
          reference="ACF r_k = Σ(x_t−x̄)(x_{t+k}−x̄)/Σ(x_t−x̄)² (same as R acf). PACF by Yule–Walker (statsmodels, as R pacf). Bands ±1.96/√n.")
def acf_pacf(d, lags=20):
    from statsmodels.tsa.stattools import pacf
    x = series(d)
    need(x.size >= 10, "need at least 10 observations")
    lags = min(lags, x.size // 2 - 1)
    a = acf_values(x, lags)
    pa = pacf(x, nlags=lags, method="ywm")
    band = 1.96 / np.sqrt(x.size)
    r = Result(f"Autocorrelation of {d.name('x')}")
    r.kv("", [("n", x.size), ("lags", lags), ("±95% band", band)])
    r.table("ACF / PACF", ["Lag", "ACF", "PACF", "significant?"], [[k, float(a[k]), float(pa[k]), "*" if abs(a[k]) > band or abs(pa[k]) > band else ""] for k in range(1, lags + 1)])
    r.values.update(dict(acf=a, pacf=pa, band=band))
    return r


@analysis("ljung_box", "Ljung–Box / Box–Pierce portmanteau test", CAT, [X], [LAGS, Param("fitdf", "Degrees of freedom used by a fitted model", "int", 0, min=0, max=50)],
          code={"stata": "wntestq {x}, lags({lags})", "r": "Box.test(df${x_r}, lag = {lags}, type = 'Ljung-Box', fitdf = {fitdf})",
                "python": "from statsmodels.stats.diagnostic import acorr_ljungbox; acorr_ljungbox(df['{x}'].dropna(), lags=[{lags}], model_df={fitdf})"},
          reference="Q_LB = n(n+2) Σ_{k=1}^{h} r_k²/(n−k) ~ χ²(h − fitdf); Box–Pierce Q = n Σ r_k². H0: no autocorrelation up to lag h.")
def ljung_box(d, lags=20, fitdf=0):
    x = series(d)
    need(x.size > lags + 2 and lags > fitdf, "need more observations than lags, and lags > fitdf")
    a = acf_values(x, lags)[1:]
    n = x.size
    k = np.arange(1, lags + 1)
    q_lb = float(n * (n + 2) * np.sum(a ** 2 / (n - k)))
    q_bp = float(n * np.sum(a ** 2))
    df = lags - fitdf
    r = Result(f"Portmanteau test for autocorrelation: {d.name('x')}")
    r.table("", ["Test", "Q", "df", "p-value"], [["Ljung–Box", q_lb, df, float(stats.chi2.sf(q_lb, df))], ["Box–Pierce", q_bp, df, float(stats.chi2.sf(q_bp, df))]])
    r.values.update(dict(Q=q_lb, p=float(stats.chi2.sf(q_lb, df)), Q_bp=q_bp))
    return r


# =========================================================================== unit root / stationarity
@analysis("adf", "Augmented Dickey–Fuller unit-root test", CAT, [X],
          [Param("regression", "Deterministic terms", "choice", "c", choices=("n", "c", "ct")), Param("maxlag", "Max lags (blank 0 = automatic AIC)", "int", 0, min=0, max=100)],
          code={"stata": "dfuller {x}, lags({maxlag})", "r": "tseries::adf.test(df${x_r})", "python": "from statsmodels.tsa.stattools import adfuller; adfuller(df['{x}'].dropna(), regression='{regression}', autolag='AIC')"},
          reference="statsmodels adfuller with MacKinnon (1994/2010) p-values. H0: unit root (non-stationary). 'c' = constant, 'ct' = constant + trend, 'n' = none.")
def adf(d, regression="c", maxlag=0):
    from statsmodels.tsa.stattools import adfuller
    x = series(d)
    need(x.size >= 12, "need at least 12 observations")
    res = adfuller(x, regression=regression, maxlag=maxlag or None, autolag="AIC" if not maxlag else None, result_object=False)
    r = Result(f"Augmented Dickey–Fuller test: {d.name('x')}")
    r.kv("", [("test statistic", float(res[0])), ("p-value (MacKinnon)", float(res[1])), ("lags used", int(res[2])), ("observations", int(res[3]))] +
         [(f"critical value {k}", float(v)) for k, v in res[4].items()])
    r.note("Reject H0 (unit root) when the statistic is more negative than the critical value / p is small -> evidence of stationarity.")
    r.values.update(dict(stat=float(res[0]), p=float(res[1])))
    return r


@analysis("kpss", "KPSS stationarity test", CAT, [X], [Param("regression", "Null hypothesis", "choice", "c", choices=("c", "ct"))],
          code={"stata": "* kpss {x} (user-written: ssc install kpss)", "r": "tseries::kpss.test(df${x_r}, null = '{regression}')", "python": "from statsmodels.tsa.stattools import kpss; kpss(df['{x}'].dropna(), regression='{regression}', nlags='auto')"},
          reference="Kwiatkowski–Phillips–Schmidt–Shin (1992); H0: series is level ('c') / trend ('ct') stationary. p-value interpolated from published tables (bounded to 0.01–0.10).")
def kpss(d, regression="c"):
    import warnings
    from statsmodels.tsa.stattools import kpss as _kpss
    x = series(d)
    need(x.size >= 12, "need at least 12 observations")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, lags, crit = _kpss(x, regression=regression, nlags="auto")
    r = Result(f"KPSS test: {d.name('x')}")
    r.kv("", [("test statistic", float(stat)), ("p-value (interpolated)", float(p)), ("lags", int(lags))] + [(f"critical value {k}", float(v)) for k, v in crit.items()])
    r.note("Small p -> reject stationarity. Use together with ADF: stationary if ADF rejects and KPSS does not.")
    r.values.update(dict(stat=float(stat), p=float(p)))
    return r


# =========================================================================== ARIMA
@analysis("arima", "ARIMA model (fit + forecast)", CAT, [X],
          [Param("p", "AR order p", "int", 1, min=0, max=10), Param("d", "Differences d", "int", 0, min=0, max=3), Param("q", "MA order q", "int", 0, min=0, max=10),
           Param("steps", "Forecast steps", "int", 5, min=0, max=500)],
          code={"stata": "arima {x}, arima({p},{d},{q})\npredict fc, dynamic(.)", "r": "fit <- arima(df${x_r}, order = c({p}, {d}, {q})); fit; predict(fit, n.ahead = {steps})",
                "python": "from statsmodels.tsa.arima.model import ARIMA; fit = ARIMA(df['{x}'].dropna().values, order=({p},{d},{q})).fit(); print(fit.summary()); fit.forecast({steps})"},
          reference="Maximum-likelihood ARIMA(p,d,q) with constant when d = 0 (statsmodels state-space, same likelihood as R arima / Stata arima; "
                    "small numerical differences in coefficients are normal).")
def arima(data, p=1, d=0, q=0, steps=5):
    from statsmodels.tsa.arima.model import ARIMA
    dd = d
    x = series(data)
    need(x.size >= 3 * (p + q + dd + 1) + 5, "too few observations for this model order")
    fit = ARIMA(x, order=(p, dd, q), trend="c" if dd == 0 else "n").fit()
    r = Result(f"ARIMA({p},{dd},{q}): {data.name('x')}")
    r.kv("", [("observations", x.size), ("log-likelihood", float(fit.llf)), ("AIC", float(fit.aic)), ("BIC", float(fit.bic))])
    names = list(fit.param_names)
    r.table("Coefficients", ["Term", "Coef.", "Std. err.", "z", "p-value"], [[names[i], float(fit.params[i]), float(fit.bse[i]), float(fit.tvalues[i]), float(fit.pvalues[i])] for i in range(len(names))])
    if steps:
        fc = fit.get_forecast(steps)
        ci = fc.conf_int()
        r.table("Forecast (95% interval)", ["Step", "Forecast", "Lower", "Upper"], [[i + 1, float(fc.predicted_mean[i]), float(ci[i, 0]), float(ci[i, 1])] for i in range(steps)])
    r.values.update(dict(aic=float(fit.aic), bic=float(fit.bic), params=dict(zip(names, map(float, fit.params)))))
    return r


# =========================================================================== runs test
@analysis("runs_test", "Wald–Wolfowitz runs test (randomness)", CAT, [X], [Param("cut", "Cut point", "choice", "median", choices=("median", "mean"))],
          code={"stata": "runtest {x}, thresh(.)", "r": "tseries::runs.test(factor(df${x_r} > median(df${x_r}, na.rm = TRUE)))",
                "python": "# runs on sign(x - median): R = 1 + sum(s[1:] != s[:-1]); z = (R - mu)/sd  (see Methods)"},
          reference="Runs of values above/below the cut (values equal to the cut dropped). μ = 2n₁n₂/n + 1, σ² = (μ−1)(μ−2)/(n−1), z = (R−μ)/σ (Wald & Wolfowitz 1940).")
def runs_test(d, cut="median"):
    x = series(d)
    c = np.median(x) if cut == "median" else x.mean()
    s = x[x != c] > c
    n1, n2 = int(s.sum()), int((~s).sum())
    need(n1 > 0 and n2 > 0 and s.size >= 10, "need at least 10 values on both sides of the cut point")
    n = n1 + n2
    R = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2 * n1 * n2 / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1)
    z = (R - mu) / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    r = Result(f"Runs test for randomness: {d.name('x')}")
    r.kv("", [("cut point", float(c)), ("n above", n1), ("n below", n2), ("runs", R), ("expected runs", mu), ("z", float(z)), ("p-value (two-sided)", p)])
    r.values.update(dict(R=R, mu=mu, z=float(z), p=p))
    return r


# =========================================================================== Mann–Kendall
@analysis("mann_kendall", "Mann–Kendall trend test (+ Sen's slope)", CAT, [X],
          code={"stata": "* Mann-Kendall: user-written  (ssc install mkendall)", "r": "Kendall::MannKendall(df${x_r}); trend::sens.slope(na.omit(df${x_r}))",
                "python": "import scipy.stats as st; st.kendalltau(range(len(v)), v)  # tau-b equivalent of MK test; Sen: st.theilslopes(v)"},
          reference="S = Σ_{i<j} sign(x_j − x_i); Var(S) = [n(n−1)(2n+5) − Σt(t−1)(2t+5)]/18; Z = (S ∓ 1)/√Var (continuity corrected). "
                    "Sen's slope = median of all pairwise slopes (scipy.stats.theilslopes). Series > 5,000 values are summarised on an evenly spaced subsample.")
def mann_kendall(d):
    x = series(d)
    need(x.size >= 8, "need at least 8 observations")
    note = None
    if x.size > 5000:
        x = x[np.linspace(0, x.size - 1, 5000).astype(int)]
        note = "series longer than 5,000: evenly spaced 5,000-point subsample used"
    n = x.size
    S = 0
    for i in range(n - 1):
        S += int(np.sign(x[i + 1:] - x[i]).sum())
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    z = 0.0 if S == 0 else (S - np.sign(S)) / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    slope, icpt, lo, hi = stats.theilslopes(x, np.arange(n))
    r = Result(f"Mann–Kendall trend test: {d.name('x')}")
    r.kv("", [("n", n), ("S", S), ("Var(S)", float(var)), ("Z", float(z)), ("p-value (two-sided)", p), ("Kendall tau", float(S / (n * (n - 1) / 2))),
              ("Sen's slope per step", float(slope)), ("Sen's slope 95% CI lower", float(lo)), ("Sen's slope 95% CI upper", float(hi))])
    r.note("Positive S / Z = increasing trend." + (" " + note if note else ""))
    r.values.update(dict(S=S, z=float(z), p=p, sen=float(slope)))
    return r


# =========================================================================== Grubbs outlier
@analysis("grubbs", "Grubbs' test for a single outlier", CAT, [X], [ALPHA],
          code={"stata": "* grubbs: user-written (ssc install grubbs)", "r": "outliers::grubbs.test(df${x_r}, two.sided = TRUE)",
                "python": "# G = max|x-mean|/s ; see Methods for the t-based critical value"},
          reference="G = max|x_i − x̄|/s. t² = nG²... exact p via t = G·√[n(n−2)/((n−1)² − nG²)] ~ t(n−2), p = min(1, 2n·P(T > t)) (Grubbs 1969).")
def grubbs(d, alpha=0.05):
    x = series(d)
    n = x.size
    need(n >= 3, "need at least 3 values")
    m, s = x.mean(), x.std(ddof=1)
    idx = int(np.argmax(np.abs(x - m)))
    G = abs(x[idx] - m) / s
    den = (n - 1) ** 2 - n * G ** 2
    t = G * np.sqrt(n * (n - 2) / den) if den > 0 else np.inf
    p = float(min(1.0, 2 * n * stats.t.sf(t, n - 2)))
    tc = stats.t.isf(alpha / (2 * n), n - 2)
    Gc = (n - 1) / np.sqrt(n) * np.sqrt(tc ** 2 / (n - 2 + tc ** 2))
    r = Result(f"Grubbs' test: {d.name('x')}")
    r.kv("", [("n", n), ("suspect value", float(x[idx])), ("suspect position (1-based)", idx + 1), ("G", float(G)), (f"critical G (α = {alpha:g})", float(Gc)), ("p-value (two-sided)", p)])
    r.note("H0: no outliers; assumes the remaining data are normal. Apply repeatedly with care.")
    r.values.update(dict(G=float(G), Gcrit=float(Gc), p=p, index=idx))
    return r


# =========================================================================== process capability
@analysis("capability", "Process capability (Cp, Cpk, Pp, Ppk)", CAT, [X],
          [Param("lsl", "Lower spec limit (LSL)", "float", 0.0), Param("usl", "Upper spec limit (USL)", "float", 1.0)],
          code={"stata": "* capability: see Python/R", "r": "qcc::process.capability(qcc::qcc(matrix(df${x_r}, ncol = 1), type = 'xbar.one', plot = FALSE), spec.limits = c({lsl}, {usl}))",
                "python": "# Cpk = min(USL-mean, mean-LSL)/(3*sigma_within); sigma_within = mean(|diff(x)|)/1.128"},
          reference="Overall (Pp/Ppk) uses the sample SD; within (Cp/Cpk) uses σ̂ = MR̄/1.128 (moving range of 2). Cp = (USL−LSL)/6σ, Cpk = min(USL−μ, μ−LSL)/3σ.")
def capability(d, lsl=0.0, usl=1.0):
    x = series(d)
    need(x.size >= 10 and usl > lsl, "need at least 10 values and USL > LSL")
    m, s = x.mean(), x.std(ddof=1)
    sw = np.mean(np.abs(np.diff(x))) / 1.128
    f = lambda sig: ((usl - lsl) / (6 * sig), min(usl - m, m - lsl) / (3 * sig))
    cp, cpk = f(sw)
    pp, ppk = f(s)
    out = float(np.mean((x < lsl) | (x > usl)) * 100)
    r = Result(f"Process capability: {d.name('x')}")
    r.kv("", [("n", x.size), ("mean", float(m)), ("SD overall", float(s)), ("SD within (MR̄/1.128)", float(sw)), ("LSL", lsl), ("USL", usl),
              ("Cp", float(cp)), ("Cpk", float(cpk)), ("Pp", float(pp)), ("Ppk", float(ppk)), ("observed % out of spec", out),
              ("expected % out of spec (overall, normal)", float((stats.norm.cdf(lsl, m, s) + stats.norm.sf(usl, m, s)) * 100))])
    r.values.update(dict(cp=float(cp), cpk=float(cpk), pp=float(pp), ppk=float(ppk)))
    return r


@analysis("control_limits", "Individuals (I-MR) control chart limits", CAT, [X],
          code={"stata": "* shewhart {x}: see Stata `shewhart` or R qcc", "r": "qcc::qcc(df${x_r}, type = 'xbar.one')", "python": "# CL = mean; sigma = mean(|diff|)/1.128; UCL/LCL = CL ± 3 sigma"},
          reference="σ̂ = MR̄/d₂ (d₂ = 1.128); UCL/LCL = x̄ ± 3σ̂; MR chart UCL = 3.267·MR̄. Points flagged: beyond 3σ (Nelson rule 1) and 8 in a row on one side (rule 2).")
def control_limits(d):
    x = series(d)
    need(x.size >= 10, "need at least 10 values")
    cl = x.mean()
    mr = np.abs(np.diff(x))
    sig = mr.mean() / 1.128
    ucl, lcl = cl + 3 * sig, cl - 3 * sig
    out1 = np.flatnonzero((x > ucl) | (x < lcl))
    side = np.sign(x - cl)
    out2 = [i for i in range(7, len(x)) if abs(side[i - 7:i + 1].sum()) == 8]
    r = Result(f"Individuals control chart limits: {d.name('x')}")
    r.kv("", [("n", x.size), ("centre line", float(cl)), ("σ̂ (MR̄/1.128)", float(sig)), ("UCL", float(ucl)), ("LCL", float(lcl)),
              ("MR̄", float(mr.mean())), ("MR UCL", float(3.267 * mr.mean())), ("points beyond limits (rule 1)", int(out1.size)), ("end points of 8-in-a-row runs (rule 2)", len(out2))])
    if out1.size:
        r.text("Out-of-control points (1-based positions)", ", ".join(str(int(i) + 1) for i in out1[:50]))
    r.values.update(dict(cl=float(cl), ucl=float(ucl), lcl=float(lcl), out=out1))
    return r
