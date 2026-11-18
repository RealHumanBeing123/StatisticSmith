"""Regression & modelling: OLS (multiple), logistic, Poisson, polynomial, Chow test, diagnostics."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..charts.base import Param, Role
from .base import Result, analysis, clean
from .common import ALPHA, CONF, need, sig_note

CAT = "Regression & modelling"

XM = Role("xs", "numeric_multi", "Predictors (numeric)")
FACT = Role("factors", "any_multi", "Categorical predictors (dummy-coded)", required=False)
YR = Role("y", "numeric", "Response")


def _frame(d, y_role="y", need_y=True, allow_factors=True, y_override=None):
    """DataFrame with unique safe column names + mapping back to display names."""
    import pandas as pd
    cols, disp = {}, {}
    if need_y:
        cols["Y"] = np.asarray(d[y_role] if y_override is None else y_override, dtype=object)
        disp["Y"] = d.name(y_role)
    xs = d["xs"] if isinstance(d.get("xs"), list) else ([d["xs"]] if d.has("xs") else [])
    xn = d.label_list("xs") if d.has("xs") else []
    xkeys = []
    for i, (c, n) in enumerate(zip(xs, xn)):
        k = f"X{i + 1}"
        cols[k] = np.asarray(c, dtype=float)
        disp[k] = n
        xkeys.append(k)
    fkeys = []
    if allow_factors and d.has("factors"):
        fs = d["factors"] if isinstance(d["factors"], list) else [d["factors"]]
        fn = d.label_list("factors")
        for i, (c, n) in enumerate(zip(fs, fn)):
            k = f"F{i + 1}"
            cols[k] = np.array(["" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in np.asarray(c, dtype=object).tolist()], dtype=object)
            disp[k] = n
            fkeys.append(k)
    df = pd.DataFrame(cols)
    if need_y:
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    for k in fkeys:
        df.loc[df[k] == "", k] = np.nan
    df = df.dropna()
    return df, disp, xkeys, fkeys


def _pretty(term, disp):
    import re
    if term == "Intercept":
        return "(intercept)"
    t = re.sub(r"C\((F\d+)\)\[T\.(.*?)\]", lambda m: f"{disp[m.group(1)]} = {m.group(2)}", term)
    for k, v in disp.items():
        t = re.sub(rf"\b{k}\b", v, t)
    return t


def _formula(xkeys, fkeys, extra=""):
    parts = xkeys + [f"C({k})" for k in fkeys]
    return "Y ~ " + (" + ".join(parts) if parts else "1") + extra


def vif_values(Xmat: np.ndarray, names) -> list[tuple[str, float]]:
    """Variance inflation factors: VIF_j = 1/(1 - R²_j) regressing x_j on the other predictors (with intercept)."""
    out = []
    n, k = Xmat.shape
    for j in range(k):
        yj = Xmat[:, j]
        others = np.delete(Xmat, j, axis=1)
        A = np.column_stack([np.ones(n), others]) if others.size else np.ones((n, 1))
        beta, *_ = np.linalg.lstsq(A, yj, rcond=None)
        resid = yj - A @ beta
        sst = ((yj - yj.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / sst if sst > 0 else 1.0
        out.append((names[j], float(1 / (1 - r2)) if r2 < 1 else float("inf")))
    return out


@analysis("ols", "Linear regression (simple / multiple)", CAT, [YR, XM, FACT],
          [Param("robust", "Standard errors", "choice", "classical", choices=("classical", "HC1", "HC3")), CONF,
           Param("diagnostics", "Residual diagnostics (DW, Breusch–Pagan, normality, VIF)", "bool", True),
           Param("standardized", "Show standardised coefficients (β)", "bool", True)],
          code={"stata": "regress {y} {xs}{factors_stata}{robust_stata}", "r": "fit <- lm({y_r} ~ {xs_plus}{factors_r}, data = df); summary(fit); confint(fit, level = {conf})",
                "python": "import statsmodels.formula.api as smf; fit = smf.ols('{y} ~ {xs_plus}{factors_py}', df).fit(); print(fit.summary())"},
          reference="OLS via statsmodels (QR/pinv). t = b/se(b), CI = b ± t(1-α/2, n-k-1)·se. HC1/HC3: MacKinnon–White heteroskedasticity-robust SEs. "
                    "VIF_j = 1/(1-R_j²). Durbin–Watson, Breusch–Pagan (studentised) and Jarque–Bera on residuals.")
def ols(d, robust="classical", conf=0.95, diagnostics=True, standardized=True):
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson
    df, disp, xkeys, fkeys = _frame(d)
    need(xkeys or fkeys, "choose at least one predictor")
    k = len(xkeys) + sum(df[f].nunique() - 1 for f in fkeys)
    need(len(df) > k + 1, f"not enough complete rows ({len(df)}) for {k} predictors")
    fit = smf.ols(_formula(xkeys, fkeys), df).fit() if robust == "classical" else smf.ols(_formula(xkeys, fkeys), df).fit(cov_type=robust)
    ci = fit.conf_int(alpha=1 - conf)
    names = list(fit.params.index)
    ysd = df["Y"].std(ddof=1)
    r = Result(f"Linear regression: {disp['Y']} ~ " + " + ".join([disp[k_] for k_ in xkeys + fkeys]))
    r.kv("Model summary", [("observations", int(fit.nobs)), ("R²", float(fit.rsquared)), ("adjusted R²", float(fit.rsquared_adj)),
                           ("F statistic", float(fit.fvalue)), ("df (model, residual)", f"{int(fit.df_model)}, {int(fit.df_resid)}"),
                           ("Prob > F", float(fit.f_pvalue)), ("root MSE (residual std.err)", float(np.sqrt(fit.mse_resid))),
                           ("AIC", float(fit.aic)), ("BIC", float(fit.bic)), ("log-likelihood", float(fit.llf))])
    rows = []
    for i, nm in enumerate(names):
        row = [_pretty(nm, disp), float(fit.params[nm]), float(fit.bse[nm]), float(fit.tvalues[nm]), float(fit.pvalues[nm]), float(ci.iloc[i, 0]), float(ci.iloc[i, 1])]
        if standardized:
            sx = df[nm].std(ddof=1) if nm in xkeys else None
            row.append(float(fit.params[nm] * sx / ysd) if sx else "")
        rows.append(row)
    heads = ["term", "coef", "std.err", "t", "P>|t|", f"{conf:.0%} CI low", f"{conf:.0%} CI high"] + (["std. β"] if standardized else [])
    r.table(f"Coefficients ({'classical' if robust == 'classical' else robust + ' robust'} standard errors)", heads, rows)
    if diagnostics and xkeys:
        resid = np.asarray(fit.resid)
        dg = [("Durbin–Watson", float(durbin_watson(resid)))]
        try:
            lm, lmp, fval, fp = het_breuschpagan(resid, fit.model.exog)
            dg += [("Breusch–Pagan LM", float(lm)), ("Breusch–Pagan p", float(lmp))]
        except Exception:
            pass
        jb = stats.jarque_bera(resid)
        dg += [("Jarque–Bera (residuals)", float(jb.statistic)), ("Jarque–Bera p", float(jb.pvalue)), ("residual skewness", float(stats.skew(resid))),
               ("residual excess kurtosis", float(stats.kurtosis(resid)))]
        r.kv("Residual diagnostics", dg)
        if len(xkeys) >= 2:
            r.table("Variance inflation factors (numeric predictors)", ["predictor", "VIF"], [[disp[n], v] for n, v in vif_values(df[xkeys].to_numpy(float), xkeys)])
            r.note("VIF above ~5-10 signals problematic multicollinearity.")
    r.values.update(dict(params=fit.params.to_dict(), bse=fit.bse.to_dict(), pvalues=fit.pvalues.to_dict(), r2=float(fit.rsquared), adj_r2=float(fit.rsquared_adj),
                         F=float(fit.fvalue), f_p=float(fit.f_pvalue), nobs=int(fit.nobs), rmse=float(np.sqrt(fit.mse_resid)), aic=float(fit.aic),
                         bic=float(fit.bic), resid=np.asarray(fit.resid), fitted=np.asarray(fit.fittedvalues), names=names))
    return r


@analysis("poly_reg", "Polynomial regression", CAT, [YR, Role("x", "numeric", "Predictor")], [Param("degree", "Degree", "int", 2, min=1, max=10), CONF],
          code={"stata": "regress {y} c.{x}##c.{x}   /* degree 2; add more ## terms for higher degrees */", "r": "lm({y_r} ~ poly({x_r}, {degree}, raw = TRUE), data = df)",
                "python": "import numpy as np; np.polyfit(df['{x}'], df['{y}'], {degree})"},
          reference="Raw (non-orthogonal) polynomial terms x, x², …, x^p fitted by OLS (centred internally for numerical stability).")
def poly_reg(d, degree=2, conf=0.95):
    import statsmodels.api as sm
    x = np.asarray(d["x"], dtype=float)
    y = np.asarray(d["y"], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    need(x.size > degree + 1, "not enough observations")
    xm = x.mean()
    Z = np.column_stack([(x - xm) ** p for p in range(1, degree + 1)])
    fit = sm.OLS(y, sm.add_constant(Z)).fit()
    # transform back to raw powers of x
    coef_c = fit.params
    raw = np.polynomial.polynomial.polyfit(x, y, degree)
    ci = fit.conf_int(alpha=1 - conf)
    r = Result(f"Polynomial regression (degree {degree}): {d.name('y')} ~ {d.name('x')}")
    r.kv("Fit", [("n", x.size), ("R²", float(fit.rsquared)), ("adjusted R²", float(fit.rsquared_adj)), ("F", float(fit.fvalue)), ("Prob > F", float(fit.f_pvalue)),
                 ("root MSE", float(np.sqrt(fit.mse_resid)))])
    names = ["intercept"] + [f"(x-{xm:.4g})^{p}" for p in range(1, degree + 1)]
    r.table("Coefficients of the polynomial in (x − mean(x)) — stable parametrisation",
            ["term", "coef", "std.err", "t", "P>|t|", f"{conf:.0%} CI low", f"{conf:.0%} CI high"],
            [[names[i], float(fit.params[i]), float(fit.bse[i]), float(fit.tvalues[i]), float(fit.pvalues[i]),
              float(ci[i, 0]), float(ci[i, 1])] for i in range(degree + 1)])
    r.text("Equivalent raw-power polynomial", "y = " + " + ".join(f"{c:.6g}·x^{p}" if p else f"{c:.6g}" for p, c in enumerate(raw)))
    r.values.update(dict(r2=float(fit.rsquared), raw_coef=raw, coef_centered=np.asarray(coef_c)))
    return r


@analysis("logit", "Logistic regression (binary outcome)", CAT, [Role("y", "any", "Binary outcome"), XM, FACT],
          [Param("success", "Value counted as 1 (blank = last level)", "text", ""), CONF],
          code={"stata": "logit {y} {xs}{factors_stata}, or", "r": "fit <- glm({y_r} ~ {xs_plus}{factors_r}, data = df, family = binomial); summary(fit); exp(cbind(OR = coef(fit), confint.default(fit)))",
                "python": "import statsmodels.formula.api as smf; smf.logit('{y} ~ {xs_plus}{factors_py}', df).fit().summary()"},
          reference="Maximum likelihood (IRLS/Newton) via statsmodels; Wald z-tests; OR = exp(b). McFadden pseudo-R² = 1 - LL/LL0; LR χ² test of the whole model.")
def logit(d, success="", conf=0.95):
    import statsmodels.formula.api as smf
    from .location import _binary
    ycol = np.asarray(d["y"], dtype=object)
    present = np.array([not (v is None or (isinstance(v, float) and np.isnan(v))) for v in ycol.tolist()])
    s, suc = _binary(ycol[present], success)
    y01 = np.full(len(ycol), np.nan)
    y01[present] = s.astype(float)
    D, disp, xk, fk = _frame(d, y_override=y01)
    need(len(D) > 5, "not enough complete rows")
    need(D["Y"].nunique() == 2, "the outcome needs both values (0 and 1) present among complete rows")
    need(xk or fk, "choose at least one predictor")
    fit = smf.logit(_formula(xk, fk), D).fit(disp=False, maxiter=200)
    ci = fit.conf_int(alpha=1 - conf)
    names = list(fit.params.index)
    r = Result(f"Logistic regression: P({d.name('y')} = {suc}) ~ " + " + ".join([disp[k] for k in xk + fk]))
    r.kv("Model summary", [("observations", int(fit.nobs)), ("events (Y=1)", int(D["Y"].sum())), ("LR χ²", float(fit.llr)), ("df", int(fit.df_model)),
                           ("Prob > χ²", float(fit.llr_pvalue)), ("log-likelihood", float(fit.llf)), ("null log-likelihood", float(fit.llnull)),
                           ("McFadden pseudo-R²", float(fit.prsquared)), ("AIC", float(fit.aic)), ("BIC", float(fit.bic))])
    rows = [[_pretty(n, disp), float(fit.params[n]), float(fit.bse[n]), float(fit.tvalues[n]), float(fit.pvalues[n]), float(np.exp(fit.params[n])),
             float(np.exp(ci.iloc[i, 0])), float(np.exp(ci.iloc[i, 1]))] for i, n in enumerate(names)]
    r.table("Coefficients (Wald z) and odds ratios", ["term", "coef", "std.err", "z", "P>|z|", "odds ratio", f"OR {conf:.0%} low", f"OR {conf:.0%} high"], rows)
    pred = np.asarray(fit.predict(D)) >= 0.5
    yv = D["Y"].to_numpy() == 1
    r.kv("Classification at cutoff 0.5", [("accuracy", float((pred == yv).mean())), ("sensitivity", float((pred & yv).sum() / max(yv.sum(), 1))),
                                          ("specificity", float(((~pred) & (~yv)).sum() / max((~yv).sum(), 1)))])
    r.values.update(dict(params=fit.params.to_dict(), bse=fit.bse.to_dict(), pvalues=fit.pvalues.to_dict(), llf=float(fit.llf), llr=float(fit.llr),
                         prsq=float(fit.prsquared)))
    return r


@analysis("poisson", "Poisson regression (count outcome)", CAT, [YR, XM, FACT], [CONF],
          code={"stata": "poisson {y} {xs}{factors_stata}, irr", "r": "fit <- glm({y_r} ~ {xs_plus}{factors_r}, data = df, family = poisson); summary(fit); exp(coef(fit))",
                "python": "import statsmodels.formula.api as smf; smf.poisson('{y} ~ {xs_plus}{factors_py}', df).fit().summary()"},
          reference="Poisson GLM with log link (ML). IRR = exp(b). Overdispersion check: Pearson χ²/df.")
def poisson(d, conf=0.95):
    import statsmodels.formula.api as smf
    df, disp, xkeys, fkeys = _frame(d)
    need(xkeys or fkeys, "choose at least one predictor")
    need((df["Y"] >= 0).all() and np.allclose(df["Y"], np.round(df["Y"])), "the response must be non-negative integer counts")
    fit = smf.poisson(_formula(xkeys, fkeys), df).fit(disp=False, maxiter=200)
    ci = fit.conf_int(alpha=1 - conf)
    names = list(fit.params.index)
    mu = fit.predict(df)
    pearson = float((((df["Y"] - mu) ** 2) / mu).sum() / fit.df_resid)
    r = Result(f"Poisson regression: {disp['Y']} ~ " + " + ".join([disp[k] for k in xkeys + fkeys]))
    r.kv("Model summary", [("observations", int(fit.nobs)), ("LR χ²", float(fit.llr)), ("Prob > χ²", float(fit.llr_pvalue)), ("log-likelihood", float(fit.llf)),
                           ("McFadden pseudo-R²", float(fit.prsquared)), ("Pearson χ² / df (dispersion)", pearson)])
    r.table("Coefficients", ["term", "coef", "std.err", "z", "P>|z|", "IRR", f"IRR {conf:.0%} low", f"IRR {conf:.0%} high"],
            [[_pretty(n, disp), float(fit.params[n]), float(fit.bse[n]), float(fit.tvalues[n]), float(fit.pvalues[n]), float(np.exp(fit.params[n])),
              float(np.exp(ci.iloc[i, 0])), float(np.exp(ci.iloc[i, 1]))] for i, n in enumerate(names)])
    if pearson > 1.5:
        r.note(f"Dispersion {pearson:.2f} > 1.5 suggests overdispersion: consider negative-binomial regression or robust standard errors.")
    r.values.update(dict(params=fit.params.to_dict(), llf=float(fit.llf), dispersion=pearson))
    return r


@analysis("chow", "Chow test for a structural break", CAT, [YR, XM],
          [Param("break_at", "First row (1-based) of the second regime", "int", 0, min=0, max=10 ** 9, help="0 = split the data in half")],
          code={"stata": "* estat sbsingle  (or: regress ... then Chow by hand);  see R/Python", "r": "# install.packages('strucchange'); strucchange::sctest({y_r} ~ {xs_plus}, data = df, type = 'Chow', point = {break_at_r})",
                "python": "# statsmith.stats.regression.chow_F(y, X, break_at)"},
          reference="F = [(RSS_p - (RSS_1+RSS_2))/k] / [(RSS_1+RSS_2)/(n-2k)] ~ F(k, n-2k), k = parameters incl. intercept (Chow 1960).")
def chow(d, break_at=0):
    df, disp, xkeys, fkeys = _frame(d, allow_factors=False)
    need(xkeys, "choose at least one predictor")
    y = df["Y"].to_numpy(float)
    X = np.column_stack([np.ones(len(df))] + [df[k].to_numpy(float) for k in xkeys])
    n, k = X.shape
    b = int(break_at) - 1 if break_at else n // 2
    need(k < b < n - k, f"each regime needs more than {k} observations (break row {b + 1} of {n})")
    F, p, rss = chow_F(y, X, b)
    r = Result(f"Chow test: {disp['Y']} ~ " + " + ".join(disp[k_] for k_ in xkeys))
    r.text("Hypotheses", f"H0: the same regression coefficients apply before and after row {b + 1}")
    r.kv("Test", [("n", n), ("regime 1 rows", b), ("regime 2 rows", n - b), ("RSS pooled", rss[0]), ("RSS regime 1", rss[1]), ("RSS regime 2", rss[2]),
                  ("F statistic", F), ("df", f"{k}, {n - 2 * k}"), ("p-value", p)])
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(F=F, p=p, rss=rss))
    return r


def chow_F(y, X, b):
    n, k = X.shape

    def rss(yy, XX):
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        e = yy - XX @ beta
        return float(e @ e)

    rp, r1, r2 = rss(y, X), rss(y[:b], X[:b]), rss(y[b:], X[b:])
    F = ((rp - (r1 + r2)) / k) / ((r1 + r2) / (n - 2 * k))
    return float(F), float(stats.f.sf(F, k, n - 2 * k)), (rp, r1, r2)


@analysis("heterosk", "Heteroskedasticity tests (Breusch–Pagan, White)", CAT, [YR, XM],
          code={"stata": "regress {y} {xs}\nestat hettest\nestat imtest, white", "r": "# lmtest::bptest(lm({y_r} ~ {xs_plus}, data = df))",
                "python": "from statsmodels.stats.diagnostic import het_breuschpagan, het_white"},
          reference="Breusch–Pagan (studentised, Koenker 1981) and White (1980) LM tests on OLS residuals.")
def heterosk(d):
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    df, disp, xkeys, _ = _frame(d, allow_factors=False)
    need(xkeys, "choose at least one predictor")
    X = sm.add_constant(df[xkeys].to_numpy(float))
    fit = sm.OLS(df["Y"].to_numpy(float), X).fit()
    bp = het_breuschpagan(fit.resid, X)
    wh = het_white(fit.resid, X)
    r = Result(f"Heteroskedasticity tests: {disp['Y']} ~ " + " + ".join(disp[k] for k in xkeys))
    r.text("Hypotheses", "H0: residual variance is constant (homoskedastic)")
    r.table("Tests", ["test", "LM statistic", "p-value", "F statistic", "F p-value"], [["Breusch–Pagan", bp[0], bp[1], bp[2], bp[3]], ["White", wh[0], wh[1], wh[2], wh[3]]])
    r.values.update(dict(bp_p=float(bp[1]), white_p=float(wh[1])))
    return r
