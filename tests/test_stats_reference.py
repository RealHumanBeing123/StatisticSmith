"""Verification of statistical analyses against (a) published textbook / R-documentation values and
(b) independent re-computation from the defining formulas (NOT by calling the same scipy routine the code uses)."""
import numpy as np
import pytest
from scipy import stats as st

from statsmith.stats import REGISTRY, run

sleep1 = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
sleep2 = np.array([1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])
sleep_y = np.r_[sleep1, sleep2]
sleep_g = np.array(["1"] * 10 + ["2"] * 10, dtype=object)
MPG = np.array([21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4,
                30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4])
WT = np.array([2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44, 3.44, 4.07, 3.73, 3.78, 5.25, 5.424, 5.345,
               2.2, 1.615, 1.835, 2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14, 1.513, 3.17, 2.77, 3.57, 2.78])


def vec(v):
    """coefficient container (dict name->value or array) -> float array"""
    return np.array(list(v.values()) if isinstance(v, dict) else v, dtype=float)


def test_every_wikipedia_test_is_registered():
    """List of statistical tests (Wikipedia, tables of tests) -> analysis ids."""
    need = {"t_one", "t_paired", "t_two", "t_welch", "f_test", "z_one", "z_two", "permutation", "kruskal", "mann_whitney",
            "wilcoxon", "sign_test", "friedman", "chi2_indep", "chi2_gof", "median_test", "multinomial", "mcnemar",
            "cochran_q", "binomial", "siegel_tukey", "chow", "fisher", "barnard", "boschloo", "shapiro_wilk", "ks_normal",
            "ks_two", "shapiro_francia", "lilliefors"}
    assert need <= set(REGISTRY), need - set(REGISTRY)


# ------------------------------------------------------------------ t / z / F  (R documentation: sleep data)
def test_welch_t_sleep_matches_R():
    r = run("t_welch", {"y": sleep_y, "group": sleep_g})
    v = r.values
    assert v["t"] == pytest.approx(-1.8608, abs=1e-4) and v["df"] == pytest.approx(17.776, abs=1e-3)
    assert v["p"] == pytest.approx(0.07939, abs=1e-5)
    assert (v["lo"], v["hi"]) == pytest.approx((-3.3654832, 0.2054832), abs=1e-6)


def test_paired_t_sleep_matches_R():
    r = run("t_paired", {"x": sleep1, "y": sleep2})
    assert abs(r.values["t"]) == pytest.approx(4.0621, abs=1e-4) and r.values["df"] == 9
    assert r.values["p"] == pytest.approx(0.002833, abs=1e-6)


def test_pooled_t_from_formula():
    r = run("t_two", {"y": sleep_y, "group": sleep_g})
    n1 = n2 = 10
    sp2 = (9 * sleep1.var(ddof=1) + 9 * sleep2.var(ddof=1)) / 18
    t = (sleep1.mean() - sleep2.mean()) / np.sqrt(sp2 * 0.2)
    assert r.values["t"] == pytest.approx(t, rel=1e-12) and r.values["df"] == 18
    assert r.values["p"] == pytest.approx(2 * st.t.sf(abs(t), 18), rel=1e-10)


def test_one_sample_t_formula_and_alternatives():
    x = sleep2
    t = (x.mean() - 1.0) / (x.std(ddof=1) / np.sqrt(10))
    for alt, p in (("two-sided", 2 * st.t.sf(abs(t), 9)), ("greater", st.t.sf(t, 9)), ("less", st.t.cdf(t, 9))):
        assert run("t_one", {"x": x}, mu0=1.0, alternative=alt).values["p"] == pytest.approx(p, rel=1e-10)
    assert run("t_one", {"x": x}, mu0=1.0).values["t"] == pytest.approx(t, rel=1e-12)


def test_z_tests_formula():
    x = np.array([5.1, 4.9, 5.6, 5.8, 6.0, 5.2, 4.7, 5.5])
    z = (x.mean() - 5.0) / (0.5 / np.sqrt(len(x)))
    r = run("z_one", {"x": x}, mu0=5.0, sigma=0.5)
    assert r.values["z"] == pytest.approx(z, rel=1e-12) and r.values["p"] == pytest.approx(2 * st.norm.sf(abs(z)), rel=1e-12)
    g = np.array(["a"] * 4 + ["b"] * 4, dtype=object)
    r = run("z_two", {"y": x, "group": g}, sigma1=0.4, sigma2=0.6)
    z = (x[:4].mean() - x[4:].mean()) / np.sqrt(0.16 / 4 + 0.36 / 4)
    assert r.values["z"] == pytest.approx(z, rel=1e-12)


def test_f_test_variances_formula():
    r = run("f_test", {"y": sleep_y, "group": sleep_g})
    F = sleep1.var(ddof=1) / sleep2.var(ddof=1)
    p = 2 * min(st.f.cdf(F, 9, 9), st.f.sf(F, 9, 9))
    assert r.values["F"] == pytest.approx(F, rel=1e-12) and r.values["p"] == pytest.approx(p, rel=1e-10)


# ------------------------------------------------------------------ nonparametric
def test_mann_whitney_sleep_matches_R():
    r = run("mann_whitney", {"y": sleep_y, "group": sleep_g})
    assert r.values["U1"] == pytest.approx(25.5) and r.values["p"] == pytest.approx(0.06933, abs=2e-5)


def test_mann_whitney_U_from_ranks():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 15), rng.normal(0.5, 1, 12)
    ranks = st.rankdata(np.r_[a, b])
    U1 = ranks[:15].sum() - 15 * 16 / 2
    r = run("mann_whitney", {"y": np.r_[a, b], "group": np.array(["a"] * 15 + ["b"] * 12, dtype=object)})
    assert r.values["U1"] == pytest.approx(U1) and r.values["U1"] + r.values["U2"] == pytest.approx(15 * 12)


def test_kruskal_from_formula():
    rng = np.random.default_rng(3)
    ys = [rng.normal(m, 1, n) for m, n in ((0, 8), (0.4, 9), (1.0, 7))]
    y = np.concatenate(ys)
    g = np.array(sum(([k] * len(a) for k, a in zip("abc", ys)), []), dtype=object)
    N = len(y)
    rk = st.rankdata(y)
    H = 12 / (N * (N + 1)) * sum(rk[g == k].sum() ** 2 / (g == k).sum() for k in "abc") - 3 * (N + 1)
    r = run("kruskal", {"y": y, "group": g})
    assert r.values["H"] == pytest.approx(H, rel=1e-10) and r.values["df"] == 2
    assert r.values["p"] == pytest.approx(st.chi2.sf(H, 2), rel=1e-10)


def test_sign_test_exact_binomial():
    x = np.array([1.2, 0.4, -0.3, 2.2, 1.1, 0.9, -0.5, 1.7, 3.0, 0.2])
    k, n = (x > 0).sum(), 10
    from math import comb
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)
    r = run("sign_test", {"x": x}, m0=0.0)
    assert r.values["pos"] == 8 and r.values["neg"] == 2 and r.values["p"] == pytest.approx(p, rel=1e-12)


def test_wilcoxon_signed_rank_statistic_by_hand():
    d = np.array([1.5, -0.5, 2.0, 3.5, -1.0, 4.0, 0.7, 2.2])
    ranks = st.rankdata(np.abs(d))
    r = run("wilcoxon", {"x": d}, m0=0.0)
    assert r.values["Wplus"] == pytest.approx(ranks[d > 0].sum()) and r.values["Wminus"] == pytest.approx(ranks[d < 0].sum())


def test_friedman_by_formula():
    rng = np.random.default_rng(4)
    M = rng.normal(size=(10, 3)) + np.array([0, 0.5, 1.0])
    rk = np.apply_along_axis(st.rankdata, 1, M)
    k, n = 3, 10
    chi = 12 / (n * k * (k + 1)) * (rk.sum(0) ** 2).sum() - 3 * n * (k + 1)
    r = run("friedman", {"xs": [M[:, 0], M[:, 1], M[:, 2]]})
    assert r.values["chi2"] == pytest.approx(chi, rel=1e-9) and r.values["p"] == pytest.approx(st.chi2.sf(chi, 2), rel=1e-9)


def test_permutation_test_is_reproducible_and_sane():
    kw = dict(n_resamples=5000, seed=7)
    a = run("permutation", {"y": sleep_y, "group": sleep_g}, **kw).values
    b = run("permutation", {"y": sleep_y, "group": sleep_g}, **kw).values
    assert a == b                                   # reproducible
    assert a["obs"] == pytest.approx(sleep1.mean() - sleep2.mean())
    assert abs(a["p"] - 0.0792) < 0.02               # close to the exact/Welch answer for these data


def test_siegel_tukey_and_median_test_run_and_agree_with_scipy():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 1, 14), rng.normal(0, 2.5, 13)
    g = np.array(["a"] * 14 + ["b"] * 13, dtype=object)
    r = run("median_test", {"y": np.r_[a, b], "group": g}, correction=False)
    chi, p, med, tab = st.median_test(a, b, correction=False)
    assert r.values["chi2"] == pytest.approx(chi, rel=1e-9) and r.values["p"] == pytest.approx(p, rel=1e-9)
    s = run("siegel_tukey", {"y": np.r_[a, b], "group": g})
    assert s.values["R1"] + s.values["R2"] == pytest.approx(27 * 28 / 2)   # Siegel-Tukey ranks sum to N(N+1)/2


# ------------------------------------------------------------------ categorical (R documentation examples)
def test_fisher_tea_tasting_matches_R():
    row = np.array(["milk"] * 4 + ["tea"] * 4, dtype=object)
    col = np.array(["milk", "milk", "milk", "tea", "milk", "tea", "tea", "tea"], dtype=object)
    r = run("fisher", {"row": row, "col": col})
    assert r.values["p"] == pytest.approx(0.4857, abs=1e-4)
    assert run("fisher", {"row": row, "col": col}, alternative="greater").values["p"] == pytest.approx(0.2429, abs=1e-4)


def _expand(table):
    rows, cols = [], []
    for i, rw in enumerate(table):
        for j, n in enumerate(rw):
            rows += [f"r{i}"] * n
            cols += [f"c{j}"] * n
    return np.array(rows, dtype=object), np.array(cols, dtype=object)


def test_chi_square_independence_matches_R_docs():
    row, col = _expand([[762, 327, 468], [484, 239, 477]])
    r = run("chi2_indep", {"row": row, "col": col})
    assert r.values["chi2"] == pytest.approx(30.07, abs=0.01) and r.values["df"] == 2
    assert r.values["p"] == pytest.approx(2.954e-07, rel=1e-3)


def test_chi_square_by_hand_with_expected_counts():
    T = np.array([[20, 15, 25], [30, 35, 15]])
    exp = np.outer(T.sum(1), T.sum(0)) / T.sum()
    chi = ((T - exp) ** 2 / exp).sum()
    r = run("chi2_indep", {"row": _expand(T)[0], "col": _expand(T)[1]})
    assert r.values["chi2"] == pytest.approx(chi, rel=1e-12) and r.values["p"] == pytest.approx(st.chi2.sf(chi, 2), rel=1e-10)
    assert np.allclose(np.asarray(r.values["expected"], float), exp)
    assert r.values["V"] == pytest.approx(np.sqrt(chi / (T.sum() * 1)), rel=1e-9)


def test_chi_square_gof_and_binomial_and_mcnemar():
    x = np.array(["a"] * 30 + ["b"] * 50 + ["c"] * 20, dtype=object)
    chi = ((np.array([30, 50, 20]) - 33.333333333) ** 2 / 33.333333333).sum()
    assert run("chi2_gof", {"x": x}).values["chi2"] == pytest.approx(chi, rel=1e-8)
    b = np.array([1] * 7 + [0] * 3, dtype=object)
    r = run("binomial", {"x": b}, success="1", p0=0.5)
    assert r.values["k"] == 7 and r.values["p"] == pytest.approx(0.34375, rel=1e-9)        # 2*P(X>=7 | 10,.5)
    assert (r.values["lo"], r.values["hi"]) == pytest.approx((0.3475, 0.9333), abs=1e-3)    # Clopper-Pearson
    a_ = np.array(["y"] * 59 + ["y"] * 6 + ["n"] * 16 + ["n"] * 80, dtype=object)
    b_ = np.array(["y"] * 59 + ["n"] * 6 + ["y"] * 16 + ["n"] * 80, dtype=object)
    r = run("mcnemar", {"a": a_, "b": b_}, correction=False)
    assert r.values["chi2"] == pytest.approx((6 - 16) ** 2 / 22, rel=1e-9)


def test_exact_2x2_tests_are_valid_p_values_and_ordered():
    row, col = _expand([[7, 2], [1, 9]])
    pf = run("fisher", {"row": row, "col": col}).values["p"]
    pb = run("barnard", {"row": row, "col": col}).values["p"]
    pbo = run("boschloo", {"row": row, "col": col}).values["p"]
    assert 0 < pbo <= pb + 1e-9 <= pf + 1e-9 + 1e-9 or (0 < pbo <= pf)   # Boschloo is uniformly more powerful than Fisher
    assert all(0 < p <= 1 for p in (pf, pb, pbo))


# ------------------------------------------------------------------ normality
def test_normality_tests_match_scipy_and_detect_non_normal():
    rng = np.random.default_rng(6)
    x = rng.normal(10, 2, 60)
    w, p = st.shapiro(x)
    r = run("shapiro_wilk", {"x": x})
    assert r.values["W"] == pytest.approx(w, rel=1e-10) and r.values["p"] == pytest.approx(p, rel=1e-8)
    jb = run("jarque_bera", {"x": x}).values
    n, s, k = len(x), st.skew(x), st.kurtosis(x)
    assert jb["JB"] == pytest.approx(n / 6 * (s ** 2 + k ** 2 / 4), rel=1e-9)
    ad = run("anderson_darling", {"x": x}).values
    ys = np.sort(x); z = st.norm.cdf((ys - x.mean()) / x.std(ddof=1)); i = np.arange(1, n + 1)
    A2 = -n - np.sum((2 * i - 1) / n * (np.log(z) + np.log(1 - z[::-1])))      # defining formula
    assert ad["A2"] == pytest.approx(A2, rel=1e-10)
    assert ad["A2star"] == pytest.approx(A2 * (1 + 0.75 / n + 2.25 / n ** 2), rel=1e-10)
    skewed = rng.exponential(1, 200)
    for tid in ("shapiro_wilk", "shapiro_francia", "lilliefors", "jarque_bera", "dagostino", "anderson_darling"):
        assert run(tid, {"x": skewed}).values["p"] < 0.01, tid
        assert run(tid, {"x": x}).values["p"] > 0.01, tid


def test_ks_two_sample_matches_ecdf_definition():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 40), rng.normal(0.8, 1, 35)
    allv = np.sort(np.r_[a, b])
    D = np.max(np.abs(np.searchsorted(np.sort(a), allv, side="right") / 40 - np.searchsorted(np.sort(b), allv, side="right") / 35))
    r = run("ks_two", {"y": np.r_[a, b], "group": np.array(["a"] * 40 + ["b"] * 35, dtype=object)})
    assert r.values["D"] == pytest.approx(D, rel=1e-12) and 0 < r.values["p"] < 0.05


def test_lilliefors_critical_behaviour():
    rng = np.random.default_rng(8)
    x = rng.normal(size=100)
    D = st.kstest((x - x.mean()) / x.std(ddof=1), "norm").statistic
    r = run("lilliefors", {"x": x})
    assert r.values["D"] == pytest.approx(D, rel=1e-10) and r.values["p"] > 0.05


# ------------------------------------------------------------------ ANOVA / variance
def test_one_way_anova_from_sums_of_squares():
    rng = np.random.default_rng(9)
    ys = [rng.normal(m, 1.5, 12) for m in (0, 0.7, 1.4)]
    y = np.concatenate(ys)
    g = np.array(sum(([k] * 12 for k in "abc"), []), dtype=object)
    ssb = sum(12 * (a.mean() - y.mean()) ** 2 for a in ys)
    ssw = sum(((a - a.mean()) ** 2).sum() for a in ys)
    F = (ssb / 2) / (ssw / 33)
    r = run("anova_one", {"y": y, "group": g})
    assert r.values["F"] == pytest.approx(F, rel=1e-10) and r.values["p"] == pytest.approx(st.f.sf(F, 2, 33), rel=1e-9)
    assert r.values["eta2"] == pytest.approx(ssb / (ssb + ssw), rel=1e-10)
    assert (r.values["dfb"], r.values["dfw"]) == (2, 33)


def test_levene_bartlett_welch_match_scipy():
    rng = np.random.default_rng(10)
    ys = [rng.normal(0, s, 15) for s in (1, 2, 3)]
    y = np.concatenate(ys)
    g = np.array(sum(([k] * 15 for k in "abc"), []), dtype=object)
    assert run("levene", {"y": y, "group": g}).values["p"] == pytest.approx(st.levene(*ys, center="median").pvalue, rel=1e-9)
    assert run("bartlett", {"y": y, "group": g}).values["p"] == pytest.approx(st.bartlett(*ys).pvalue, rel=1e-9)
    # Welch (1951) ANOVA from the textbook formula
    n = np.array([15.0, 15.0, 15.0]); m = np.array([a.mean() for a in ys]); v = np.array([a.var(ddof=1) for a in ys])
    w = n / v; mw = (w * m).sum() / w.sum(); k = 3
    tmp = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    F = ((w * (m - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
    df2 = (k ** 2 - 1) / (3 * tmp)
    rw = run("anova_welch", {"y": y, "group": g}).values
    assert rw["F"] == pytest.approx(F, rel=1e-9) and rw["df2"] == pytest.approx(df2, rel=1e-9)
    assert rw["p"] == pytest.approx(st.f.sf(F, k - 1, df2), rel=1e-9)


def test_tukey_hsd_matches_scipy():
    rng = np.random.default_rng(11)
    ys = [rng.normal(m, 1, 10) for m in (0, 1, 2)]
    y = np.concatenate(ys)
    g = np.array(sum(([k] * 10 for k in "abc"), []), dtype=object)
    r = run("tukey_hsd", {"y": y, "group": g})
    sp = st.tukey_hsd(*ys).pvalue
    P = np.asarray(r.values["p"], float).reshape(3, 3)
    assert P[np.triu_indices(3, 1)] == pytest.approx(sp[np.triu_indices(3, 1)], rel=1e-6)


# ------------------------------------------------------------------ correlation / descriptive
def test_pearson_spearman_kendall_and_fisher_ci():
    r_ = run("corr_test", {"x": WT, "y": MPG})
    assert r_.values["r"] == pytest.approx(-0.8676594, abs=1e-6)            # R: cor(mtcars$wt, mtcars$mpg)
    assert r_.values["p"] == pytest.approx(1.294e-10, rel=1e-3)             # R: cor.test(mtcars$wt, mtcars$mpg)
    z = np.arctanh(r_.values["r"]); se = 1 / np.sqrt(32 - 3)
    assert (r_.values["lo"], r_.values["hi"]) == pytest.approx((np.tanh(z - 1.959964 * se), np.tanh(z + 1.959964 * se)), abs=1e-6)
    assert run("corr_test", {"x": WT, "y": MPG}, method="spearman").values["r"] == pytest.approx(st.spearmanr(WT, MPG)[0], rel=1e-10)
    assert run("corr_test", {"x": WT, "y": MPG}, method="kendall").values["r"] == pytest.approx(st.kendalltau(WT, MPG)[0], rel=1e-10)


def test_summary_zscores_ci_known_values():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])          # classic: population sd = 2, sample sd = 2.138
    r = run("ci_mean", {"x": x})
    se = x.std(ddof=1) / np.sqrt(8)
    q = st.t.ppf(0.975, 7)
    assert (r.values["lower"], r.values["upper"]) == pytest.approx((5 - q * se, 5 + q * se), rel=1e-12)
    z = run("zscores", {"x": x}, ddof="sample (n-1)")
    text = z.to_text()
    assert "2.1381" in run("summary", {"xs": [x]}).to_text()          # sample sd with n-1 shown in the table
    zs = (x - 5) / x.std(ddof=1)
    assert f"{zs[-1]:.4f}" in text


def test_variance_ci_matches_chi_square_formula():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]); n = 8; s2 = x.var(ddof=1)
    r = run("ci_var", {"x": x})
    assert (r.values["var_lo"], r.values["var_hi"]) == pytest.approx(
        ((n - 1) * s2 / st.chi2.ppf(0.975, n - 1), (n - 1) * s2 / st.chi2.ppf(0.025, n - 1)), rel=1e-10)


def test_odds_ratio_relative_risk():
    row, col = _expand([[10, 20], [5, 40]])
    r = run("odds_ratio", {"row": row, "col": col})
    assert r.values["OR"] == pytest.approx((10 * 40) / (20 * 5)) or r.values["OR"] == pytest.approx(1 / ((10 * 40) / (20 * 5)))


# ------------------------------------------------------------------ regression (R: lm(mpg ~ wt, mtcars))
def test_simple_ols_matches_R_mtcars():
    r = run("ols", {"y": MPG, "xs": [WT]}, names={"y": "mpg", "xs": ["wt"]})
    p = vec(r.values["params"])
    assert p == pytest.approx([37.2851, -5.3445], abs=1e-4)
    assert r.values["r2"] == pytest.approx(0.7528, abs=1e-4) and r.values["F"] == pytest.approx(91.375, abs=1e-2)
    assert r.values["f_p"] == pytest.approx(1.294e-10, rel=1e-3)
    assert vec(r.values["bse"]) == pytest.approx([1.8776, 0.5591], abs=1e-4)
    assert r.values["rmse"] == pytest.approx(3.046, abs=1e-3)


def test_multiple_ols_matches_normal_equations():
    rng = np.random.default_rng(12)
    n = 200
    X = rng.normal(size=(n, 3)); beta = np.array([1.0, 2.0, -1.0, 0.5])
    y = beta[0] + X @ beta[1:] + rng.normal(size=n)
    A = np.column_stack([np.ones(n), X])
    b = np.linalg.solve(A.T @ A, A.T @ y)
    res = y - A @ b
    s2 = res @ res / (n - 4)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    r = run("ols", {"y": y, "xs": [X[:, 0], X[:, 1], X[:, 2]]})
    assert vec(r.values["params"]) == pytest.approx(b, rel=1e-10)
    assert vec(r.values["bse"]) == pytest.approx(se, rel=1e-8)
    assert r.values["r2"] == pytest.approx(1 - res @ res / ((y - y.mean()) ** 2).sum(), rel=1e-10)
    t = b / se
    assert vec(r.values["pvalues"]) == pytest.approx(2 * st.t.sf(abs(t), n - 4), rel=1e-6, abs=1e-14)


def test_ols_with_categorical_predictor_dummy_codes():
    rng = np.random.default_rng(13)
    n = 90
    g = np.array(["a", "b", "c"] * 30, dtype=object)
    x = rng.normal(size=n)
    y = 1 + 2 * x + (g == "b") * 1.5 + (g == "c") * -1 + rng.normal(0, .3, n)
    r = run("ols", {"y": y, "xs": [x], "factors": [g]})
    p = dict(zip(r.values["names"], vec(r.values["params"])))
    assert len(p) == 4                                         # intercept, x, two dummies (reference level = first level "a")
    num = [v for k, v in p.items() if k.lower().startswith("x")][0]
    dum = sorted(v for k, v in p.items() if k != "Intercept" and not k.lower().startswith("x"))
    assert p["Intercept"] == pytest.approx(1, abs=0.3) and num == pytest.approx(2, abs=0.2)
    assert dum == pytest.approx([-1.0, 1.5], abs=0.3)


def test_logistic_regression_matches_newton_raphson():
    rng = np.random.default_rng(14)
    n = 400
    x = rng.normal(size=n)
    pr = 1 / (1 + np.exp(-(0.5 + 1.2 * x)))
    y = (rng.uniform(size=n) < pr).astype(int)
    A = np.column_stack([np.ones(n), x]); b = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-A @ b)); W = mu * (1 - mu)
        b = b + np.linalg.solve(A.T @ (A * W[:, None]), A.T @ (y - mu))
    r = run("logit", {"y": y.astype(object), "xs": [x]}, success="1")
    assert vec(r.values["params"]) == pytest.approx(b, rel=1e-6)


def test_poisson_regression_matches_irls_and_chow_test_formula():
    rng = np.random.default_rng(15)
    n = 300
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.5 * x)).astype(float)
    A = np.column_stack([np.ones(n), x]); b = np.zeros(2)
    for _ in range(100):
        mu = np.exp(A @ b)
        b = b + np.linalg.solve(A.T @ (A * mu[:, None]), A.T @ (y - mu))
    assert vec(run("poisson", {"y": y, "xs": [x]}).values["params"]) == pytest.approx(b, rel=1e-6)
    # Chow: stable relationship split in halves -> compare with RSS formula
    t = np.arange(100.0); yy = 1 + 0.2 * t + rng.normal(size=100); yy[50:] += 0.3 * (t[50:] - 50)
    def rss(a, c):
        A_ = np.column_stack([np.ones_like(a), a]); r_ = c - A_ @ np.linalg.lstsq(A_, c, rcond=None)[0]; return r_ @ r_
    S, S1, S2, k = rss(t, yy), rss(t[:50], yy[:50]), rss(t[50:], yy[50:]), 2
    F = ((S - S1 - S2) / k) / ((S1 + S2) / (100 - 2 * k))
    r = run("chow", {"y": yy, "xs": [t]}, break_at=51)   # 1-based first row of regime 2
    assert r.values["F"] == pytest.approx(F, rel=1e-8) and r.values["p"] == pytest.approx(st.f.sf(F, k, 96), rel=1e-8)


def test_every_registered_analysis_declares_stata_r_python_code():
    missing = [i for i, s in REGISTRY.items() if not all(k in s.code for k in ("stata", "r", "python"))]
    assert not missing, missing
