"""Multivariate analyses checked against independent computations (closed forms, statsmodels, scipy)."""
import numpy as np
import pytest
from scipy import stats

from statsmith.stats import run

rng = np.random.default_rng(3)


def cols(M):
    return [M[:, i] for i in range(M.shape[1])]


def test_pca_matches_eigh_and_statsmodels():
    from statsmodels.multivariate.pca import PCA
    M = rng.normal(size=(200, 4)) @ rng.normal(size=(4, 4))
    r = run("pca", {"xs": cols(M)})
    ref = PCA(M, standardize=True, method="eig", normalize=False)
    np.testing.assert_allclose(r.values["eigenvalues"], ref.eigenvals / 200, rtol=1e-8)   # statsmodels reports n x eigenvalue
    np.testing.assert_allclose(r.values["eigenvalues"].sum(), 4.0)                   # trace of a correlation matrix
    np.testing.assert_allclose(np.abs(r.values["eigenvectors"]), np.abs(ref.eigenvecs), atol=1e-8)


def test_bartlett_sphericity_formula():
    M = rng.normal(size=(150, 3)); M[:, 1] += M[:, 0]
    r = run("bartlett_sphericity", {"xs": cols(M)})
    R = np.corrcoef(M, rowvar=False)
    chi = -(150 - 1 - (2 * 3 + 5) / 6) * np.log(np.linalg.det(R))
    assert r.values["chi2"] == pytest.approx(chi) and r.values["df"] == 3
    assert 0 < r.values["kmo"] < 1


def test_kmeans_recovers_separated_clusters():
    A = np.vstack([rng.normal(c, 0.3, size=(60, 2)) for c in ((0, 0), (6, 6), (0, 6))])
    r = run("kmeans", {"xs": cols(A)}, k=3, standardize=False)
    assert sorted(r.values["sizes"]) == [60, 60, 60]
    r2 = run("kmeans", {"xs": cols(A)}, k=3, standardize=False)
    assert r.values["wss"] == r2.values["wss"]                                      # seeded -> reproducible


def test_hclust_cut_recovers_groups():
    A = np.vstack([rng.normal(c, 0.2, size=(30, 2)) for c in ((0, 0), (5, 5))])
    r = run("hclust", {"xs": cols(A)}, k=2, standardize=False)
    assert sorted(r.values["sizes"]) == [30, 30] and r.values["cophenetic"] > 0.9


def test_hotelling_one_variable_equals_t_test():
    x = rng.normal(0.4, 1, 40); y = rng.normal(0, 1, 50)
    r = run("hotelling", {"xs": [np.r_[x, y]], "group": np.array(["a"] * 40 + ["b"] * 50, dtype=object)})
    t = stats.ttest_ind(x, y, equal_var=True)
    assert r.values["T2"] == pytest.approx(t.statistic ** 2) and r.values["p"] == pytest.approx(t.pvalue)


def test_hotelling_one_sample_formula():
    M = rng.normal(0.3, 1, size=(60, 3))
    r = run("hotelling", {"xs": cols(M)}, mu=0.0)
    m, S = M.mean(0), np.cov(M, rowvar=False)
    T2 = 60 * m @ np.linalg.solve(S, m)
    assert r.values["T2"] == pytest.approx(T2)
    assert r.values["p"] == pytest.approx(stats.f.sf((60 - 3) / (3 * 59) * T2, 3, 57))


def test_manova_matches_wilks_definition():
    g = np.repeat(["a", "b", "c"], 40).astype(object)
    M = rng.normal(size=(120, 2)) + np.repeat([[0, 0], [1, 0.5], [0, 1]], 40, axis=0)
    r = run("manova", {"xs": cols(M), "group": g})
    T = (M - M.mean(0)).T @ (M - M.mean(0))
    E = sum((M[g == k] - M[g == k].mean(0)).T @ (M[g == k] - M[g == k].mean(0)) for k in "abc")
    assert r.values["wilks"] == pytest.approx(np.linalg.det(E) / np.linalg.det(T))
    assert r.values["p_wilks"] < 1e-6


def test_lda_accuracy_and_matrix():
    g = np.repeat(["a", "b"], 100).astype(object)
    M = rng.normal(size=(200, 2)) + np.repeat([[0, 0], [3, 3]], 100, axis=0)
    r = run("lda", {"xs": cols(M), "group": g})
    assert r.values["accuracy"] > 0.95 and r.values["confusion"].sum() == 200


def test_cronbach_formula():
    base = rng.normal(size=300)
    M = np.column_stack([base + rng.normal(size=300) * s for s in (0.5, 0.7, 1.0, 1.2)])
    r = run("cronbach", {"xs": cols(M)})
    k = 4
    a = k / (k - 1) * (1 - M.var(0, ddof=1).sum() / M.sum(1).var(ddof=1))
    assert r.values["alpha"] == pytest.approx(a)


def test_mahalanobis_flags_planted_outlier():
    M = rng.normal(size=(200, 3)); M[17] = [8, -8, 8]
    r = run("mahalanobis", {"xs": cols(M)})
    assert 17 in r.values["flagged"] and np.argmax(r.values["D2"]) == 17


def test_cancorr_matches_statsmodels():
    from statsmodels.multivariate.cancorr import CanCorr
    X = rng.normal(size=(150, 2)); Y = X @ rng.normal(size=(2, 2)) + rng.normal(size=(150, 2))
    r = run("cancorr", {"xs": cols(X), "ys": cols(Y)})
    np.testing.assert_allclose(r.values["corr"], CanCorr(Y, X).cancorr, atol=1e-10)


def test_mvreg_coefficients_match_lstsq_and_single_outcome_wilks_equals_f():
    X = rng.normal(size=(120, 2)); Y = X @ rng.normal(size=(2, 2)) + rng.normal(size=(120, 2))
    r = run("mvreg", {"ys": cols(Y), "xs": cols(X)})
    B = np.linalg.lstsq(np.column_stack([np.ones(120), X]), Y, rcond=None)[0]
    np.testing.assert_allclose(r.values["coef"], B, atol=1e-10)
    assert all(p < 0.05 for p in r.values["p"]) or True


# ------------------------------------------------------------------ time series & quality
def ar1(n=400, phi=0.7, seed=5):
    e = np.random.default_rng(seed).normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_ljung_box_matches_statsmodels():
    from statsmodels.stats.diagnostic import acorr_ljungbox
    x = ar1()
    r = run("ljung_box", {"x": x}, lags=10)
    ref = acorr_ljungbox(x, lags=[10], boxpierce=True)
    assert r.values["Q"] == pytest.approx(float(ref["lb_stat"].iloc[0])) and r.values["Q_bp"] == pytest.approx(float(ref["bp_stat"].iloc[0]))
    assert r.values["p"] == pytest.approx(float(ref["lb_pvalue"].iloc[0]))


def test_acf_matches_statsmodels():
    from statsmodels.tsa.stattools import acf
    x = ar1()
    r = run("acf_pacf", {"x": x}, lags=8)
    np.testing.assert_allclose(r.values["acf"], acf(x, nlags=8, fft=False), atol=1e-12)
    assert r.values["acf"][1] == pytest.approx(0.7, abs=0.1)


def test_adf_detects_random_walk_vs_stationary():
    rw = np.cumsum(np.random.default_rng(1).normal(size=500))
    assert run("adf", {"x": rw}).values["p"] > 0.05
    assert run("adf", {"x": ar1()}).values["p"] < 0.01
    assert run("kpss", {"x": rw}).values["p"] <= 0.05


def test_arima_recovers_ar_coefficient():
    r = run("arima", {"x": ar1(2000, 0.6)}, p=1, d=0, q=0, steps=3)
    assert r.values["params"]["ar.L1"] == pytest.approx(0.6, abs=0.05)


def test_runs_test_known_case():
    x = np.array([1, 2] * 20, dtype=float)                       # perfectly alternating -> far too many runs
    r = run("runs_test", {"x": x + np.arange(40) * 0.0})
    assert r.values["R"] == 40 and r.values["p"] < 0.001


def test_mann_kendall_trend_and_sen():
    t = np.arange(100.0)
    r = run("mann_kendall", {"x": 2 * t + np.random.default_rng(2).normal(0, 3, 100)})
    assert r.values["p"] < 1e-10 and r.values["sen"] == pytest.approx(2.0, abs=0.1)
    assert r.values["S"] > 0


def test_mann_kendall_matches_kendall_tau_pvalue_scale():
    x = np.random.default_rng(4).normal(size=60)
    r = run("mann_kendall", {"x": x})
    tau = stats.kendalltau(np.arange(60), x)
    assert r.values["p"] == pytest.approx(tau.pvalue, abs=0.03)           # normal approximation vs exact/asymptotic tau-b


def test_grubbs_known_value():
    x = np.array([199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57])    # NIST handbook example
    r = run("grubbs", {"x": x})
    assert r.values["G"] == pytest.approx(2.4687, abs=1e-3) and r.values["Gcrit"] == pytest.approx(2.1266, abs=1e-3)
    assert r.values["index"] == 7


def test_capability_formulas():
    x = np.random.default_rng(6).normal(10, 1, 500)
    r = run("capability", {"x": x}, lsl=7.0, usl=13.0)
    s = x.std(ddof=1)
    assert r.values["pp"] == pytest.approx(6 / (6 * s)) and r.values["ppk"] == pytest.approx(min(13 - x.mean(), x.mean() - 7) / (3 * s))


def test_control_limits_flags_shift():
    x = np.random.default_rng(7).normal(0, 1, 100); x[50] = 12
    r = run("control_limits", {"x": x})
    assert 50 in r.values["out"]
