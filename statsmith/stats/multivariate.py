"""Multivariate methods: PCA, sphericity/KMO, k-means, hierarchical clustering, MANOVA, Hotelling T², LDA,
Cronbach's alpha, Mahalanobis outliers, canonical correlation, multivariate multiple regression."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..charts.base import Param, Role
from .base import Result, analysis, clean, fmt_p
from .common import ALPHA, CONF, XS
from .common import need

CAT = "Multivariate"
_GRP = Role("group", "categorical", "Group")


def complete_matrix(cols, extra=None):
    """Listwise-complete numeric matrix (n × p) (+ aligned ``extra`` array)."""
    X = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    ok = np.isfinite(X).all(axis=1)
    if extra is not None:
        e = np.asarray(extra, dtype=object)
        ok &= np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in e.tolist()])
        return X[ok], e[ok]
    return X[ok]


def _names(d, role="xs"):
    return d.label_list(role)


# =========================================================================== PCA
@analysis("pca", "Principal component analysis (PCA)", CAT, [XS],
          [Param("basis", "Analyse", "choice", "correlation", choices=("correlation", "covariance")),
           Param("n_show", "Components in loadings table", "int", 5, min=1, max=50)],
          code={"stata": "pca {xs}", "r": "fit <- prcomp(df[, c({xs_q})], scale. = TRUE); summary(fit); fit$rotation",
                "python": "from sklearn.decomposition import PCA; from sklearn.preprocessing import StandardScaler; "
                          "Z = StandardScaler().fit_transform(df[[{xs_q}]].dropna()); p = PCA().fit(Z); print(p.explained_variance_, p.components_)"},
          reference="Eigen-decomposition of the correlation (default) or covariance matrix (ddof = 1). Loadings = eigenvectors scaled by √eigenvalue "
                    "for the correlation basis (as Stata `pca` reports eigenvectors; sign of each component is arbitrary).")
def pca(d, basis="correlation", n_show=5):
    X = complete_matrix(d["xs"])
    names = _names(d)
    need(X.shape[1] >= 2 and X.shape[0] > X.shape[1], "need at least 2 variables and more rows than variables")
    n, p = X.shape
    M = np.corrcoef(X, rowvar=False) if basis == "correlation" else np.cov(X, rowvar=False)
    w, V = np.linalg.eigh(M)
    o = np.argsort(w)[::-1]
    w, V = np.clip(w[o], 0, None), V[:, o]
    V = V * np.where(V[np.abs(V).argmax(axis=0), np.arange(p)] < 0, -1, 1)     # deterministic sign: largest |loading| positive
    prop = w / w.sum()
    r = Result(f"Principal component analysis ({basis} matrix)")
    r.kv("", [("observations (complete cases)", n), ("variables", p)])
    r.table("Eigenvalues", ["Component", "Eigenvalue", "Proportion", "Cumulative"],
            [[f"PC{i + 1}", w[i], prop[i], prop[: i + 1].sum()] for i in range(p)])
    k = min(n_show, p)
    load = V[:, :k] * (np.sqrt(w[:k]) if basis == "correlation" else 1)
    r.table("Eigenvectors" if basis == "covariance" else "Loadings (correlation of variable with component)",
            ["Variable"] + [f"PC{i + 1}" for i in range(k)], [[names[j]] + list(load[j]) for j in range(p)])
    kaiser = int((w > 1).sum()) if basis == "correlation" else None
    if kaiser is not None:
        r.note(f"Kaiser criterion (eigenvalue > 1) suggests {kaiser} component(s).")
    r.values.update(dict(eigenvalues=w, proportion=prop, eigenvectors=V, n=n))
    return r


# =========================================================================== sphericity + KMO
@analysis("bartlett_sphericity", "Bartlett's test of sphericity & KMO", CAT, [XS],
          code={"stata": "factor {xs}\nestat kmo", "r": "psych::cortest.bartlett(cor(df[, c({xs_q})], use = 'complete.obs'), n = nrow(na.omit(df[, c({xs_q})]))); psych::KMO(df[, c({xs_q})])",
                "python": "from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo; calculate_bartlett_sphericity(df[[{xs_q}]].dropna()); calculate_kmo(df[[{xs_q}]].dropna())"},
          reference="χ² = -(n - 1 - (2p+5)/6)·ln|R|, df = p(p-1)/2 (Bartlett 1951). KMO = Σr² / (Σr² + Σa²), a = partial correlations (from R⁻¹).")
def bartlett_sphericity(d):
    X = complete_matrix(d["xs"])
    names = _names(d)
    n, p = X.shape
    need(p >= 2 and n > p, "need at least 2 variables and more rows than variables")
    R = np.corrcoef(X, rowvar=False)
    chi = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    df = p * (p - 1) / 2
    pv = float(stats.chi2.sf(chi, df))
    Ri = np.linalg.pinv(R)
    A = -Ri / np.sqrt(np.outer(np.diag(Ri), np.diag(Ri)))
    off = ~np.eye(p, dtype=bool)
    r2, a2 = (R[off] ** 2), (A[off] ** 2)
    kmo = float(r2.sum() / (r2.sum() + a2.sum()))
    msa = [float((R[j, off[j]] ** 2).sum() / ((R[j, off[j]] ** 2).sum() + (A[j, off[j]] ** 2).sum())) for j in range(p)]
    r = Result("Bartlett's test of sphericity and Kaiser–Meyer–Olkin measure")
    r.kv("Bartlett", [("n", n), ("chi-square", float(chi)), ("df", df), ("p-value", pv)])
    r.kv("Kaiser–Meyer–Olkin", [("overall KMO", kmo)] + list(zip(names, msa)))
    r.note("H0: the correlation matrix is an identity matrix (variables uncorrelated). KMO > 0.6 is usually considered adequate for factor/PCA work.")
    r.values.update(dict(chi2=float(chi), df=df, p=pv, kmo=kmo))
    return r


# =========================================================================== k-means
@analysis("kmeans", "k-means clustering", CAT, [XS],
          [Param("k", "Number of clusters (k)", "int", 3, min=2, max=50), Param("standardize", "Standardise variables (z-scores)", "bool", True),
           Param("n_init", "Random restarts", "int", 10, min=1, max=200), Param("seed", "Random seed", "int", 12345, min=0, max=2**31 - 1)],
          code={"stata": "cluster kmeans {xs}, k({k}) start(krandom(12345))", "r": "set.seed(12345); km <- kmeans(scale(df[, c({xs_q})]), centers = {k}, nstart = {n_init}); km$centers; table(km$cluster)",
                "python": "from sklearn.cluster import KMeans; KMeans(n_clusters={k}, n_init={n_init}, random_state=12345).fit(df[[{xs_q}]].dropna())"},
          reference="Lloyd's algorithm with k-means++ seeding (scipy.cluster.vq.kmeans2), best of n restarts by within-cluster sum of squares. "
                    "Cluster labels are arbitrary and results depend on the seed; R/Stata/sklearn can find a different local optimum.")
def kmeans(d, k=3, standardize=True, n_init=10, seed=12345):
    from scipy.cluster.vq import kmeans2
    X = complete_matrix(d["xs"])
    names = _names(d)
    n, p = X.shape
    need(n >= k * 2, f"need at least {2 * k} complete rows")
    Z = (X - X.mean(0)) / np.where(X.std(0, ddof=1) > 0, X.std(0, ddof=1), 1) if standardize else X
    best = None
    for i in range(n_init):
        cen, lab = kmeans2(Z, k, minit="++", seed=np.random.default_rng(seed + i), missing="warn")
        wss = float(((Z - cen[lab]) ** 2).sum())
        if best is None or wss < best[0]:
            best = (wss, cen, lab)
    wss, cen, lab = best
    tss = float(((Z - Z.mean(0)) ** 2).sum())
    r = Result(f"k-means clustering (k = {k})")
    r.kv("", [("observations", n), ("within-cluster SS", wss), ("total SS", tss), ("between / total SS", 1 - wss / tss)])
    sizes = np.bincount(lab, minlength=k)
    means = np.array([X[lab == j].mean(0) if sizes[j] else np.full(p, np.nan) for j in range(k)])
    r.table("Cluster sizes and means (original units)", ["Cluster", "n"] + names, [[j + 1, int(sizes[j])] + list(means[j]) for j in range(k)])
    r.values.update(dict(wss=wss, tss=tss, sizes=sizes, labels=lab, centers=means))
    return r


# =========================================================================== hierarchical clustering
@analysis("hclust", "Hierarchical clustering (agglomerative)", CAT, [XS],
          [Param("method", "Linkage", "choice", "ward", choices=("ward", "complete", "average", "single")),
           Param("k", "Cut tree into k clusters", "int", 3, min=2, max=50), Param("standardize", "Standardise variables", "bool", True)],
          code={"stata": "cluster {method}linkage {xs}\ncluster generate grp = groups({k})", "r": "hc <- hclust(dist(scale(df[, c({xs_q})])), method = '{method}.D2'); cutree(hc, k = {k})",
                "python": "from scipy.cluster.hierarchy import linkage, fcluster; fcluster(linkage(df[[{xs_q}]].dropna(), '{method}'), {k}, 'maxclust')"},
          reference="scipy.cluster.hierarchy.linkage on Euclidean distances (Ward minimum-variance by default); tree cut with 'maxclust'. "
                    "Limited to 5,000 rows (O(n²) memory) - use k-means for more.")
def hclust(d, method="ward", k=3, standardize=True):
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist
    X = complete_matrix(d["xs"])
    names = _names(d)
    need(5 <= X.shape[0] <= 5000, "hierarchical clustering needs 5 to 5,000 complete rows (use k-means for larger data)")
    Z = (X - X.mean(0)) / np.where(X.std(0, ddof=1) > 0, X.std(0, ddof=1), 1) if standardize else X
    L = linkage(Z, method)
    lab = fcluster(L, k, "maxclust")
    coph = float(cophenet(L, pdist(Z))[0])
    r = Result(f"Hierarchical clustering ({method} linkage)")
    r.kv("", [("observations", X.shape[0]), ("clusters", k), ("cophenetic correlation", coph)])
    sizes = np.bincount(lab)[1:]
    means = np.array([X[lab == j].mean(0) for j in range(1, len(sizes) + 1)])
    r.table("Clusters", ["Cluster", "n"] + names, [[j + 1, int(sizes[j])] + list(means[j]) for j in range(len(sizes))])
    r.table("Last merges (height = distance at which clusters join)", ["Step", "Height"],
            [[int(i + 1), float(L[i, 2])] for i in range(max(len(L) - 8, 0), len(L))])
    r.values.update(dict(labels=lab, sizes=sizes, cophenetic=coph, linkage=L))
    return r


# =========================================================================== MANOVA
@analysis("manova", "MANOVA (one-way)", CAT, [XS, _GRP],
          code={"stata": "manova {xs} = {group}", "r": "summary(manova(cbind({xs}) ~ factor({group_r}), data = df), test = 'Wilks')",
                "python": "from statsmodels.multivariate.manova import MANOVA; MANOVA.from_formula('{xs_plus} ~ C({group})', df).mv_test()"},
          reference="Wilks' Λ = |E|/|E+H|, Pillai's trace, Hotelling–Lawley trace and Roy's largest root with the usual F approximations "
                    "(statsmodels.multivariate.manova). E, H = within / between SSCP matrices.")
def manova(d):
    import pandas as pd
    from statsmodels.multivariate.manova import MANOVA
    X, g = complete_matrix(d["xs"], d["group"])
    names = _names(d)
    levels = list(dict.fromkeys(g.tolist()))
    need(len(levels) >= 2 and X.shape[0] > X.shape[1] + len(levels), "need ≥ 2 groups and enough rows")
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(X.shape[1])])
    df["g"] = [str(v) for v in g]
    res = MANOVA.from_formula(" + ".join(df.columns[:-1]) + " ~ C(g)", df).mv_test()
    tab = res.results["C(g)"]["stat"]
    r = Result(f"One-way MANOVA: {', '.join(names)} by {d.name('group')}")
    rows = [[idx, float(row["Value"]), float(row["F Value"]), float(row["Num DF"]), float(row["Den DF"]), float(row["Pr > F"])] for idx, row in tab.iterrows()]
    r.table("Multivariate tests", ["Statistic", "Value", "F", "Num DF", "Den DF", "p-value"], rows)
    gm = [(str(l), int((df["g"] == str(l)).sum())) for l in levels]
    r.kv("Group sizes", gm)
    r.values.update(dict(wilks=float(tab.loc["Wilks' lambda", "Value"]), p_wilks=float(tab.loc["Wilks' lambda", "Pr > F"]),
                         pillai=float(tab.loc["Pillai's trace", "Value"]), p_pillai=float(tab.loc["Pillai's trace", "Pr > F"])))
    return r


# =========================================================================== Hotelling T²
@analysis("hotelling", "Hotelling's T² (one- or two-sample)", CAT, [XS, Role("group", "categorical", "Group (blank = one-sample vs 0)", required=False)],
          [Param("mu", "Hypothesised mean for every variable (one-sample)", "float", 0.0)],
          code={"stata": "* Hotelling T2: mvtest means {xs}, by({group})", "r": "# install.packages('ICSNP'); ICSNP::HotellingsT2(as.matrix(df[, c({xs_q})]), mu = rep({mu}, length(c({xs_q}))))",
                "python": "# scipy has no T2: F = (n-p)/(p(n-1))·T², T² = n(x̄-μ)'S⁻¹(x̄-μ)   (see documentation)"},
          reference="One-sample T² = n(x̄-μ)'S⁻¹(x̄-μ), F = (n-p)/(p(n-1))·T² ~ F(p, n-p). Two-sample uses the pooled covariance: "
                    "T² = n1n2/(n1+n2)·(x̄1-x̄2)'S_p⁻¹(x̄1-x̄2), F = (n1+n2-p-1)/(p(n1+n2-2))·T² ~ F(p, n1+n2-p-1) (Hotelling 1931).")
def hotelling(d, mu=0.0):
    names = _names(d)
    if d.has("group"):
        X, g = complete_matrix(d["xs"], d["group"])
        lv = list(dict.fromkeys(g.tolist()))
        need(len(lv) == 2, f"two-sample T² needs exactly 2 groups (found {len(lv)}); use MANOVA for more")
        A, B = X[g == lv[0]], X[g == lv[1]]
        n1, n2, p = len(A), len(B), X.shape[1]
        need(n1 + n2 - p - 1 > 0, "too few rows for this many variables")
        Sp = ((n1 - 1) * np.cov(A, rowvar=False).reshape(p, p) + (n2 - 1) * np.cov(B, rowvar=False).reshape(p, p)) / (n1 + n2 - 2)
        dm = A.mean(0) - B.mean(0)
        T2 = float(n1 * n2 / (n1 + n2) * dm @ np.linalg.solve(Sp, dm))
        F = (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * T2
        df1, df2 = p, n1 + n2 - p - 1
        title = f"Two-sample Hotelling T²: {lv[0]} vs {lv[1]}"
        extra = [("n1", n1), ("n2", n2)]
    else:
        X = complete_matrix(d["xs"])
        n, p = X.shape
        need(n > p, "need more rows than variables")
        dm = X.mean(0) - mu
        T2 = float(n * dm @ np.linalg.solve(np.cov(X, rowvar=False).reshape(p, p), dm))
        F = (n - p) / (p * (n - 1)) * T2
        df1, df2 = p, n - p
        title = f"One-sample Hotelling T² (H0: all means = {mu:g})"
        extra = [("n", n)]
    pv = float(stats.f.sf(F, df1, df2))
    r = Result(title)
    r.kv("", extra + [("variables", p), ("T²", T2), ("F", float(F)), ("df1", df1), ("df2", df2), ("p-value", pv)])
    r.values.update(dict(T2=T2, F=float(F), df1=df1, df2=df2, p=pv))
    return r


# =========================================================================== LDA
@analysis("lda", "Linear discriminant analysis", CAT, [XS, _GRP],
          code={"stata": "discrim lda {xs}, group({group})", "r": "MASS::lda(factor({group_r}) ~ {xs_plus}, data = df)",
                "python": "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA; LDA().fit(df[[{xs_q}]].dropna(), df['{group}'])"},
          reference="Fisher/Gaussian LDA with pooled within-group covariance and empirical priors. Discriminant functions come from the eigenvectors of "
                    "S_w⁻¹S_b; classification uses the linear score δ_k(x) = x'S⁻¹μ_k − ½μ_k'S⁻¹μ_k + ln π_k. Accuracy is resubstitution (optimistic).")
def lda(d):
    X, g = complete_matrix(d["xs"], d["group"])
    names = _names(d)
    levels = list(dict.fromkeys(g.tolist()))
    n, p = X.shape
    K = len(levels)
    need(K >= 2 and n > p + K, "need ≥ 2 groups and more rows than variables + groups")
    gi = np.array([levels.index(v) for v in g.tolist()])
    mus = np.array([X[gi == k].mean(0) for k in range(K)])
    nk = np.bincount(gi, minlength=K)
    Sw = sum(((X[gi == k] - mus[k]).T @ (X[gi == k] - mus[k])) for k in range(K)) / (n - K)
    Sw = Sw.reshape(p, p)
    pri = nk / n
    Si = np.linalg.pinv(Sw)
    scores = X @ Si @ mus.T - 0.5 * np.einsum("kp,pq,kq->k", mus, Si, mus) + np.log(pri)
    pred = scores.argmax(1)
    acc = float((pred == gi).mean())
    conf = np.zeros((K, K), dtype=int)
    for a, b in zip(gi, pred):
        conf[a, b] += 1
    grand = X.mean(0)
    Sb = sum(nk[k] * np.outer(mus[k] - grand, mus[k] - grand) for k in range(K)) / (K - 1)
    w, V = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    o = np.argsort(-w.real)[: min(K - 1, p)]
    w = w.real[o]
    r = Result(f"Linear discriminant analysis: {d.name('group')}")
    r.kv("", [("observations", n), ("groups", K), ("resubstitution accuracy", acc)])
    r.table("Group sizes, priors", ["Group", "n", "Prior"], [[str(levels[k]), int(nk[k]), float(pri[k])] for k in range(K)])
    r.table("Classification (rows = actual, columns = predicted)", ["Actual \\ Pred"] + [str(l) for l in levels],
            [[str(levels[a])] + [int(v) for v in conf[a]] for a in range(K)])
    r.table("Discriminant functions", ["Function", "Eigenvalue", "% of between-group variance"],
            [[f"LD{i + 1}", float(w[i]), float(w[i] / w.sum() * 100)] for i in range(len(w))])
    r.values.update(dict(accuracy=acc, confusion=conf, eigenvalues=w))
    return r


# =========================================================================== Cronbach alpha
@analysis("cronbach", "Cronbach's alpha (scale reliability)", CAT, [XS],
          code={"stata": "alpha {xs}, item", "r": "psych::alpha(df[, c({xs_q})])", "python": "X = df[[{xs_q}]].dropna(); k = X.shape[1]; k/(k-1)*(1 - X.var(ddof=1).sum()/X.sum(axis=1).var(ddof=1))"},
          reference="α = k/(k-1)·(1 − Σσ²_i / σ²_total) over listwise-complete rows; 'alpha if item deleted' recomputes with each item removed.")
def cronbach(d):
    X = complete_matrix(d["xs"])
    names = _names(d)
    n, k = X.shape
    need(k >= 2 and n >= 3, "need at least 2 items and 3 rows")
    def alpha(M):
        kk = M.shape[1]
        return kk / (kk - 1) * (1 - M.var(axis=0, ddof=1).sum() / M.sum(axis=1).var(ddof=1))
    a = float(alpha(X))
    std = X - X.mean(0)
    Rm = np.corrcoef(X, rowvar=False)
    rbar = float((Rm.sum() - k) / (k * (k - 1)))
    a_std = k * rbar / (1 + (k - 1) * rbar)
    rows = []
    for j in range(k):
        rest = np.delete(X, j, axis=1)
        rows.append([names[j], float(alpha(rest)) if k > 2 else float("nan"), float(np.corrcoef(X[:, j], rest.sum(1))[0, 1])])
    r = Result("Cronbach's alpha")
    r.kv("", [("items", k), ("observations (complete)", n), ("alpha (raw)", a), ("alpha (standardised)", float(a_std)), ("mean inter-item correlation", rbar)])
    r.table("Item statistics", ["Item", "Alpha if deleted", "Item-rest correlation"], rows)
    r.values.update(dict(alpha=a, alpha_std=float(a_std)))
    return r


# =========================================================================== Mahalanobis
@analysis("mahalanobis", "Multivariate outliers (Mahalanobis distance)", CAT, [XS], [ALPHA],
          code={"stata": "* Mahalanobis: see Python/R", "r": "X <- na.omit(df[, c({xs_q})]); d2 <- mahalanobis(X, colMeans(X), cov(X)); which(d2 > qchisq(1 - {alpha}, ncol(X)))",
                "python": "import numpy as np; X = df[[{xs_q}]].dropna().values; d2 = np.einsum('ij,jk,ik->i', X-X.mean(0), np.linalg.inv(np.cov(X.T)), X-X.mean(0))"},
          reference="D² = (x−x̄)'S⁻¹(x−x̄) ~ χ²(p) approximately under multivariate normality; rows with p < α are flagged.")
def mahalanobis(d, alpha=0.05):
    X = complete_matrix(d["xs"])
    n, p = X.shape
    need(n > p + 1, "need more rows than variables")
    dc = X - X.mean(0)
    D2 = np.einsum("ij,jk,ik->i", dc, np.linalg.pinv(np.cov(X, rowvar=False).reshape(p, p)), dc)
    pv = stats.chi2.sf(D2, p)
    flag = np.flatnonzero(pv < alpha)
    r = Result("Mahalanobis distance outlier screen")
    r.kv("", [("complete rows", n), ("variables", p), ("critical D² (χ²)", float(stats.chi2.isf(alpha, p))), (f"rows flagged at α = {alpha:g}", int(flag.size))])
    top = np.argsort(-D2)[:15]
    r.table("Largest distances (row = position among complete rows, 1-based)", ["Row", "D²", "p-value"], [[int(i) + 1, float(D2[i]), float(pv[i])] for i in top])
    r.values.update(dict(D2=D2, flagged=flag))
    return r


# =========================================================================== canonical correlation
@analysis("cancorr", "Canonical correlation analysis", CAT, [Role("xs", "numeric_multi", "Set X"), Role("ys", "numeric_multi", "Set Y")],
          code={"stata": "canon ({xs}) ({ys})", "r": "cancor(df[, c({xs_q})], df[, c({ys_q})])", "python": "from statsmodels.multivariate.cancorr import CanCorr; CanCorr(df[[{ys_q}]], df[[{xs_q}]]).corr_test()"},
          reference="Canonical correlations via SVD of Qx'Qy (QR of centred sets); Wilks' Λ with Bartlett's χ² approximation tests all correlations ≥ k (Bartlett 1941).")
def cancorr(d):
    cols = list(d["xs"]) + list(d["ys"])
    M = complete_matrix(cols)
    px = len(d["xs"])
    X, Y = M[:, :px], M[:, px:]
    n, py = Y.shape
    need(n > px + py + 1, "need more rows than total variables + 1")
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    s = np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), 0, 1)
    m = len(s)
    r = Result("Canonical correlation analysis")
    rows = []
    for i in range(m):
        lam = float(np.prod(1 - s[i:] ** 2))
        chi = -(n - 1 - (px + py + 1) / 2) * np.log(lam)
        df = (px - i) * (py - i)
        rows.append([i + 1, float(s[i]), float(s[i] ** 2), lam, float(chi), df, float(stats.chi2.sf(chi, df))])
    r.kv("", [("observations", n), ("X variables", px), ("Y variables", py)])
    r.table("Canonical correlations (test: this and all later correlations are zero)", ["Pair", "Canonical r", "r²", "Wilks' Λ", "χ²", "df", "p-value"], rows)
    r.values.update(dict(corr=s, p=[row[-1] for row in rows]))
    return r


# =========================================================================== multivariate regression
@analysis("mvreg", "Multivariate multiple regression (several outcomes)", CAT,
          [Role("ys", "numeric_multi", "Outcomes (Y)"), Role("xs", "numeric_multi", "Predictors (X)")],
          code={"stata": "mvreg {ys} = {xs}", "r": "fit <- lm(cbind({ys}) ~ {xs_plus}, data = df); summary(fit); car::Manova(fit)",
                "python": "import statsmodels.api as sm; [sm.OLS(df[y], sm.add_constant(df[[{xs_q}]])).fit().summary() for y in [{ys_q}]]"},
          reference="Equation-by-equation OLS (identical coefficients to multivariate least squares B = (X'X)⁻¹X'Y); overall test of each predictor via "
                    "Wilks' Λ with Rao's F approximation, as in Stata `mvreg` / `mvtest`.")
def mvreg(d):
    import statsmodels.api as sm
    ys, xs = list(d["ys"]), list(d["xs"])
    yn, xn = d.label_list("ys"), d.label_list("xs")
    M = complete_matrix(ys + xs)
    q, p = len(ys), len(xs)
    Y, X = M[:, :q], M[:, q:]
    n = len(M)
    need(q >= 2 and n > p + q + 1, "need at least 2 outcomes and enough rows")
    Xc = sm.add_constant(X)
    B = np.linalg.lstsq(Xc, Y, rcond=None)[0]
    E = Y - Xc @ B
    Esscp = E.T @ E
    r = Result("Multivariate multiple regression")
    r.kv("", [("observations", n), ("outcomes", q), ("predictors", p)])
    eq = []
    for j in range(q):
        fit = sm.OLS(Y[:, j], Xc).fit()
        eq.append([yn[j], p, float(np.sqrt(fit.mse_resid)), float(fit.rsquared), float(fit.fvalue), float(fit.f_pvalue)])
        r.table(f"Equation: {yn[j]}", ["Term", "Coef.", "Std. err.", "t", "p-value"],
                [[t, float(fit.params[i]), float(fit.bse[i]), float(fit.tvalues[i]), float(fit.pvalues[i])] for i, t in enumerate(["(intercept)"] + xn)])
    r.table("Equation summary", ["Outcome", "Parms", "RMSE", "R²", "F", "p-value"], eq)
    # Wilks lambda for each predictor (drop-one)
    rows = []
    dfe = n - p - 1
    for j in range(p):
        Xr = np.delete(Xc, j + 1, axis=1)
        Er = Y - Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]
        lam = float(np.linalg.det(Esscp) / np.linalg.det(Er.T @ Er))
        qh = 1
        t = np.sqrt((q ** 2 * qh ** 2 - 4) / (q ** 2 + qh ** 2 - 5)) if q ** 2 + qh ** 2 - 5 > 0 else 1.0
        df1 = q * qh
        df2 = t * (dfe - (q - qh + 1) / 2) - (q * qh - 2) / 2
        F = (1 - lam ** (1 / t)) / lam ** (1 / t) * df2 / df1
        rows.append([xn[j], lam, float(F), float(df1), float(df2), float(stats.f.sf(F, df1, df2))])
    r.table("Multivariate test of each predictor (Wilks' Λ, Rao's F)", ["Predictor", "Wilks' Λ", "F", "df1", "df2", "p-value"], rows)
    r.values.update(dict(coef=B, equations=eq, wilks=[row[1] for row in rows], p=[row[-1] for row in rows]))
    return r
