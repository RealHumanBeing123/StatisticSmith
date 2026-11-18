"""Rank-based / distribution-free tests."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..charts.base import Param, Role
from .base import Result, analysis, clean, clean_pair
from .common import ALT, CONF, GROUP, XS, Y, alt_label, groups_of, need, p_from_z, sig_note, tie_term, two_groups

CAT = "Nonparametric & rank tests"


def _rank_summary(r, items):
    r.table("Group summary", ["group", "n", "median", "mean rank"],
            [[n, a.size, float(np.median(a)), float(m)] for n, a, m in items])


@analysis("mann_whitney", "Mann–Whitney U / Wilcoxon rank-sum test", CAT, [Y, GROUP],
          [ALT, Param("method", "p-value method", "choice", "auto", choices=("auto", "exact", "asymptotic")),
           Param("continuity", "Continuity correction", "bool", True)],
          code={"stata": "ranksum {y}, by({group})", "r": "wilcox.test({y_r} ~ {group_r}, data = df, alternative = \"{alternative}\", exact = {exact_r}, correct = {continuity_r})",
                "python": "import scipy.stats as st; g=[v.dropna() for _,v in df.groupby('{group}')['{y}']]; st.mannwhitneyu(g[0], g[1], alternative='{alternative}', method='{method}', use_continuity={continuity_r})"},
          reference="U1 = R1 - n1(n1+1)/2. Asymptotic z uses the tie-corrected variance n1n2/12·[(N+1) - Σ(t³-t)/(N(N-1))].")
def mann_whitney(d, alternative="two-sided", method="auto", continuity=True):
    a, x, b, y = two_groups(d, "Mann–Whitney U test")
    res = stats.mannwhitneyu(x, y, alternative=alternative, method=method, use_continuity=continuity)
    n1, n2 = x.size, y.size
    allv = np.concatenate([x, y])
    rk = stats.rankdata(allv)
    r1 = rk[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    r = Result(f"Mann–Whitney U test: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", f"H0: P(X > Y) + 0.5·P(X = Y) = 0.5  (distributions equal / no location shift)\nGroups: X = {a}, Y = {b}")
    _rank_summary(r, [(a, x, rk[:n1].mean()), (b, y, rk[n1:].mean())])
    r.kv("Test", [("U (group 1)", u1), ("U (group 2)", u2), ("scipy statistic (U1)", float(res.statistic)), ("p-value", float(res.pvalue)),
                  ("rank-biserial r (group1 vs group2)", 2 * u1 / (n1 * n2) - 1),
                  ("common-language effect P(X>Y)", u1 / (n1 * n2))])
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(U=float(res.statistic), U1=u1, U2=u2, p=float(res.pvalue)))
    return r


@analysis("wilcoxon", "Wilcoxon signed-rank test", CAT, [Role("x", "numeric", "Variable / first measure"), Role("y", "numeric", "Second measure (blank = test x against m0)", required=False)],
          [Param("m0", "Hypothesised median (one-sample)", "float", 0.0), ALT, Param("method", "p-value method", "choice", "auto", choices=("auto", "exact", "approx")),
           Param("zero_method", "Zero differences", "choice", "wilcox", choices=("wilcox", "pratt", "zsplit"))],
          code={"stata": "signrank {x} = {y}", "r": "wilcox.test(df${x}, df${y}, paired = TRUE, alternative = \"{alternative}\")",
                "python": "import scipy.stats as st; st.wilcoxon(df['{x}'] - df['{y}'], alternative='{alternative}', method='{method}', zero_method='{zero_method}')"},
          reference="W+ = sum of ranks of positive differences |d_i| (zero differences dropped by default).")
def wilcoxon(d, m0=0.0, alternative="two-sided", method="auto", zero_method="wilcox"):
    if d.has("y"):
        x, y = clean_pair(d["x"], d["y"])
        diff = x - y
        what = f"{d.name('x')} − {d.name('y')}"
    else:
        diff = clean(d["x"]) - m0
        what = f"{d.name('x')} − {m0:g}"
    need(diff.size > 0, "no data")
    nz = diff[diff != 0]
    need(nz.size > 0, "all differences are zero")
    res = stats.wilcoxon(diff, alternative=alternative, method=method, zero_method=zero_method)
    rk = stats.rankdata(np.abs(nz))
    wp, wm = rk[nz > 0].sum(), rk[nz < 0].sum()
    r = Result(f"Wilcoxon signed-rank test: {what}")
    r.text("Hypotheses", "H0: the distribution of differences is symmetric about 0")
    r.kv("Test", [("n (non-zero differences)", nz.size), ("zeros dropped", diff.size - nz.size), ("W+ (positive ranks)", wp), ("W− (negative ranks)", wm),
                  ("scipy statistic", float(res.statistic)), ("p-value", float(res.pvalue)),
                  ("matched-pairs rank-biserial", (wp - wm) / (wp + wm))])
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(Wplus=wp, Wminus=wm, p=float(res.pvalue), n=nz.size))
    return r


@analysis("sign_test", "Sign test", CAT, [Role("x", "numeric", "Variable / first measure"), Role("y", "numeric", "Second measure (blank = test x against m0)", required=False)],
          [Param("m0", "Hypothesised median (one-sample)", "float", 0.0), ALT],
          code={"stata": "signtest {x} = {y}", "r": "d <- df${x} - df${y}; binom.test(sum(d > 0), sum(d != 0), 0.5, alternative = \"{alternative}\")",
                "python": "import scipy.stats as st; d=(df['{x}']-df['{y}']).dropna(); st.binomtest((d>0).sum(), (d!=0).sum(), 0.5, alternative='{alternative}')"},
          reference="Exact binomial on the number of positive differences, ties (zeros) dropped.")
def sign_test(d, m0=0.0, alternative="two-sided"):
    if d.has("y"):
        x, y = clean_pair(d["x"], d["y"])
        diff, what = x - y, f"{d.name('x')} − {d.name('y')}"
    else:
        diff, what = clean(d["x"]) - m0, f"{d.name('x')} − {m0:g}"
    pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
    n = pos + neg
    need(n > 0, "all differences are zero")
    bt = stats.binomtest(pos, n, 0.5, alternative=alternative)
    r = Result(f"Sign test: {what}")
    r.text("Hypotheses", "H0: median difference = 0  (P(positive) = P(negative) = 0.5)")
    r.kv("Test", [("positive", pos), ("negative", neg), ("ties (dropped)", int((diff == 0).sum())), ("p-value (exact binomial)", bt.pvalue)])
    r.text("Conclusion", sig_note(bt.pvalue))
    r.values.update(dict(pos=pos, neg=neg, p=bt.pvalue))
    return r


@analysis("kruskal", "Kruskal–Wallis H test", CAT, [Y, GROUP],
          [Param("n_perm", "Monte-Carlo permutations for exact-style p (0 = off)", "int", 0, min=0, max=2_000_000), Param("seed", "Random seed", "int", 12345)],
          code={"stata": "kwallis {y}, by({group})", "r": "kruskal.test({y_r} ~ {group_r}, data = df)",
                "python": "import scipy.stats as st; st.kruskal(*[v.dropna() for _,v in df.groupby('{group}')['{y}']])"},
          reference="H = [12/(N(N+1)) Σ R_i²/n_i - 3(N+1)] / [1 - Σ(t³-t)/(N³-N)] ~ χ²(k-1).")
def kruskal(d, n_perm=0, seed=12345):
    gs = groups_of(d["y"], d["group"])
    need(len(gs) >= 2, "need at least 2 groups")
    arrs = list(gs.values())
    res = stats.kruskal(*arrs)
    allv = np.concatenate(arrs)
    rk = stats.rankdata(allv)
    N = allv.size
    pos = np.cumsum([0] + [a.size for a in arrs])
    mr = [rk[pos[i]:pos[i + 1]].mean() for i in range(len(arrs))]
    r = Result(f"Kruskal–Wallis test: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", "H0: all groups come from the same distribution")
    _rank_summary(r, [(n, a, m) for (n, a), m in zip(gs.items(), mr)])
    pairs = [("H statistic", float(res.statistic)), ("df", len(arrs) - 1), ("p-value (χ² approximation)", float(res.pvalue)),
             ("epsilon²", float(res.statistic) / (N - 1) if N > 1 else np.nan)]
    p_final = float(res.pvalue)
    if n_perm:
        rng = np.random.default_rng(int(seed))
        cnt = 0
        h0 = float(res.statistic)
        for _ in range(int(n_perm)):
            perm = rng.permutation(allv)
            h = stats.kruskal(*[perm[pos[i]:pos[i + 1]] for i in range(len(arrs))]).statistic
            cnt += h >= h0 - 1e-12
        pm = (cnt + 1) / (n_perm + 1)
        pairs.append(("p-value (permutation)", pm))
    r.kv("Test", pairs)
    r.text("Conclusion", sig_note(p_final))
    r.values.update(dict(H=float(res.statistic), p=p_final, df=len(arrs) - 1))
    return r


@analysis("friedman", "Friedman test (repeated measures)", CAT, [Role("xs", "numeric_multi", "Measurements (one column per condition; rows = subjects)")],
          code={"stata": "* Stata: friedman via user-written command (ssc install friedman) - or reshape long and use xtfrch", "r": "friedman.test(as.matrix(na.omit(df[, c({xs_q})])))",
                "python": "import scipy.stats as st; d=df[[{xs_q}]].dropna(); st.friedmanchisquare(*[d[c] for c in d.columns])"},
          reference="Q = 12/(nk(k+1)) Σ R_j² - 3n(k+1), tie-corrected, ~ χ²(k-1). Rows with missing values are dropped.")
def friedman(d):
    names = d.label_list("xs")
    M = np.column_stack([np.asarray(c, dtype=float) for c in d["xs"]])
    M = M[np.all(np.isfinite(M), axis=1)]
    n, k = M.shape
    need(k >= 3, "Friedman needs at least 3 conditions (use Wilcoxon for 2)")
    need(n >= 2, "need at least 2 complete rows")
    res = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    R = np.apply_along_axis(stats.rankdata, 1, M)
    r = Result("Friedman test")
    r.text("Hypotheses", "H0: the k conditions have identical distributions within subjects")
    r.table("Mean ranks", ["condition", "mean rank", "median"], [[names[j], R[:, j].mean(), float(np.median(M[:, j]))] for j in range(k)])
    W = float(res.statistic) / (n * (k - 1))
    r.kv("Test", [("subjects (complete rows)", n), ("χ² statistic", float(res.statistic)), ("df", k - 1), ("p-value", float(res.pvalue)), ("Kendall's W", W)])
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(chi2=float(res.statistic), p=float(res.pvalue), W=W))
    return r


@analysis("median_test", "Mood's median test", CAT, [Y, GROUP], [Param("correction", "Yates correction", "bool", True)],
          code={"stata": "median {y}, by({group})", "r": "# Mood's median test\nm <- median(df${y}, na.rm=TRUE); chisq.test(table(df${group}, df${y} > m))",
                "python": "import scipy.stats as st; st.median_test(*[v.dropna() for _,v in df.groupby('{group}')['{y}']])"},
          reference="Pearson χ² on the group × (above / not above the grand median) table.")
def median_test(d, correction=True):
    gs = groups_of(d["y"], d["group"])
    need(len(gs) >= 2, "need at least 2 groups")
    stat, p, med, tbl = stats.median_test(*gs.values(), correction=correction)
    r = Result(f"Median test: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", "H0: all groups have the same median")
    r.table("Counts above / not above the grand median", ["group", "above", "not above"], [[n, int(tbl[0, i]), int(tbl[1, i])] for i, n in enumerate(gs)])
    r.kv("Test", [("grand median", float(med)), ("χ² statistic", float(stat)), ("df", len(gs) - 1), ("p-value", float(p))])
    r.text("Conclusion", sig_note(float(p)))
    r.values.update(dict(chi2=float(stat), p=float(p), median=float(med)))
    return r


def siegel_tukey_ranks(x, y):
    """Siegel–Tukey ranks of the pooled sample (extreme-alternating ranking), ties averaged.

    Order of the sorted pooled sample gets ranks: 1 to the smallest, 2-3 to the two largest, 4-5 to the next two smallest,
    6-7 to the next two largest, ... Returns (ranks for x, ranks for y).
    """
    z = np.concatenate([x, y])
    lab = np.concatenate([np.zeros(len(x), int), np.ones(len(y), int)])
    o = np.argsort(z, kind="stable")
    zs, ls = z[o], lab[o]
    N = len(z)
    rank = np.zeros(N)
    lo, hi, r = 0, N - 1, 1
    # first rank goes to the low end, then two ranks alternately to the high end and the low end
    rank[lo] = r
    lo += 1
    r += 1
    take_high = True
    while lo <= hi:
        for _ in range(2):
            if lo > hi:
                break
            if take_high:
                rank[hi] = r
                hi -= 1
            else:
                rank[lo] = r
                lo += 1
            r += 1
        take_high = not take_high
    # average over ties in the *value* (as in Siegel & Tukey 1960 / R package RVAideMemoire)
    out = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j + 1 < N and zs[j + 1] == zs[i]:
            j += 1
        out[i:j + 1] = rank[i:j + 1].mean()
        i = j + 1
    return out[ls == 0], out[ls == 1]


@analysis("siegel_tukey", "Siegel–Tukey test (equal dispersion)", CAT, [Y, GROUP],
          [Param("center", "Centre samples at their medians first", "bool", False), ALT, Param("continuity", "Continuity correction", "bool", True)],
          code={"stata": "* Siegel-Tukey: no built-in; see the Python / R versions", "r": "# install.packages('RVAideMemoire'); RVAideMemoire::siegel.tukey({y_r} ~ {group_r}, data = df)",
                "python": "# statsmith.stats.nonparametric.siegel_tukey_ranks then scipy.stats.mannwhitneyu on the ranks"},
          reference="Siegel & Tukey (1960): Wilcoxon rank-sum applied to extreme-alternating ranks; a small rank sum for a group means larger spread.")
def siegel_tukey(d, center=False, alternative="two-sided", continuity=True):
    a, x, b, y = two_groups(d, "Siegel–Tukey test")
    xx, yy = (x - np.median(x), y - np.median(y)) if center else (x, y)
    rx, ry = siegel_tukey_ranks(xx, yy)
    n1, n2 = len(x), len(y)
    N = n1 + n2
    R1 = rx.sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    allr = np.concatenate([rx, ry])
    var = n1 * n2 / 12 * ((N + 1) - tie_term(allr) / (N * (N - 1)))
    cc = 0.5 if continuity else 0.0
    diff = U1 - mu
    if alternative == "two-sided":
        z = (abs(diff) - cc) / np.sqrt(var) * np.sign(diff) if var > 0 else np.nan
        p = min(1.0, 2 * stats.norm.sf(abs(z)))
    elif alternative == "less":
        z = (diff + cc) / np.sqrt(var)
        p = stats.norm.cdf(z)
    else:
        z = (diff - cc) / np.sqrt(var)
        p = stats.norm.sf(z)
    r = Result(f"Siegel–Tukey test: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", "H0: the two groups have equal dispersion (same spread) — assumes equal medians; use 'centre' otherwise.\n"
                         "H1 (two-sided): the spreads differ. A smaller rank sum indicates larger dispersion.")
    r.kv("Test", [(f"rank sum {a}", R1), (f"rank sum {b}", float(ry.sum())), ("U", U1), ("z", z), ("p-value (normal approx.)", float(p))])
    r.text("Conclusion", sig_note(float(p)))
    r.values.update(dict(R1=R1, R2=float(ry.sum()), U1=U1, z=z, p=float(p)))
    return r


@analysis("ks_two", "Kolmogorov–Smirnov test (two samples)", CAT, [Y, GROUP], [ALT, Param("method", "p-value method", "choice", "auto", choices=("auto", "exact", "asymp"))],
          code={"stata": "ksmirnov {y}, by({group})", "r": "ks.test(df${y}[df${group} == levels(factor(df${group}))[1]], df${y}[df${group} == levels(factor(df${group}))[2]])",
                "python": "import scipy.stats as st; g=[v.dropna() for _,v in df.groupby('{group}')['{y}']]; st.ks_2samp(g[0], g[1], method='{method}')"},
          reference="D = sup|F1(x) - F2(x)|.")
def ks_two(d, alternative="two-sided", method="auto"):
    a, x, b, y = two_groups(d, "Two-sample KS test")
    res = stats.ks_2samp(x, y, alternative=alternative, method=method)
    r = Result(f"Two-sample Kolmogorov–Smirnov test: {d.name('y')} by {d.name('group')}")
    r.text("Hypotheses", "H0: both samples come from the same continuous distribution")
    r.kv("Test", [("D statistic", float(res.statistic)), ("p-value", float(res.pvalue)), ("n1", x.size), ("n2", y.size)])
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(D=float(res.statistic), p=float(res.pvalue)))
    return r
