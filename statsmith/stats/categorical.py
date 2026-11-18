"""Contingency tables, exact tests, binomial / multinomial, McNemar, Cochran's Q."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats
from scipy.special import gammaln

from ..charts.base import Param, Role
from .base import Result, analysis, fmt_p
from .common import ALT, CONF, sig_note, need
from .location import _binary

CAT = "Categorical & contingency"


def crosstab(a, b):
    """Counts table from two categorical columns. Returns (row labels, col labels, int matrix)."""
    def keys(v):
        return np.array(["(missing)" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in np.asarray(v, dtype=object).tolist()], dtype=object)
    ka, kb = keys(a), keys(b)
    ok = (ka != "(missing)") & (kb != "(missing)")
    ka, kb = ka[ok], kb[ok]
    ra, ca = list(dict.fromkeys(ka.tolist())), list(dict.fromkeys(kb.tolist()))
    ia = {v: i for i, v in enumerate(ra)}
    ib = {v: i for i, v in enumerate(ca)}
    M = np.zeros((len(ra), len(ca)), dtype=np.int64)
    np.add.at(M, ([ia[v] for v in ka.tolist()], [ib[v] for v in kb.tolist()]), 1)
    return ra, ca, M


def _tbl_text(r, title, rows, cols, M, row_name="", col_name=""):
    r.table(title, [f"{row_name}\\{col_name}"] + cols + ["total"],
            [[rows[i]] + [int(v) for v in M[i]] + [int(M[i].sum())] for i in range(len(rows))] + [["total"] + [int(v) for v in M.sum(0)] + [int(M.sum())]])


@analysis("chi2_indep", "Pearson χ² test of independence", CAT, [Role("row", "categorical", "Row variable"), Role("col", "categorical", "Column variable")],
          [Param("correction", "Yates continuity correction (2×2 only)", "bool", False), Param("show_expected", "Show expected counts", "bool", True)],
          code={"stata": "tabulate {row} {col}, chi2 expected", "r": "chisq.test(table(df${row}, df${col}), correct = {correction_r})",
                "python": "import scipy.stats as st; st.chi2_contingency(pd.crosstab(df['{row}'], df['{col}']), correction={correction_r})"},
          reference="χ² = Σ (O-E)²/E, E = row·col/N, df = (r-1)(c-1). Valid when most expected counts ≥ 5 (otherwise use Fisher's exact test). "
                    "Cramér's V = √(χ²/(N·min(r-1,c-1))).")
def chi2_indep(d, correction=False, show_expected=True):
    rows, cols, M = crosstab(d["row"], d["col"])
    need(M.shape[0] >= 2 and M.shape[1] >= 2, "both variables need at least 2 levels")
    chi2, p, dof, E = stats.chi2_contingency(M, correction=correction)
    N = M.sum()
    V = float(np.sqrt(chi2 / (N * (min(M.shape) - 1))))
    r = Result(f"Pearson χ² test of independence: {d.name('row')} × {d.name('col')}")
    r.text("Hypotheses", "H0: the two variables are independent")
    _tbl_text(r, "Observed counts", rows, cols, M, d.name("row"), d.name("col"))
    if show_expected:
        r.table("Expected counts under H0", [""] + cols, [[rows[i]] + list(E[i]) for i in range(len(rows))], decimals=3)
    r.kv("Test", [("χ² statistic", float(chi2)), ("df", int(dof)), ("p-value", float(p)), ("N", int(N)), ("Cramér's V", V),
                  ("cells with expected < 5", int((E < 5).sum())), ("min expected", float(E.min()))])
    if (E < 5).mean() > 0.2:
        r.note("More than 20% of cells have expected counts below 5: the χ² approximation may be unreliable — consider Fisher's exact test.")
    if M.shape == (2, 2) and not correction:
        r.note("2×2 table: tick the Yates correction option for the continuity-corrected statistic.")
    r.text("Conclusion", sig_note(float(p)))
    r.values.update(dict(chi2=float(chi2), df=int(dof), p=float(p), V=V, expected=E, observed=M))
    return r


@analysis("chi2_gof", "χ² goodness-of-fit test", CAT, [Role("x", "categorical", "Variable")],
          [Param("probs", "Expected proportions (comma-separated, in order of first appearance; blank = equal)", "text", "")],
          code={"stata": "* Stata: csgof or tab {x} then chitest;  use R/Python for arbitrary expected proportions", "r": "chisq.test(table(df${x}), p = c({probs_r}))",
                "python": "import scipy.stats as st; st.chisquare(df['{x}'].value_counts(sort=False), f_exp=len(df['{x}'].dropna())*np.array([{probs_r}]))"},
          reference="χ² = Σ (O_i - N p_i)²/(N p_i), df = k-1.")
def chi2_gof(d, probs=""):
    x = [str(v) for v in np.asarray(d["x"], dtype=object).tolist() if not (v is None or (isinstance(v, float) and np.isnan(v)))]
    lv = list(dict.fromkeys(x))
    obs = np.array([x.count(l) for l in lv], dtype=float)
    k = len(lv)
    need(k >= 2, "need at least 2 categories")
    p = np.full(k, 1 / k) if not probs.strip() else np.array([float(t) for t in probs.replace(";", ",").split(",") if t.strip()])
    need(len(p) == k, f"expected {k} proportions (levels: {', '.join(lv)}), got {len(p)}")
    need(abs(p.sum() - 1) < 1e-6 or p.sum() > 0, "proportions must be positive")
    p = p / p.sum()
    E = obs.sum() * p
    chi2, pv = stats.chisquare(obs, E)
    r = Result(f"χ² goodness-of-fit: {d.name('x')}")
    r.table("", ["level", "observed", "expected", "(O-E)²/E"], [[lv[i], obs[i], E[i], (obs[i] - E[i]) ** 2 / E[i]] for i in range(k)])
    r.kv("Test", [("χ² statistic", float(chi2)), ("df", k - 1), ("p-value", float(pv))])
    r.text("Conclusion", sig_note(float(pv)))
    r.values.update(dict(chi2=float(chi2), p=float(pv)))
    return r


# --------------------------------------------------------------------------- Fisher exact (r x c) -----------------------
def _log_prob_table(M, rows, cols, n):
    return (gammaln(rows + 1).sum() + gammaln(cols + 1).sum() - gammaln(n + 1) - gammaln(M + 1).sum())


def fisher_rxc_exact(M, max_nodes=3_000_000, n_mc=200_000, seed=12345):
    """Freeman–Halton exact p-value for an r×c table (sum of probabilities of tables ≤ P(observed)).

    Enumerates tables with fixed margins by depth-first search over rows; falls back to Monte-Carlo
    (scipy.stats.random_table) when the search exceeds ``max_nodes``. Returns (p, method).
    """
    M = np.asarray(M, dtype=np.int64)
    if M.shape[0] > M.shape[1]:
        M = M.T
    r, c = M.shape
    rows, cols, n = M.sum(1), M.sum(0), int(M.sum())
    lp_obs = _log_prob_table(M, rows, cols, n)
    tol = 1e-7
    count = [0]
    total = [0.0]

    class Overflow(Exception):
        pass

    # enumerate column by column: each column is a composition of col sum into r parts bounded by remaining row sums
    def rec(j, rem_rows, acc_lp):
        count[0] += 1
        if count[0] > max_nodes:
            raise Overflow
        if j == c - 1:
            full = acc_lp - gammaln(rem_rows + 1).sum() + const
            if full <= lp_obs + tol:
                total[0] += math.exp(full)
            return
        for col in compositions(int(cols[j]), rem_rows):
            rec(j + 1, rem_rows - col, acc_lp - gammaln(col + 1).sum())

    def compositions(s, caps):
        k = len(caps)
        out = [0] * k

        def go(i, left):
            if i == k - 1:
                if left <= caps[i]:
                    out[i] = left
                    yield np.array(out)
                return
            tail = int(sum(caps[i + 1:]))
            for v in range(max(0, left - tail), min(int(caps[i]), left) + 1):
                out[i] = v
                yield from go(i + 1, left - v)
        yield from go(0, s)

    const = gammaln(rows + 1).sum() + gammaln(cols + 1).sum() - gammaln(n + 1)
    # note: P(table) = const - sum gammaln(M+1); rec accumulates -sum gammaln(M+1) (last column handled at the leaf)
    try:
        rec(0, rows.astype(np.int64), 0.0)
        return min(1.0, total[0]), "exact (Freeman–Halton enumeration)"
    except Overflow:
        rng = np.random.default_rng(seed)
        rt = stats.random_table(rows, cols, seed=rng)
        sims = rt.rvs(n_mc)
        lps = np.array([_log_prob_table(s, rows, cols, n) for s in sims])
        p = (np.sum(lps <= lp_obs + tol) + 1) / (n_mc + 1)
        return float(p), f"Monte-Carlo ({n_mc:,} random tables, seed {seed})"


@analysis("fisher", "Fisher's exact test", CAT, [Role("row", "categorical", "Row variable"), Role("col", "categorical", "Column variable")],
          [ALT, Param("seed", "Random seed (large r×c tables only)", "int", 12345)],
          code={"stata": "tabulate {row} {col}, exact", "r": "fisher.test(table(df${row}, df${col}), alternative = \"{alternative}\")",
                "python": "import scipy.stats as st; st.fisher_exact(pd.crosstab(df['{row}'], df['{col}']), alternative='{alternative}')"},
          reference="Hypergeometric probability of the table given fixed margins; 2×2 uses scipy.stats.fisher_exact (conditional MLE odds ratio "
                    "is NOT reported, the sample odds ratio is); r×c uses the Freeman–Halton extension.")
def fisher(d, alternative="two-sided", seed=12345):
    rows, cols, M = crosstab(d["row"], d["col"])
    need(M.shape[0] >= 2 and M.shape[1] >= 2, "both variables need at least 2 levels")
    r = Result(f"Fisher's exact test: {d.name('row')} × {d.name('col')}")
    _tbl_text(r, "Observed counts", rows, cols, M, d.name("row"), d.name("col"))
    if M.shape == (2, 2):
        orr, p = stats.fisher_exact(M, alternative=alternative)
        r.kv("Test", [("sample odds ratio", float(orr)), ("p-value", float(p))])
        r.values.update(dict(odds=float(orr), p=float(p)))
    else:
        p, how = fisher_rxc_exact(M, seed=int(seed))
        r.kv("Test", [("p-value", p)])
        r.note(f"r×c table: {how}. Alternative is always two-sided.")
        r.values.update(dict(p=float(p)))
    r.text("Conclusion", sig_note(r.values["p"]))
    return r


def _two_by_two(d):
    rows, cols, M = crosstab(d["row"], d["col"])
    need(M.shape == (2, 2), f"this test needs a 2×2 table (got {M.shape[0]}×{M.shape[1]})")
    return rows, cols, M


@analysis("barnard", "Barnard's exact test (2×2)", CAT, [Role("row", "categorical", "Row variable (2 levels)"), Role("col", "categorical", "Column variable (2 levels)")],
          [ALT, Param("pooled", "Pooled variance statistic", "bool", True)],
          code={"stata": "* Barnard: use R/Python", "r": "# install.packages('Barnard'); Barnard::barnard.test({a}, {b}, {c}, {d})", "python": "import scipy.stats as st; st.barnard_exact(pd.crosstab(df['{row}'], df['{col}']).values, alternative='{alternative}', pooled={pooled_r})"},
          reference="Unconditional exact test (Barnard 1945) via scipy.stats.barnard_exact: maximises p over the nuisance parameter.")
def barnard(d, alternative="two-sided", pooled=True):
    rows, cols, M = _two_by_two(d)
    res = stats.barnard_exact(M, alternative=alternative, pooled=pooled)
    r = Result(f"Barnard's exact test: {d.name('row')} × {d.name('col')}")
    _tbl_text(r, "Observed counts", rows, cols, M, d.name("row"), d.name("col"))
    r.kv("Test", [("Wald statistic", float(res.statistic)), ("p-value", float(res.pvalue))])
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(stat=float(res.statistic), p=float(res.pvalue)))
    return r


@analysis("boschloo", "Boschloo's exact test (2×2)", CAT, [Role("row", "categorical", "Row variable (2 levels)"), Role("col", "categorical", "Column variable (2 levels)")],
          [ALT],
          code={"stata": "* Boschloo: use R/Python", "r": "# install.packages('exact2x2'); exact2x2::boschloo(x1, n1, x2, n2)", "python": "import scipy.stats as st; st.boschloo_exact(pd.crosstab(df['{row}'], df['{col}']).values, alternative='{alternative}')"},
          reference="Boschloo (1970): unconditional test using Fisher's p-value as the test statistic; uniformly more powerful than Fisher's test.")
def boschloo(d, alternative="two-sided"):
    rows, cols, M = _two_by_two(d)
    res = stats.boschloo_exact(M, alternative=alternative)
    r = Result(f"Boschloo's exact test: {d.name('row')} × {d.name('col')}")
    _tbl_text(r, "Observed counts", rows, cols, M, d.name("row"), d.name("col"))
    r.kv("Test", [("statistic (Fisher p)", float(res.statistic)), ("p-value", float(res.pvalue))])
    r.text("Conclusion", sig_note(float(res.pvalue)))
    r.values.update(dict(p=float(res.pvalue)))
    return r


@analysis("mcnemar", "McNemar's test (paired binary)", CAT, [Role("a", "categorical", "Measure 1 (binary)"), Role("b", "categorical", "Measure 2 (binary)")],
          [Param("exact", "Exact binomial version", "bool", False), Param("correction", "Continuity correction (χ² version)", "bool", True)],
          code={"stata": "mcc {a} {b}", "r": "mcnemar.test(table(df${a}, df${b}), correct = {correction_r})",
                "python": "from statsmodels.stats.contingency_tables import mcnemar; mcnemar(pd.crosstab(df['{a}'], df['{b}']).values, exact={exact_r}, correction={correction_r})"},
          reference="Uses discordant pairs b, c only: χ² = (|b-c|-1)²/(b+c) ~ χ²(1) (corrected); exact: 2·P(Bin(b+c, .5) ≤ min(b,c)).")
def mcnemar(d, exact=False, correction=True):
    rows, cols, M = crosstab(d["a"], d["b"])
    need(M.shape == (2, 2), "both measures must be binary and use the same two levels")
    b, c = int(M[0, 1]), int(M[1, 0])
    need(b + c > 0, "no discordant pairs: the test is undefined")
    chi2 = (abs(b - c) - (1 if correction else 0)) ** 2 / (b + c)
    p_chi = float(stats.chi2.sf(chi2, 1))
    p_ex = float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))
    r = Result(f"McNemar's test: {d.name('a')} vs {d.name('b')}")
    r.text("Hypotheses", "H0: the marginal proportions are equal (P(b) = P(c) among discordant pairs)")
    _tbl_text(r, "Paired table", rows, cols, M, d.name("a"), d.name("b"))
    r.kv("Test", [("discordant b", b), ("discordant c", c), ("χ² statistic", chi2), ("p-value (χ²)", p_chi), ("p-value (exact binomial)", p_ex)])
    p = p_ex if exact else p_chi
    r.text("Conclusion", sig_note(p) + ("  (exact)" if exact else "  (χ²)"))
    r.values.update(dict(chi2=chi2, p_chi2=p_chi, p_exact=p_ex, p=p, b=b, c=c))
    return r


@analysis("cochran_q", "Cochran's Q test", CAT, [Role("xs", "any_multi", "Binary measurements (one column per condition)")],
          [Param("success", "Value counted as 1 (blank = 1/TRUE/yes)", "text", "")],
          code={"stata": "* Stata: cochran command (ssc install cochran)", "r": "# install.packages('RVAideMemoire'); RVAideMemoire::cochran.qtest(...)",
                "python": "from statsmodels.stats.contingency_tables import cochrans_q; cochrans_q(df[[{xs_q}]].dropna().astype(int).values)"},
          reference="Q = (k-1)[kΣC_j² - T²]/(kT - ΣR_i²) ~ χ²(k-1), subjects with all-0 or all-1 rows contribute nothing.")
def cochran_q(d, success=""):
    names = d.label_list("xs")
    cols = d["xs"]
    def conv(v):
        out = []
        for a in np.asarray(v, dtype=object).tolist():
            if a is None or (isinstance(a, float) and np.isnan(a)):
                out.append(np.nan)
            elif success:
                out.append(1.0 if str(a) == success else 0.0)
            else:
                s = str(a).strip().lower()
                out.append(1.0 if s in ("1", "1.0", "true", "yes", "y", "t", "success") else 0.0 if s in ("0", "0.0", "false", "no", "n", "f", "failure") else np.nan)
        return np.array(out)
    X = np.column_stack([conv(c) for c in cols])
    X = X[np.all(np.isfinite(X), axis=1)]
    n, k = X.shape
    need(k >= 2 and n >= 2, "need at least 2 conditions and 2 complete subjects")
    Cj, Ri, T = X.sum(0), X.sum(1), X.sum()
    den = k * T - (Ri ** 2).sum()
    need(den > 0, "all subjects respond identically in every condition: Q undefined")
    Q = (k - 1) * (k * (Cj ** 2).sum() - T ** 2) / den
    p = float(stats.chi2.sf(Q, k - 1))
    r = Result("Cochran's Q test")
    r.text("Hypotheses", "H0: the proportion of successes is the same in all k conditions")
    r.table("Successes per condition", ["condition", "successes", "proportion"], [[names[j], int(Cj[j]), Cj[j] / n] for j in range(k)])
    r.kv("Test", [("subjects", n), ("Q statistic", float(Q)), ("df", k - 1), ("p-value", p)])
    r.text("Conclusion", sig_note(p))
    r.values.update(dict(Q=float(Q), p=p, df=k - 1))
    return r


@analysis("binomial", "Exact binomial test", CAT, [Role("x", "any", "Binary variable")],
          [Param("success", "Value counted as success (blank = last level)", "text", ""), Param("p0", "Hypothesised probability p0", "float", 0.5, min=0.0001, max=0.9999), ALT, CONF],
          code={"stata": "bitest {x} == {p0}", "r": "binom.test(sum(df${x} == {success_r}), sum(!is.na(df${x})), p = {p0}, alternative = \"{alternative}\", conf.level = {conf})",
                "python": "import scipy.stats as st; st.binomtest((df['{x}']=={success_r}).sum(), df['{x}'].count(), {p0}, alternative='{alternative}')"},
          reference="Two-sided p = Σ P(X=k) over all k with P(X=k) ≤ P(X=observed) (the R / scipy convention).")
def binomial(d, success="", p0=0.5, alternative="two-sided", conf=0.95):
    s, suc = _binary(d["x"], success)
    n, k = s.size, int(s.sum())
    bt = stats.binomtest(k, n, p0, alternative=alternative)
    ci = bt.proportion_ci(conf, method="exact")
    r = Result(f"Exact binomial test: {d.name('x')} = {suc}")
    r.kv("Test", [("successes", k), ("trials", n), ("estimate", k / n), ("p0", p0), ("p-value", bt.pvalue),
                  (f"{conf:.0%} CI lower (Clopper–Pearson)", ci.low), (f"{conf:.0%} CI upper", ci.high)])
    r.text("Conclusion", sig_note(bt.pvalue))
    r.values.update(dict(k=k, n=n, p=float(bt.pvalue), lo=ci.low, hi=ci.high))
    return r


def multinomial_exact_p(obs, probs, max_tables=2_000_000, n_mc=500_000, seed=12345):
    """Exact multinomial test p-value: Σ P(table) over tables with P ≤ P(observed); Monte-Carlo fallback."""
    obs = np.asarray(obs, dtype=np.int64)
    probs = np.asarray(probs, dtype=float)
    n, k = int(obs.sum()), len(obs)
    lp = np.log(probs)
    def logp(t):
        return gammaln(n + 1) - gammaln(t + 1).sum() + (t * lp).sum()
    lobs = logp(obs)
    ntab = math.comb(n + k - 1, k - 1)
    if ntab <= max_tables:
        total = 0.0
        out = np.zeros(k, dtype=np.int64)

        def go(i, left):
            nonlocal total
            if i == k - 1:
                out[i] = left
                l = logp(out)
                if l <= lobs + 1e-9:
                    total += math.exp(l)
                return
            for v in range(left + 1):
                out[i] = v
                go(i + 1, left - v)
        go(0, n)
        return min(total, 1.0), f"exact (enumerated {ntab:,} outcomes)"
    rng = np.random.default_rng(seed)
    sims = rng.multinomial(n, probs, size=n_mc)
    ls = gammaln(n + 1) - gammaln(sims + 1).sum(1) + (sims * lp).sum(1)
    return float((np.sum(ls <= lobs + 1e-9) + 1) / (n_mc + 1)), f"Monte-Carlo ({n_mc:,} multinomial draws, seed {seed})"


@analysis("multinomial", "Exact multinomial test", CAT, [Role("x", "categorical", "Variable")],
          [Param("probs", "Hypothesised probabilities (comma-separated; blank = equal)", "text", ""), Param("seed", "Random seed (large problems only)", "int", 12345)],
          code={"stata": "* multinomial exact: use R/Python", "r": "# install.packages('EMT'); EMT::multinomial.test(as.numeric(table(df${x})), c({probs_r}))",
                "python": "# statsmith.stats.categorical.multinomial_exact_p(counts, probs)"},
          reference="Sum of multinomial probabilities of all outcomes no more likely than the observed one (Read & Cressie 1988).")
def multinomial(d, probs="", seed=12345):
    x = [str(v) for v in np.asarray(d["x"], dtype=object).tolist() if not (v is None or (isinstance(v, float) and np.isnan(v)))]
    lv = list(dict.fromkeys(x))
    obs = np.array([x.count(l) for l in lv])
    k = len(lv)
    need(k >= 2, "need at least 2 categories")
    p = np.full(k, 1 / k) if not probs.strip() else np.array([float(t) for t in probs.replace(";", ",").split(",") if t.strip()])
    need(len(p) == k, f"expected {k} probabilities for levels {', '.join(lv)}")
    p = p / p.sum()
    pv, how = multinomial_exact_p(obs, p, seed=int(seed))
    chi2, pchi = stats.chisquare(obs, obs.sum() * p)
    r = Result(f"Exact multinomial test: {d.name('x')}")
    r.table("", ["level", "observed", "hypothesised p", "expected"], [[lv[i], int(obs[i]), p[i], obs.sum() * p[i]] for i in range(k)])
    r.kv("Test", [("exact p-value", pv), ("for comparison: χ² approx. p-value", float(pchi))])
    r.note(how)
    r.text("Conclusion", sig_note(pv))
    r.values.update(dict(p=pv, p_chi2=float(pchi)))
    return r


@analysis("odds_ratio", "2×2 table: odds ratio, relative risk, risk difference", CAT, [Role("row", "categorical", "Exposure (2 levels)"), Role("col", "categorical", "Outcome (2 levels)")],
          [CONF],
          code={"stata": "cc {col} {row}   /* case-control  OR; or: cs {col} {row} for risk ratios */", "r": "# epitools::oddsratio / riskratio on table(df${row}, df${col})",
                "python": "from statsmodels.stats.contingency_tables import Table2x2; t=Table2x2(pd.crosstab(df['{row}'], df['{col}']).values); t.oddsratio, t.oddsratio_confint(), t.riskratio, t.riskratio_confint()"},
          reference="OR = ad/bc with Woolf logit CI exp(ln OR ± z√(1/a+1/b+1/c+1/d)); RR = [a/(a+b)]/[c/(c+d)] with log-method CI.")
def odds_ratio(d, conf=0.95):
    rows, cols, M = _two_by_two(d)
    a, b, c, dd = [int(v) for v in M.ravel()]
    # Haldane–Anscombe 0.5 correction only when a zero cell would make the estimate undefined
    zero = min(a, b, c, dd) == 0
    A, B, C, D = (a + .5, b + .5, c + .5, dd + .5) if zero else (a, b, c, dd)
    q = stats.norm.ppf(1 - (1 - conf) / 2)
    OR = (A * D) / (B * C)
    se = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)
    r1, r2 = A / (A + B), C / (C + D)
    RR = r1 / r2
    serr = np.sqrt(1 / A - 1 / (A + B) + 1 / C - 1 / (C + D))
    rd = a / (a + b) - c / (c + dd) if (a + b) and (c + dd) else float("nan")
    r = Result(f"2×2 table effect sizes: {d.name('row')} × {d.name('col')}")
    _tbl_text(r, "Counts", rows, cols, M, d.name("row"), d.name("col"))
    r.table("Effect sizes", ["measure", "estimate", f"{conf:.0%} lower", f"{conf:.0%} upper"],
            [["odds ratio", OR, np.exp(np.log(OR) - q * se), np.exp(np.log(OR) + q * se)],
             ["relative risk", RR, np.exp(np.log(RR) - q * serr), np.exp(np.log(RR) + q * serr)],
             ["risk difference", rd, "", ""]])
    if zero:
        r.note("A zero cell was present: the Haldane–Anscombe correction (+0.5 to every cell) was used for OR and RR.")
    r.values.update(dict(OR=OR, RR=RR, rd=rd))
    return r
