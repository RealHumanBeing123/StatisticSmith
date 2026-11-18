# Methods & verification

Everything below is generated from the code (`python scripts/gen_methods.py`). Each analysis states the formula or library routine it uses.

## How results are verified

* **Reference values** - `tests/test_stats_reference.py` reproduces published examples (R documentation datasets: sleep, mtcars; Fisher's tea-tasting;
  NIST Grubbs example) and compares to independent SciPy / statsmodels / closed-form calculations.
* **Multivariate & time series** - `tests/test_stats_multivariate.py` checks PCA, Bartlett/KMO, Hotelling T² (equals the t-test for one variable), MANOVA (Wilks' Λ from the
  SSCP definition), LDA, Cronbach's α, canonical correlation (vs statsmodels), Ljung–Box / ACF (vs statsmodels), ADF/KPSS, ARIMA, Mann–Kendall, runs, capability.
* **Charts** - `tests/test_charts_*.py` read the artists that were actually drawn (bar heights, polygon vertices, line data, tile counts, Sankey link widths,
  chord arc angles ...) and compare them with independent numpy computations; a smoke test renders every chart with normal, NaN-containing, tiny and dark-theme data.
  `tests/test_chart_coverage.py` asserts every chart in the Datylon article exists under the article's category.
* **Reproducible scripts** - `tests/test_codegen.py` executes the generated Python script and checks it gives the same numbers as the app, and that every Stata / R / Python
  template resolves with no leftover placeholders.
* **Large data** - `tests/test_engine_terminal.py` imports multi-million-row CSVs, checks file order is preserved, cancellation, memory/thread limits and all file formats.

### Known limits (please read)

* The generated **Stata and R scripts could not be executed in the build environment** (neither Stata nor R was installed). They are template-generated and were checked
  for resolved placeholders and syntax-by-construction only; the Python script *is* executed in the tests. Run the `.do` / `.R` file in Stata / R to confirm - that is its purpose.
* The Windows executable build (`statistic_smith.spec`, GitHub Actions) was prepared but **not run on Windows** by the author of this code.
* The Stata and R terminals implement a documented **subset** of commands (type `help` in the terminal), not full interpreters.
* Randomised methods (k-means, permutation tests, chart sampling) use the seed in Settings; other software can reach different (equally valid) local optima.
* Charts that draw one mark per row use a reproducible random sample of 200,000 rows for speed; all statistics use every row.


## Statistical analyses

**77 analyses and 90 chart types are registered.**

### Summary & descriptive

* **Confidence interval for a mean** (`ci_mean`) - mean ± t(1-α/2, n-1)·s/√n
* **Confidence interval for variance / std.dev.** (`ci_var`) - (n-1)s²/χ²(1-α/2, n-1) ≤ σ² ≤ (n-1)s²/χ²(α/2, n-1)
* **Frequency table** (`frequency`) - 
* **Summary statistics** (`summary`) - Sample sd and variance use n-1; skewness/kurtosis are the bias-corrected (type 2) estimators, kurtosis is excess kurtosis.
* **Z-scores (standardise)** (`zscores`) - z = (x - mean) / sd.

### Means & location (parametric)

* **F-test for equality of two variances** (`f_test`) - F = s1²/s2² ~ F(n1-1, n2-1) under normality; two-sided p = 2·min(P(F≤f), P(F≥f)).
* **One-sample proportion test (z and exact)** (`prop_one`) - z = (p̂-p0)/√(p0(1-p0)/n); exact binomial p-value and Clopper–Pearson / Wilson intervals.
* **One-sample t-test** (`t_one`) - t = (x̄ - μ0)/(s/√n), df = n-1.
* **Paired t-test** (`t_paired`) - One-sample t on the differences x - y, df = n-1.
* **Permutation test (two groups)** (`permutation`) - Monte-Carlo permutation distribution of the group difference (exact when all permutations are enumerated).
* **Two-sample proportion test** (`prop_two`) - Pooled z = (p̂1-p̂2)/√(p̂(1-p̂)(1/n1+1/n2)); CI uses the unpooled (Wald) standard error.
* **Two-sample t-test (pooled variance)** (`t_two`) - Student's t with pooled variance, df = n1+n2-2. Assumes equal variances (see F-test / Levene).
* **Welch's t-test (unequal variances)** (`t_welch`) - Welch–Satterthwaite df = (s1²/n1+s2²/n2)² / [(s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1)].
* **Z-test for one mean (σ known)** (`z_one`) - z = (x̄ - μ0)/(σ/√n) ~ N(0,1).
* **Z-test for two means (σ known)** (`z_two`) - z = (x̄1 - x̄2 - Δ0)/√(σ1²/n1 + σ2²/n2).

### Nonparametric & rank tests

* **Friedman test (repeated measures)** (`friedman`) - Q = 12/(nk(k+1)) Σ R_j² - 3n(k+1), tie-corrected, ~ χ²(k-1). Rows with missing values are dropped.
* **Kolmogorov–Smirnov test (two samples)** (`ks_two`) - D = sup|F1(x) - F2(x)|.
* **Kruskal–Wallis H test** (`kruskal`) - H = [12/(N(N+1)) Σ R_i²/n_i - 3(N+1)] / [1 - Σ(t³-t)/(N³-N)] ~ χ²(k-1).
* **Mann–Whitney U / Wilcoxon rank-sum test** (`mann_whitney`) - U1 = R1 - n1(n1+1)/2. Asymptotic z uses the tie-corrected variance n1n2/12·[(N+1) - Σ(t³-t)/(N(N-1))].
* **Mood's median test** (`median_test`) - Pearson χ² on the group × (above / not above the grand median) table.
* **Siegel–Tukey test (equal dispersion)** (`siegel_tukey`) - Siegel & Tukey (1960): Wilcoxon rank-sum applied to extreme-alternating ranks; a small rank sum for a group means larger spread.
* **Sign test** (`sign_test`) - Exact binomial on the number of positive differences, ties (zeros) dropped.
* **Wilcoxon signed-rank test** (`wilcoxon`) - W+ = sum of ranks of positive differences |d_i| (zero differences dropped by default).

### Categorical & contingency

* **2×2 table: odds ratio, relative risk, risk difference** (`odds_ratio`) - OR = ad/bc with Woolf logit CI exp(ln OR ± z√(1/a+1/b+1/c+1/d)); RR = [a/(a+b)]/[c/(c+d)] with log-method CI.
* **Barnard's exact test (2×2)** (`barnard`) - Unconditional exact test (Barnard 1945) via scipy.stats.barnard_exact: maximises p over the nuisance parameter.
* **Boschloo's exact test (2×2)** (`boschloo`) - Boschloo (1970): unconditional test using Fisher's p-value as the test statistic; uniformly more powerful than Fisher's test.
* **Cochran's Q test** (`cochran_q`) - Q = (k-1)[kΣC_j² - T²]/(kT - ΣR_i²) ~ χ²(k-1), subjects with all-0 or all-1 rows contribute nothing.
* **Exact binomial test** (`binomial`) - Two-sided p = Σ P(X=k) over all k with P(X=k) ≤ P(X=observed) (the R / scipy convention).
* **Exact multinomial test** (`multinomial`) - Sum of multinomial probabilities of all outcomes no more likely than the observed one (Read & Cressie 1988).
* **Fisher's exact test** (`fisher`) - Hypergeometric probability of the table given fixed margins; 2×2 uses scipy.stats.fisher_exact (conditional MLE odds ratio is NOT reported, the sample odds ratio is); r×c uses the Freeman–Halton extension.
* **McNemar's test (paired binary)** (`mcnemar`) - Uses discordant pairs b, c only: χ² = (|b-c|-1)²/(b+c) ~ χ²(1) (corrected); exact: 2·P(Bin(b+c, .5) ≤ min(b,c)).
* **Pearson χ² test of independence** (`chi2_indep`) - χ² = Σ (O-E)²/E, E = row·col/N, df = (r-1)(c-1). Valid when most expected counts ≥ 5 (otherwise use Fisher's exact test). Cramér's V = √(χ²/(N·min(r-1,c-1))).
* **χ² goodness-of-fit test** (`chi2_gof`) - χ² = Σ (O_i - N p_i)²/(N p_i), df = k-1.

### Normality & distribution fit

* **Anderson–Darling normality test** (`anderson_darling`) - A²* = A²(1 + 0.75/n + 2.25/n²); p-value by D'Agostino & Stephens (1986) piecewise formula (R nortest::ad.test).
* **D'Agostino–Pearson omnibus normality test** (`dagostino`) - K² = Z(skewness)² + Z(kurtosis)² ~ χ²(2) (D'Agostino 1971, D'Agostino & Pearson 1973), n ≥ 20 recommended.
* **Jarque–Bera normality test** (`jarque_bera`) - JB = n/6·(S² + K²/4) ~ χ²(2), S = skewness (biased), K = excess kurtosis (biased).
* **Kolmogorov–Smirnov test vs a fully specified distribution** (`ks_normal`) - D = sup|F_n(x) - F0(x)|. Parameters must be specified IN ADVANCE: if they are estimated from the same data the p-value is conservative — use Lilliefors' test for normality with estimated mean/sd.
* **Lilliefors normality test** (`lilliefors`) - Lilliefors (1967): KS statistic with mean and sd estimated from the data; p-value by Dallal & Wilkinson (1986) approximation (as R nortest) or by Monte-Carlo simulation of the null distribution.
* **Shapiro–Francia normality test** (`shapiro_francia`) - Shapiro & Francia (1972): squared correlation of the ordered data with Blom normal scores; p via Royston (1993) normal approximation, 5 ≤ n ≤ 5000.
* **Shapiro–Wilk normality test** (`shapiro_wilk`) - Shapiro & Wilk (1965); algorithm AS R94 (Royston 1995) as implemented in scipy. Valid for 3 ≤ n ≤ 5000.

### Variance & ANOVA

* **Bartlett's test (equal variances)** (`bartlett`) - Bartlett (1937): T = [(N-k)ln s_p² - Σ(n_i-1)ln s_i²] / [1 + (Σ1/(n_i-1) - 1/(N-k))/(3(k-1))] ~ χ²(k-1). Sensitive to non-normality.
* **Levene / Brown–Forsythe test (equal variances)** (`levene`) - W = ANOVA F on |x_ij - centre_i|. centre = median is the Brown–Forsythe variant (robust).
* **One-way ANOVA** (`anova_one`) - F = MSB/MSW, df = (k-1, N-k). η² = SSB/SST; ω² = (SSB-(k-1)MSW)/(SST+MSW).
* **Repeated-measures ANOVA (one within factor)** (`anova_rm`) - SS_cond / MS_error with error = subjects × conditions; Greenhouse–Geisser ε reported for the sphericity violation.
* **Tukey HSD pairwise comparisons** (`tukey_hsd`) - Studentised-range based simultaneous intervals (Tukey–Kramer for unequal n) via scipy.stats.tukey_hsd.
* **Two-way ANOVA (with interaction)** (`anova_two`) - statsmodels OLS + anova_lm. Type II/III results coincide with Stata's `anova` / R car::Anova for balanced designs; type III uses sum-to-zero (effect) coding.
* **Welch's ANOVA (unequal variances)** (`anova_welch`) - Welch (1951): F* = [Σw_i(x̄_i - x̄_w)²/(k-1)] / [1 + 2(k-2)/(k²-1)·Σ(1-w_i/W)²/(n_i-1)], df2 = (k²-1)/(3Σ(1-w_i/W)²/(n_i-1)).

### Correlation & association

* **Correlation matrix (r, p-values)** (`correlation`) - Pearson r with t = r√(n-2)/√(1-r²), df = n-2; Spearman via scipy.stats.spearmanr; Kendall τ-b via scipy.stats.kendalltau.
* **Correlation test (one pair, with CI)** (`corr_test`) - Pearson: Fisher z-transform CI, z' = atanh(r), se = 1/√(n-3).
* **Covariance matrix** (`covariance`) - 

### Regression & modelling

* **Chow test for a structural break** (`chow`) - F = [(RSS_p - (RSS_1+RSS_2))/k] / [(RSS_1+RSS_2)/(n-2k)] ~ F(k, n-2k), k = parameters incl. intercept (Chow 1960).
* **Heteroskedasticity tests (Breusch–Pagan, White)** (`heterosk`) - Breusch–Pagan (studentised, Koenker 1981) and White (1980) LM tests on OLS residuals.
* **Linear regression (simple / multiple)** (`ols`) - OLS via statsmodels (QR/pinv). t = b/se(b), CI = b ± t(1-α/2, n-k-1)·se. HC1/HC3: MacKinnon–White heteroskedasticity-robust SEs. VIF_j = 1/(1-R_j²). Durbin–Watson, Breusch–Pagan (studentised) and Jarque–Bera on residuals.
* **Logistic regression (binary outcome)** (`logit`) - Maximum likelihood (IRLS/Newton) via statsmodels; Wald z-tests; OR = exp(b). McFadden pseudo-R² = 1 - LL/LL0; LR χ² test of the whole model.
* **Poisson regression (count outcome)** (`poisson`) - Poisson GLM with log link (ML). IRR = exp(b). Overdispersion check: Pearson χ²/df.
* **Polynomial regression** (`poly_reg`) - Raw (non-orthogonal) polynomial terms x, x², …, x^p fitted by OLS (centred internally for numerical stability).

### Multivariate

* **Bartlett's test of sphericity & KMO** (`bartlett_sphericity`) - χ² = -(n - 1 - (2p+5)/6)·ln|R|, df = p(p-1)/2 (Bartlett 1951). KMO = Σr² / (Σr² + Σa²), a = partial correlations (from R⁻¹).
* **Canonical correlation analysis** (`cancorr`) - Canonical correlations via SVD of Qx'Qy (QR of centred sets); Wilks' Λ with Bartlett's χ² approximation tests all correlations ≥ k (Bartlett 1941).
* **Cronbach's alpha (scale reliability)** (`cronbach`) - α = k/(k-1)·(1 − Σσ²_i / σ²_total) over listwise-complete rows; 'alpha if item deleted' recomputes with each item removed.
* **Hierarchical clustering (agglomerative)** (`hclust`) - scipy.cluster.hierarchy.linkage on Euclidean distances (Ward minimum-variance by default); tree cut with 'maxclust'. Limited to 5,000 rows (O(n²) memory) - use k-means for more.
* **Hotelling's T² (one- or two-sample)** (`hotelling`) - One-sample T² = n(x̄-μ)'S⁻¹(x̄-μ), F = (n-p)/(p(n-1))·T² ~ F(p, n-p). Two-sample uses the pooled covariance: T² = n1n2/(n1+n2)·(x̄1-x̄2)'S_p⁻¹(x̄1-x̄2), F = (n1+n2-p-1)/(p(n1+n2-2))·T² ~ F(p, n1+n2-p-1) (Hotelling 1931).
* **Linear discriminant analysis** (`lda`) - Fisher/Gaussian LDA with pooled within-group covariance and empirical priors. Discriminant functions come from the eigenvectors of S_w⁻¹S_b; classification uses the linear score δ_k(x) = x'S⁻¹μ_k − ½μ_k'S⁻¹μ_k + ln π_k. Accuracy is resubstitution (optimistic).
* **MANOVA (one-way)** (`manova`) - Wilks' Λ = |E|/|E+H|, Pillai's trace, Hotelling–Lawley trace and Roy's largest root with the usual F approximations (statsmodels.multivariate.manova). E, H = within / between SSCP matrices.
* **Multivariate multiple regression (several outcomes)** (`mvreg`) - Equation-by-equation OLS (identical coefficients to multivariate least squares B = (X'X)⁻¹X'Y); overall test of each predictor via Wilks' Λ with Rao's F approximation, as in Stata `mvreg` / `mvtest`.
* **Multivariate outliers (Mahalanobis distance)** (`mahalanobis`) - D² = (x−x̄)'S⁻¹(x−x̄) ~ χ²(p) approximately under multivariate normality; rows with p < α are flagged.
* **Principal component analysis (PCA)** (`pca`) - Eigen-decomposition of the correlation (default) or covariance matrix (ddof = 1). Loadings = eigenvectors scaled by √eigenvalue for the correlation basis (as Stata `pca` reports eigenvectors; sign of each component is arbitrary).
* **k-means clustering** (`kmeans`) - Lloyd's algorithm with k-means++ seeding (scipy.cluster.vq.kmeans2), best of n restarts by within-cluster sum of squares. Cluster labels are arbitrary and results depend on the seed; R/Stata/sklearn can find a different local optimum.

### Time series & quality

* **ARIMA model (fit + forecast)** (`arima`) - Maximum-likelihood ARIMA(p,d,q) with constant when d = 0 (statsmodels state-space, same likelihood as R arima / Stata arima; small numerical differences in coefficients are normal).
* **Augmented Dickey–Fuller unit-root test** (`adf`) - statsmodels adfuller with MacKinnon (1994/2010) p-values. H0: unit root (non-stationary). 'c' = constant, 'ct' = constant + trend, 'n' = none.
* **Autocorrelation (ACF) & partial autocorrelation (PACF)** (`acf_pacf`) - ACF r_k = Σ(x_t−x̄)(x_{t+k}−x̄)/Σ(x_t−x̄)² (same as R acf). PACF by Yule–Walker (statsmodels, as R pacf). Bands ±1.96/√n.
* **Grubbs' test for a single outlier** (`grubbs`) - G = max|x_i − x̄|/s. t² = nG²... exact p via t = G·√[n(n−2)/((n−1)² − nG²)] ~ t(n−2), p = min(1, 2n·P(T > t)) (Grubbs 1969).
* **Individuals (I-MR) control chart limits** (`control_limits`) - σ̂ = MR̄/d₂ (d₂ = 1.128); UCL/LCL = x̄ ± 3σ̂; MR chart UCL = 3.267·MR̄. Points flagged: beyond 3σ (Nelson rule 1) and 8 in a row on one side (rule 2).
* **KPSS stationarity test** (`kpss`) - Kwiatkowski–Phillips–Schmidt–Shin (1992); H0: series is level ('c') / trend ('ct') stationary. p-value interpolated from published tables (bounded to 0.01–0.10).
* **Ljung–Box / Box–Pierce portmanteau test** (`ljung_box`) - Q_LB = n(n+2) Σ_{k=1}^{h} r_k²/(n−k) ~ χ²(h − fitdf); Box–Pierce Q = n Σ r_k². H0: no autocorrelation up to lag h.
* **Mann–Kendall trend test (+ Sen's slope)** (`mann_kendall`) - S = Σ_{i<j} sign(x_j − x_i); Var(S) = [n(n−1)(2n+5) − Σt(t−1)(2t+5)]/18; Z = (S ∓ 1)/√Var (continuity corrected). Sen's slope = median of all pairwise slopes (scipy.stats.theilslopes). Series > 5,000 values are summarised on an evenly spaced subsample.
* **Process capability (Cp, Cpk, Pp, Ppk)** (`capability`) - Overall (Pp/Ppk) uses the sample SD; within (Cp/Cpk) uses σ̂ = MR̄/1.128 (moving range of 2). Cp = (USL−LSL)/6σ, Cpk = min(USL−μ, μ−LSL)/3σ.
* **Wald–Wolfowitz runs test (randomness)** (`runs_test`) - Runs of values above/below the cut (values equal to the cut dropped). μ = 2n₁n₂/n + 1, σ² = (μ−1)(μ−2)/(n−1), z = (R−μ)/σ (Wald & Wolfowitz 1940).

## Charts

### Comparison (23)

Bar chart (horizontal), Column chart (vertical), Grouped column chart, Grouped bar chart, Lollipop chart, Bullet graph, Dot plot (Cleveland), Dumbbell chart, Range plot, Pictogram, Proportional icon chart, Radial bar chart, Parallel coordinates, Radar / spider chart, Nightingale rose (coxcomb), Waterfall chart, Matrix chart, Small multiples, Word cloud, Slope chart, Table chart, Categorical scatter plot, Quadrant chart

### Correlation (6)

Heatmap, Bubble chart, Scatter plot, Connected scatter plot, Hexbin plot, Contour plot

### Part-to-whole & hierarchical (21)

Stacked bar chart, Stacked column chart, Diverging bar chart, Diverging column chart, Population pyramid, Waffle chart, Icon array, Pie chart, Donut chart, Semi-circle donut chart, Marimekko chart, Treemap, Circular treemap, Convex treemap, Sunburst chart, Dendrogram, Venn diagram, Euler diagram, Circular gauge, Funnel chart, Pyramid chart

### Data over time (12)

Line chart, Spline chart, Step line chart, Area chart, Stacked area chart, Stream graph, Bump chart, Bump area chart, Candlestick chart, OHLC chart, Gantt chart, Barcode chart

### Distribution (11)

Histogram, Density plot, Ridgeline plot, Horizon chart, Radial histogram, Strip plot, Jitter plot, Beeswarm plot, 1-D heatmap, Box plot, Violin plot

### Geospatial & other (8)

Choropleth map, Geographic heat map, Tile grid map (US states), Sankey diagram, Chord diagram, Arc diagram, Network diagram, Flowchart

### Statistical (9)

Pareto chart, Q-Q plot, ECDF, Scatter plot matrix, Error bar chart, Regression plot, Residual diagnostics, Control chart (individuals), Bland-Altman plot
