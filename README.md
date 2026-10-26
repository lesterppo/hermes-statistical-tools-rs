# hermes-statistical-tools-rs

A Rust port of [`lesterppo/hermes-statistical-tools`](https://github.com/lesterppo/hermes-statistical-tools)
— five AI-agent-native statistical tools that take CSV data in and return
compact JSON out. The original is Python (scipy / statsmodels / semopy / girth /
lifelines); this port is a single static binary with a hand-rolled numeric
core (statrs + nalgebra for distributions and linear algebra), no Python venv
required.

## Why Rust

- **One binary, ~2.4 MB, 0.016 s cold start** (vs ~5 s for the Python tools'
  scipy/pandas import on the same machine).
- **Deterministic numeric core**: distributions, exact rank tests, OLS by SVD
  pseudo-inverse, IRLS for GLMs, Breslow Cox PH, Kaplan-Meier with log-log
  Greenwood CI, varimax factor rotation — all verified against the Python
  original in [`DEEP_TEST.md`](DEEP_TEST.md) (62/64 harness cases pass; the 2
  divergences are Python-side failures, documented there).
- **Same contract where implemented**: same action names, same compact JSON
  short keys (`c` = coef, `s` = std_err, `p` = p_val, …), errors as
  `{"e": "message"}` (never a stack trace), exit code 1 on error.

## Install

```sh
cargo build --release
# binary: target/release/hst
```

Requires Rust 1.99+ (tested with 1.99.1). No other runtime dependencies.

## Usage

```sh
# PSPP-style analysis: action, then CSV (string, file, or stdin with --csv -)
hst pspp desc --csv "v
1
2
3" --v v
# {"k":{"vars":[{"N":3,"m":2.0,"max":3.0,"min":1.0,"n":"v","sd":1.0}]}}

hst pspp ttest --csv-file data.csv --a g --v v --g 1,2
hst pspp anova --csv-file data.csv --a g --v v --posthoc tukey
hst pspp reg --csv-file data.csv --o y --p x1,x2
hst pspp factor --csv-file data.csv --v i1,i2,i3,i4 --rotate varimax

# medical_ext: survival / meta-analysis / power
hst medical_ext km --csv-file surv.csv --params '{"t":"time","e":"event","g":"grp"}'
hst medical_ext cox --csv-file surv.csv --params '{"t":"time","e":"event","p":["x1","x2"]}'
hst medical_ext forest --csv-file meta.csv --params '{"p":["study"],"effect_col":"effect","se_col":"se"}'
hst medical_ext prop --params '{"p1":0.5,"p2":0.7,"nobs1":100}'   # solve power

# IRT: ability by MLE (girth-compatible) and classical test theory
hst irt score --csv-file items.csv --params '{"diff":"-1,-0.5,0,0.5,1","model":"2pl"}'
hst irt ctt --csv-file items.csv

# What is implemented
hst --list
```

PSPP output is wrapped as `{"k": {...}}` to match the Python tool;
medical_ext, IRT, statsmodels and SEM output is direct. Rounding: 4 dp
for PSPP actions, 6 dp for medical_ext and IRT, matching the Python
tools. Statsmodels/SEM use the same rounding as their Python tools
(see `scripts/verify_sm.py`).

## Parity table

Legend: ✅ Implemented · ◐ Partial · ⏳ Not yet (reason given, never faked).

### pspp (45 ✅)

| Action | Status | Notes |
|---|---|---|
| desc, freq, means, crosstab | ✅ | |
| ttest, pttest, ttest1 | ✅ | Welch t-test, zero-variance returns `warn` payload |
| anova | ✅ | Tukey / Bonferroni / Games-Howell post-hoc |
| mw, kw, wilcoxon, friedman, sign, runs | ✅ | Exact p by DP for MW / Wilcoxon (no ties); normal approx otherwise |
| ks | ✅ | D exact; one-sample p exact (Simard & L'Ecuyer 2011: DMTW/Pomeranz); two-sample p exact lattice DP incl. ties; asymptotic only for n > 2000 / n·m > 5e7 |
| corr, spearman, partial | ✅ | |
| reg, logistic, poisson | ✅ | OLS by SVD; IRLS/Newton for GLMs |
| factor | ✅ | Eigendecomposition + varimax (Python's exact M term); loadings sign is arbitrary |
| reliability, roc, eval, surv, meta, blandaltman, kappa, evalue, power, rank, hedgesg, omega | ✅ | |
| examine | ✅ | Incl. Shapiro-Wilk (Royston AS R94); n > 5000 returns null, matching Python |
| dca, quantreg, negbin, mediation, pscore, gam, zip, lasso, adf, beta | ✅ | First-principles reimplementations incl. Python's quirks (see DEEP_TEST.md: gam `df_val` shadowing, mediation ignoring `p` covariates, zip's 5-iteration linear M-steps, lasso's `lam/2` threshold) |

### medical_ext (7 ✅)

| Action | Status | Notes |
|---|---|---|
| km, cox, logrank, forest, ttest, anova, prop | ✅ | KM follows lifelines timeline (0.0 point + censor times, log-log Greenwood CI); log-rank `z` is lifelines' chi-square statistic; Cox uses Efron ties (lifelines default) + lifelines concordance rule; forest follows statsmodels ≥ 0.15 `combine_effects` (DL) with RE = FE fallback when tau² ≤ 0; prop solve-power returns the correct value (Python bug fixed, see DEEP_TEST.md) |

### irt (7 ✅)

| Action | Status | Notes |
|---|---|---|
| score | ✅ | MLE, EAP, MAP (2PL/3PL). 2PL extremes (all-zero/all-one) are null, matching girth; 3PL clips to ±6, matching the Python path |
| ctt | ✅ | Item p / point-biserial r, Cronbach alpha, score mean/SD |
| rasch, 2pl, 3pl, grm, pcm (model fits) | ✅ | Bock–Aitkin EM over girth's Gauss–Legendre-41 grid; 3PL bounded (flat (b,g) tradeoff at the g=0 bound, LL agrees to 1e-4); GRM bounded (Rust is the exact MML optimum, girth stops early — see DEEP_TEST.md) |

### statsmodels (6 ✅)

| Action | Status | Notes |
|---|---|---|
| mlm, gee, gee_nom, anova_rm | ✅ | Exact vs statsmodels 0.15.0 (incl. Gamma GEE log-likelihood) |
| gee_ord | ✅ | Identifiable quantities exact (slope, ll/aic/bic, threshold+const sums); individual threshold/const splits are unidentifiable by construction (rank-deficient design) — see DEEP_TEST.md |
| mice | ✅ | Seeded (20261026) PMM re-implementation of statsmodels' procedure; distributionally matching (Python is unseeded) — see DEEP_TEST.md |

### sem (2 ✅)

| Action | Status | Notes |
|---|---|---|
| sem_fit, sem_inspect | ✅ | Exact vs semopy 2.3.11 (MLW covariance divides by n; variances via u² so the 0.0 boundary is reachable); param tables compared canonically (semopy's ~~ direction and row order vary run to run) |

## License

MIT — see `LICENSE`. Original Python implementation:
https://github.com/lesterppo/hermes-statistical-tools
