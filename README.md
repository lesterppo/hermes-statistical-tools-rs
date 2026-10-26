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
medical_ext and IRT output is direct. Rounding: 4 dp for PSPP actions,
6 dp for medical_ext and IRT, matching the Python tools.

## Parity table

Legend: ✅ Implemented · ◐ Partial · ⏳ Not yet (reason given, never faked).

### pspp (35 ✅, 1 ◐, 10 ⏳)

| Action | Status | Notes |
|---|---|---|
| desc, freq, means, crosstab | ✅ | |
| ttest, pttest, ttest1 | ✅ | Welch t-test, zero-variance returns `warn` payload |
| anova | ✅ | Tukey / Bonferroni / Games-Howell post-hoc |
| mw, kw, wilcoxon, friedman, sign, runs | ✅ | Exact p by DP for MW / Wilcoxon (no ties); normal approx otherwise |
| ks | ✅ | D exact; p asymptotic (scipy exact differs slightly, see DEEP_TEST.md) |
| corr, spearman, partial | ✅ | |
| reg, logistic, poisson | ✅ | OLS by SVD; IRLS/Newton for GLMs |
| factor | ✅ | Eigendecomposition + varimax (Python's exact M term); loadings sign is arbitrary |
| reliability, roc, eval, surv, meta, blandaltman, kappa, evalue, power, rank, hedgesg, omega | ✅ | |
| examine | ◐ | Everything except Shapiro-Wilk (`sw`/`swp` are null) |
| dca, quantreg, negbin, mediation, pscore, gam, zip, lasso, adf, beta | ⏳ | Need statsmodels internals (quantile/negative-binomial/GAM/zero-inflated/LASSO/ADF/beta regression, decision-curve, mediation, propensity scoring) not yet reimplemented from first principles |

### medical_ext (7 ✅)

| Action | Status | Notes |
|---|---|---|
| km, cox, logrank, forest, ttest, anova, prop | ✅ | KM follows lifelines timeline (0.0 point + censor times, log-log Greenwood CI); log-rank `z` is lifelines' chi-square statistic; forest follows statsmodels ≥ 0.15 `combine_effects` (DL); prop solve-power returns the correct value where Python returns null (documented bug, see DEEP_TEST.md) |

### irt (2 ✅, 5 ⏳)

| Action | Status | Notes |
|---|---|---|
| score | ✅ | MLE only, 2PL/3PL. 2PL extremes (all-zero/all-one) are null, matching girth; 3PL clips to ±6, matching the Python path. EAP/MAP not yet implemented |
| ctt | ✅ | Item p / point-biserial r, Cronbach alpha, score mean/SD |
| rasch, 2pl, 3pl, grm, pcm (model fits) | ⏳ | Need girth's joint-MML estimation internals; not reimplemented yet |

### statsmodels (0 ✅, 6 ⏳)

| Action | Status | Notes |
|---|---|---|
| mlm, gee, gee_ord, gee_nom, anova_rm, mice | ⏳ | Need statsmodels internals (mixed models, GEE, repeated-measures ANOVA, multiple imputation); not reimplemented yet |

### sem (0 ✅, 2 ⏳)

| Action | Status | Notes |
|---|---|---|
| sem_fit, sem_inspect | ⏳ | Need semopy internals (SEM estimation, fit indices); not reimplemented yet |

## License

MIT — see `LICENSE`. Original Python implementation:
https://github.com/lesterppo/hermes-statistical-tools
