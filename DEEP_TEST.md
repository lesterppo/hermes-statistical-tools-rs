# Deep test: Rust `hst` vs Python original

Run 2026-10-26 on this VM. Python side: a scratch copy of
`lesterppo/hermes-statistical-tools` at `~/workspace/hst-py-orig` with its own
venv (numpy, scipy 1.18.1, pandas, statsmodels 0.15.0, lifelines, girth) — the
original tree `~/workspace/.rs-orig-stat` was never modified or run from.
Rust side: `target/release/hst` built from this repo.

Harness: `scripts/deep_test.py` (run with the scratch venv's python). It feeds
the same deterministic synthetic datasets (seed 42: two-group, paired,
one-sample, 3-group ANOVA, 2×2 and 3×3 crosstabs, correlation, regression /
logistic / Poisson, factor items, ROC/eval, kappa, meta/forest, survival, Cox,
IRT items, Friedman, plus 5 error cases) through both implementations and
compares parsed JSON with tolerance |a−b| ≤ 5e-4 **or** relative ≤ 5e-3.
Error cases are compared by presence of `"e"` only.

## Verdict: 62/64 PASS

Unit tests: `cargo test --release` — **12/12 pass**.
Release binary: `target/release/hst`, **2,514,360 bytes (~2.4 MB)**.
Cold start (trivial descriptives call): **Rust 0.016 s vs Python 5.0 s**
(Python time is dominated by importing numpy/scipy/pandas).

## Sampled exact matches (values actually run)

| Case | Python | Rust |
|---|---|---|
| ttest (Welch) | t = −28.7795, df = 21.2566, p = 0.0, m1 = 10.0017, m2 = 19.9767, d = 11.7492 | identical |
| anova | F = 211.9802, df1 = 2, df2 = 27, p = 0.0, eta2 = 0.9401 | identical |
| reg (OLS) | R² = 0.916, F = 147.2278, intercept B = 2.7601, x1 B = 1.7878, x2 B = −1.5148 | identical |

(Key order in JSON objects differs between implementations; comparison is on
parsed values, not text.)

## Full case table

PASS (62): pspp desc, freq, examine (minus sw/swp, see below), means,
crosstab2, crosstab3, ttest, pttest, ttest1, anova, anova_bonf, anova_gh, mw,
kw, wilcoxon, friedman, sign, ks2, runs, corr, spearman, partial, reg,
logistic, poisson, factor, factor_varimax, reliability, roc, eval, surv, meta,
blandaltman, kappa, evalue_or, evalue_2x2, power_ttest_n, power_ttest_pw,
power_prop_n, power_anova_pw, rank, hedgesg, omega, med km, km_grouped,
logrank, cox, forest, power_ttest_n, power_ttest_pw, power_ttest_d,
power_anova_n, power_anova_pw, power_prop_n, irt ctt, score_2pl, score_3pl,
and all 5 error cases (empty CSV, missing column, non-numeric cell, too-small
n, unknown action).

DIVERGENCE (2) — both are Python-side failures, not Rust bugs:

1. **pspp/ks1 (one-sample KS).** Python returns
   `{"e": "Analysis error: ndtr() takes from 1 to 2 positional arguments but 3 were given"}`:
   scipy 1.18.1's `kstest` with a distribution name + `args` is broken in this
   environment (verified standalone: the name+args call fails, the callable-CDF
   call works). Rust returns D and p. Against the callable-CDF scipy path on
   the same data, Rust D = 0.0872 matches scipy exactly; Rust p = 0.9702 vs
   scipy exact p = 0.9615 — the Rust p is the asymptotic Kolmogorov series,
   scipy's is exact. Known approximation, documented here, not a pass.
2. **med/power_prop_pw (solve power for a proportion test).** Python returns
   `power: null`; Rust returns 0.828919. The Python `_power_prop` returns
   `_f(power)` (the *input*, which is `None` when solving) in both branches
   instead of the solved value — a bug in the original. Rust's 0.828919 is the
   correct value (effect size h = 0.411517, n = 100, p1 = 0.5, p2 = 0.7).
   Documented here, not a pass.

## Known approximations / partial coverage (not counted as failures)

- **Shapiro-Wilk** is not implemented in Rust: PSPP `examine` returns
  `"sw": null, "swp": null` (the harness skips those keys). All other examine
  statistics match.
- **KS p-values** use the asymptotic Kolmogorov series (see ks1 above);
  two-sample KS matched within tolerance on the harness dataset.
- **Factor loadings** are compared up to per-component sign: eigenvector sign
  is arbitrary (numpy vs nalgebra choose different signs). Eigenvalues and
  var_pct match exactly; varimax rotation is verified by the `factor_varimax`
  case (Python's varimax M term divides the column mean by n_vars a second
  time — replicated exactly).
- **Forest tau² < 0 edge case**: Rust follows the statsmodels ≥ 0.15
  `combine_effects` path (DL tau² kept raw when positive; the weight rule
  matches that path). The harness forest case (tau² > 0) matches exactly.
- **Cox gradient**: algebraically equivalent loop, verified by the passing
  `med/cox` case (coefficients, HRs, SEs, c-index, log-likelihood match).

## Reproduce

```sh
cargo test --release
cargo build --release
~/workspace/hst-py-orig/.venv/bin/python scripts/deep_test.py
```
