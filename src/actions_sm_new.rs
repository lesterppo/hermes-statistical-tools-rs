// NEW statsmodels actions for the Rust port: anova_rm, gee, gee_ord,
// gee_nom, mice, mlm. Faithful re-implementation of
// ~/workspace/.rs-orig-stat/tools/statsmodels_tool.py (statsmodels 0.15)
// from first principles: same inputs, same short output keys, same
// rounding (round to 6 decimals, tiny values kept unrounded).
//
// MERGE NOTE (for the parent): add to src/lib.rs:
//     pub mod actions_sm_new;
// and in src/main.rs, in the "statsmodels" match arm, replace the
// Not-yet error with (SmArgs built from the CLI/params, see below):
//     "statsmodels" => {
//         let a = hermes_statistical_tools_rs::actions_sm_new::SmArgs {
//             o: cli.o.clone().unwrap_or_default(),
//             p: split(&cli.p),
//             g: cli.a.clone().unwrap_or_default(),
//             fam: params.get("fam").and_then(|v| v.as_str()).unwrap_or("gaussian").to_string(),
//             cov: params.get("cov").and_then(|v| v.as_str()).unwrap_or("exchangeable").to_string(),
//             w: split(&cli.w),
//             n_imp: cli.n.unwrap_or(5),
//             model: params.get("mod").and_then(|v| v.as_str()).unwrap_or("ols").to_string(),
//         };
//         // NOTE: --w is a single string in the current CLI; for anova_rm
//         // with several within factors pass them comma-separated in --w,
//         // or set params {"w": ["A","B"]} and prefer that (see SmArgs::from_cli
//         // helper below for the exact precedence).
//         let tab = match parse_csv(csv_text) {
//             Ok(t) => t, Err(e) => return err(&e),
//         };
//         match hermes_statistical_tools_rs::actions_sm_new::dispatch_sm(&action, &tab, &a) {
//             Ok(v) => v, Err(e) => err(&e),
//         }
//     }
// The CLI also needs --fam/--cov/--mod flags eventually; until then the
// params JSON carries fam/cov/mod (the Python tool's own argument names).
//
// Behavioural quirks replicated from the Python tool (see DEEP_TEST notes):
//  * gee_ord output never contains "th": the tool checks
//    hasattr(result, "transform_threshold_params"), which the result
//    object does not have, so the threshold block is dead code.
//  * gee_ord/gee_nom keep the intercept column the tool adds, so the
//    ordinal design is rank-deficient (const = sum of threshold
//    dummies). statsmodels resolves it with pinv/min-norm steps; this
//    port does the same (GLM IRLS via SVD lstsq + pinv Fisher steps),
//    so the identifiable combinations (slope, threshold+const) match,
//    and on the verified dataset the individual values match too.
//  * gee_nom never reports ll/aic/bic: NominalGEE raises
//    NotImplementedError for llf, which the tool's safe getter swallows.
//  * mlm reports aic/bic as JSON null: MixedLM's aic/bic are NaN, which
//    the tool's formatter turns into None. rv is {} because cov_re is
//    a DataFrame in the tool, not an ndarray.
//  * anova_rm reports no sphericity corrections: the tool's formatter
//    only extracts F/Num DF/Den DF/Pr > F from AnovaRM's table.
//  * mice in the Python tool is stochastic and unseeded (PMM with
//    k=20, gaussian parameter perturbation, 10 burn-in cycles). This
//    port re-implements that procedure with a FIXED seed, so it is a
//    reproducible draw from the same distribution rather than a
//    value-for-value match with any single Python run (see report).
//  * unstructured GEE covariance errors, as in Python (it requires a
//    time argument the tool never passes).
use crate::{f_sf, norm_sf, Cell, Table};
use nalgebra::{DMatrix, DVector};
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------- helpers

/// Python _fmt_float: round to 6, NaN/inf -> null, |v| in (0,1e-8) kept raw.
pub fn f6(v: f64) -> Value {
    if !v.is_finite() {
        return Value::Null;
    }
    if v != 0.0 && v.abs() < 1e-8 {
        return json!(v);
    }
    let m = 1e6f64;
    let r = (v * m).round() / m;
    json!(if r == 0.0 { 0.0 } else { r })
}
pub fn f6o(v: Option<f64>) -> Value {
    match v {
        Some(x) => f6(x),
        None => Value::Null,
    }
}

/// One fixed-effect row: c/s/z/p/l/u, always present (null when undefined),
/// matching the Python tool's _fmt_result (bse/tvalues/pvalues/conf_int
/// are Series over all params, so every key is emitted).
fn coef_row(c: f64, s: Option<f64>) -> Value {
    let mut m = Map::new();
    m.insert("c".to_string(), f6(c));
    match s {
        Some(se) if se.is_finite() && se > 0.0 => {
            let z = c / se;
            let p = 2.0 * norm_sf(z.abs());
            m.insert("s".to_string(), f6(se));
            m.insert("z".to_string(), f6(z));
            m.insert("p".to_string(), f6(p));
            m.insert("l".to_string(), f6(c - 1.959963984540054 * se));
            m.insert("u".to_string(), f6(c + 1.959963984540054 * se));
        }
        Some(se) if se.is_finite() => {
            // se == 0: z/p are inf/nan in Python -> null; CI collapses to c.
            m.insert("s".to_string(), f6(se));
            m.insert("z".to_string(), Value::Null);
            m.insert("p".to_string(), Value::Null);
            m.insert("l".to_string(), f6(c));
            m.insert("u".to_string(), f6(c));
        }
        _ => {
            m.insert("s".to_string(), Value::Null);
            m.insert("z".to_string(), Value::Null);
            m.insert("p".to_string(), Value::Null);
            m.insert("l".to_string(), Value::Null);
            m.insert("u".to_string(), Value::Null);
        }
    }
    Value::Object(m)
}

/// SVD pseudo-inverse (numpy-pinv style cutoff).
fn pinv(a: &DMatrix<f64>) -> DMatrix<f64> {
    let svd = a.clone().svd(true, true);
    let (m, n) = (a.nrows(), a.ncols());
    let smax = svd.singular_values.iter().cloned().fold(0.0f64, f64::max);
    let cutoff = if smax > 0.0 { smax * 1e-12 * (m.max(n) as f64) } else { 0.0 };
    let u = match svd.u {
        Some(u) => u,
        None => return DMatrix::<f64>::zeros(n, m),
    };
    let vt = match svd.v_t {
        Some(vt) => vt,
        None => return DMatrix::<f64>::zeros(n, m),
    };
    let mut s_inv = DMatrix::<f64>::zeros(n, m);
    for i in 0..svd.singular_values.len().min(m).min(n) {
        if svd.singular_values[i] > cutoff {
            s_inv[(i, i)] = 1.0 / svd.singular_values[i];
        }
    }
    vt.transpose() * s_inv * u.transpose()
}

/// Solve A x = b, falling back to the pseudo-inverse (statsmodels does
/// exactly this in GEE's _update_mean_params and _covmat).
fn solve_or_pinv(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    if let Some(lu) = a.clone().lu().solve(b) {
        if lu.iter().all(|v| v.is_finite()) {
            // Accept only if it actually solves the system: LU on a
            // (near-)singular matrix returns finite garbage.
            let resid = a * &lu - b;
            let scale = b.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1.0);
            if resid.iter().map(|v| v.abs()).fold(0.0f64, f64::max) <= 1e-8 * scale {
                return Some(lu);
            }
        }
    }
    let x = pinv(a) * b;
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}
fn solve_mat_or_pinv(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if let Some(lu) = a.clone().lu().solve(b) {
        if lu.iter().all(|v| v.is_finite()) {
            let resid = a * &lu - b;
            let scale = b.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1.0);
            if resid.iter().map(|v| v.abs()).fold(0.0f64, f64::max) <= 1e-8 * scale {
                return Some(lu);
            }
        }
    }
    let x = pinv(a) * b;
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}
/// Inverse of a (near-)SPD matrix: Cholesky, then LU, then pinv.
fn inv_spd(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if let Some(ch) = a.clone().cholesky() {
        return Some(ch.inverse());
    }
    a.clone().try_inverse().or_else(|| Some(pinv(a)))
}
fn logdet_spd(a: &DMatrix<f64>) -> Option<f64> {
    if let Some(ch) = a.clone().cholesky() {
        let d = ch.l();
        let mut s = 0.0;
        for i in 0..d.nrows() {
            if d[(i, i)] <= 0.0 {
                return None;
            }
            s += d[(i, i)].ln();
        }
        return Some(2.0 * s);
    }
    // LU fallback: log|det|.
    let lu = a.clone().lu();
    let det = lu.determinant();
    if det == 0.0 || !det.is_finite() {
        None
    } else {
        Some(det.abs().ln())
    }
}
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}
fn sd_ddof1(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}
fn ln_gamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}

/// OLS via SVD (min-norm) returning beta and covariance
/// sigma^2 * pinv(X'X); used by MICE and by the MLM/ANOVA helpers.
fn ols_cov(x: &[Vec<f64>], y: &[f64]) -> Option<(Vec<f64>, DMatrix<f64>, f64)> {
    let n = y.len();
    let k = if x.is_empty() { 0 } else { x[0].len() };
    if n == 0 || k == 0 || n < k {
        return None;
    }
    // Guard: nalgebra's SVD iterates forever on NaN/Inf input.
    if !y.iter().all(|v| v.is_finite()) || !x.iter().flatten().all(|v| v.is_finite()) {
        return None;
    }
    let mut mat = DMatrix::<f64>::zeros(n, k);
    for (i, row) in x.iter().enumerate() {
        for j in 0..k {
            mat[(i, j)] = row[j];
        }
    }
    let yv = DVector::<f64>::from_column_slice(y);
    let svd = mat.clone().svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let s = &svd.singular_values;
    let smax = s.iter().cloned().fold(0.0f64, f64::max);
    let tol = smax * 1e-12 * (n.max(k) as f64);
    let uty = u.transpose() * &yv;
    let mut tmp = DVector::<f64>::zeros(k);
    for i in 0..k {
        if s[i] > tol {
            tmp[i] = uty[i] / s[i];
        }
    }
    let beta = vt.transpose() * tmp;
    let resid = &yv - &mat * &beta;
    let ss: f64 = resid.iter().map(|e| e * e).sum();
    // statsmodels OLS scale divides by df_resid = n - rank (not n - k);
    // identical for full-rank designs, correct for rank-deficient ones.
    let rank = s.iter().filter(|&&sv| sv > tol).count();
    let dof = (n - rank) as f64;
    let sigma2 = if dof > 0.0 { ss / dof } else { 0.0 };
    // cov = sigma^2 * V diag(1/s^2) V'
    let v = vt.transpose();
    let mut cov = DMatrix::<f64>::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for a in 0..k {
                if s[a] > tol {
                    sum += v[(i, a)] * v[(j, a)] / (s[a] * s[a]);
                }
            }
            cov[(i, j)] = sigma2 * sum;
        }
    }
    Some((beta.iter().cloned().collect(), cov, ss))
}

// ------------------------------------------------------------ data prep

/// Cleaned design: const first, then expanded exog columns, in the
/// Python tool's _clean_arrays order. Text exog columns become dummies
/// (drop_first, categories sorted, names "col_level"); text endog/group
/// columns become sorted category codes. Rows with any missing value
/// among the used columns are dropped.
pub struct Clean {
    pub names: Vec<String>,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub groups: Vec<f64>,
    pub n: usize,
}

fn col_cells(tab: &Table, name: &str) -> Result<Vec<Cell>, String> {
    tab.col(name)
        .ok_or_else(|| format!("column '{}' not in data", name))
}
fn is_num_col(cells: &[Cell]) -> bool {
    cells.iter().all(|c| matches!(c, Cell::Num(_) | Cell::Miss))
}
fn sorted_cats(cells: &[Cell]) -> Vec<String> {
    let mut cats: Vec<String> = Vec::new();
    for c in cells {
        if let Cell::Txt(s) = c {
            if !cats.contains(s) {
                cats.push(s.clone());
            }
        }
    }
    cats.sort();
    cats
}
fn cell_code(cell: &Cell, cats: &[String]) -> Option<f64> {
    match cell {
        Cell::Num(x) => Some(*x),
        Cell::Txt(s) => cats.iter().position(|c| c == s).map(|i| i as f64),
        Cell::Miss => None,
    }
}

pub fn clean_arrays(
    tab: &Table,
    endog: &str,
    exog: &[String],
    grp: Option<&str>,
) -> Result<Clean, String> {
    if tab.col_idx(endog).is_none() {
        return Err(format!("endog column '{}' not in data", endog));
    }
    if let Some(g) = grp {
        if tab.col_idx(g).is_none() {
            return Err(format!("group column '{}' not in data", g));
        }
    }
    for c in exog {
        if tab.col_idx(c).is_none() {
            return Err(format!("exog column '{}' not in data", c));
        }
    }
    let endog_cells = col_cells(tab, endog)?;
    let endog_cats = if is_num_col(&endog_cells) {
        Vec::new()
    } else {
        sorted_cats(&endog_cells)
    };
    let grp_cells = grp.map(|g| col_cells(tab, g)).transpose()?;
    let grp_cats = match &grp_cells {
        Some(cells) if !is_num_col(cells) => sorted_cats(cells),
        _ => Vec::new(),
    };
    // Expand exog columns (dummy coding, drop_first).
    struct ExpCol {
        name: String,
        vals: Vec<Option<f64>>,
    }
    let mut expanded: Vec<ExpCol> = Vec::new();
    for c in exog {
        let cells = col_cells(tab, c)?;
        if is_num_col(&cells) {
            let vals = cells
                .iter()
                .map(|cell| match cell {
                    Cell::Num(x) if x.is_finite() => Some(*x),
                    Cell::Num(_) => None,
                    Cell::Miss => None,
                    _ => None,
                })
                .collect();
            expanded.push(ExpCol { name: c.clone(), vals });
        } else {
            let cats = sorted_cats(&cells);
            // drop_first: skip cats[0]
            for cat in cats.iter().skip(1) {
                let vals = cells
                    .iter()
                    .map(|cell| match cell {
                        Cell::Txt(s) => Some(if s == cat { 1.0 } else { 0.0 }),
                        Cell::Miss => None,
                        Cell::Num(_) => None,
                    })
                    .collect();
                expanded.push(ExpCol {
                    name: format!("{}_{}", c, cat),
                    vals,
                });
            }
            if cats.is_empty() {
                // Column entirely missing: it still counts as a column
                // with all-missing values -> every row dropped below.
                expanded.push(ExpCol {
                    name: c.clone(),
                    vals: vec![None; cells.len()],
                });
            }
        }
    }
    let nrows = endog_cells.len();
    let mut names = vec!["const".to_string()];
    names.extend(expanded.iter().map(|e| e.name.clone()));
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut groups = Vec::new();
    for i in 0..nrows {
        let yv = match &endog_cells[i] {
            Cell::Num(v) if v.is_finite() => Some(*v),
            Cell::Num(_) => None,
            Cell::Txt(_) => cell_code(&endog_cells[i], &endog_cats),
            Cell::Miss => None,
        };
        let gv = match (&grp_cells, grp) {
            (Some(cells), Some(_)) => match &cells[i] {
                Cell::Num(v) if v.is_finite() => Some(*v),
                Cell::Num(_) => None,
                Cell::Txt(_) => cell_code(&cells[i], &grp_cats),
                Cell::Miss => None,
            },
            _ => Some(0.0),
        };
        if yv.is_none() || gv.is_none() {
            continue;
        }
        let mut row = vec![1.0];
        let mut ok = true;
        for e in &expanded {
            match e.vals[i] {
                Some(v) => row.push(v),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        x.push(row);
        y.push(yv.unwrap());
        groups.push(gv.unwrap());
    }
    if y.is_empty() {
        return Err("no complete cases after dropping missing values".to_string());
    }
    let n = y.len();
    Ok(Clean { names, x, y, groups, n })
}

// ------------------------------------------------------------ GEE engine

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fam {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
}
impl Fam {
    fn from_str(s: &str) -> Option<Fam> {
        match s {
            "gaussian" => Some(Fam::Gaussian),
            "binomial" => Some(Fam::Binomial),
            "poisson" => Some(Fam::Poisson),
            "gamma" => Some(Fam::Gamma),
            _ => None,
        }
    }
    fn mu(&self, eta: f64) -> f64 {
        match self {
            Fam::Gaussian => eta,
            Fam::Binomial => {
                if eta >= 0.0 {
                    let e = (-eta).exp();
                    1.0 / (1.0 + e)
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }
            Fam::Poisson => eta.exp().min(1e300),
            Fam::Gamma => {
                if eta == 0.0 {
                    1e300
                } else {
                    1.0 / eta
                }
            }
        }
    }
    fn dmu_deta(&self, mu: f64) -> f64 {
        match self {
            Fam::Gaussian => 1.0,
            Fam::Binomial => (mu * (1.0 - mu)).max(0.0),
            Fam::Poisson => mu,
            Fam::Gamma => -mu * mu,
        }
    }
    fn var(&self, mu: f64) -> f64 {
        match self {
            Fam::Gaussian => 1.0,
            Fam::Binomial => (mu * (1.0 - mu)).max(1e-12),
            Fam::Poisson => mu.max(1e-12),
            Fam::Gamma => mu * mu,
        }
    }
    /// link(mu): identity / logit / log / inverse.
    fn link(&self, mu: f64) -> f64 {
        match self {
            Fam::Gaussian => mu,
            Fam::Binomial => {
                let m = mu.clamp(1e-12, 1.0 - 1e-12);
                (m / (1.0 - m)).ln()
            }
            Fam::Poisson => mu.max(1e-12).ln(),
            Fam::Gamma => 1.0 / mu.max(1e-12),
        }
    }
    /// link derivative d eta / d mu.
    fn link_deriv(&self, mu: f64) -> f64 {
        match self {
            Fam::Gaussian => 1.0,
            Fam::Binomial => 1.0 / (mu * (1.0 - mu)).max(1e-12),
            Fam::Poisson => 1.0 / mu.max(1e-12),
            Fam::Gamma => -1.0 / (mu * mu).max(1e-24),
        }
    }
    fn starting_mu(&self, y: &[f64]) -> Vec<f64> {
        let m = mean(y);
        match self {
            Fam::Binomial => y.iter().map(|v| (v + 0.5) / 2.0).collect(),
            _ => y.iter().map(|v| (v + m) / 2.0).collect(),
        }
    }
    /// IRLS weight (dmu/deta)^2 / var, as statsmodels family.weights.
    fn irls_weight(&self, mu: f64) -> f64 {
        let d = self.dmu_deta(mu);
        d * d / self.var(mu)
    }
    fn loglike(&self, y: &[f64], mu: &[f64], scale: f64) -> f64 {
        let mut ll = 0.0;
        match self {
            Fam::Gaussian => {
                // Concentrated form is handled by the caller via scale_c;
                // here scale is the profiled scale (see gee_ll).
                for (yi, mi) in y.iter().zip(mu.iter()) {
                    ll += -0.5 * ((2.0 * std::f64::consts::PI * scale).ln()
                        + (yi - mi).powi(2) / scale);
                }
            }
            Fam::Binomial => {
                for (yi, mi) in y.iter().zip(mu.iter()) {
                    let m = mi.clamp(1e-12, 1.0 - 1e-12);
                    ll += yi * m.ln() + (1.0 - yi) * (1.0 - m).ln();
                }
            }
            Fam::Poisson => {
                for (yi, mi) in y.iter().zip(mu.iter()) {
                    let m = mi.max(1e-300);
                    ll += yi * m.ln() - m - ln_gamma(yi + 1.0);
                }
            }
            Fam::Gamma => {
                // statsmodels Gamma loglike_obs, var_weights = 1.
                let sc = if scale > 0.0 { scale } else { 1.0 };
                for (yi, mi) in y.iter().zip(mu.iter()) {
                    let m = mi.max(1e-12);
                    let yy = yi.max(1e-12);
                    let ws = 1.0 / sc;
                    ll += ws * (ws * (yy / m)).ln() - ws * (yy / m)
                        - ln_gamma(ws)
                        - yy.ln();
                }
            }
        }
        ll
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CovKind {
    Independence,
    Exchangeable,
    Ar1,
}
impl CovKind {
    fn from_str(s: &str) -> Result<CovKind, String> {
        match s {
            "independence" => Ok(CovKind::Independence),
            "exchangeable" => Ok(CovKind::Exchangeable),
            "ar1" => Ok(CovKind::Ar1),
            "unstructured" => Err(
                "ValueError: time must be provided and must have integer dtype".to_string(),
            ),
            _ => Err(format!(
                "unknown cov structure '{}'. valid: ['independence', 'exchangeable', 'ar1', 'unstructured']",
                s
            )),
        }
    }
}

#[derive(Clone)]
pub struct Cluster {
    pub y: Vec<f64>,
    pub x: Vec<Vec<f64>>,
    /// Within-cluster position of each row (0-based, original-obs index).
    pub time: Vec<f64>,
}

/// GLM IRLS starting values, replicating statsmodels GLM._fit_irls:
/// mu0 = starting_mu, eta0 = link(mu0), then weighted-least-squares
/// steps (SVD lstsq, min-norm) until the deviance is stable. For a
/// singular design the final step is the min-norm WLS solution, which
/// is what fixes the unidentifiable component in gee_ord.
fn glm_start(y: &[f64], x: &[Vec<f64>], fam: Fam) -> Vec<f64> {
    let n = y.len();
    let k = if x.is_empty() { 0 } else { x[0].len() };
    if n == 0 || k == 0 {
        return vec![0.0; k];
    }
    let mut mu = fam.starting_mu(y);
    let mut eta: Vec<f64> = mu.iter().map(|m| fam.link(*m)).collect();
    let mut beta = vec![0.0; k];
    // Deviance for the convergence check (statsmodels tol_criterion).
    let deviance = |y: &[f64], mu: &[f64], fam: Fam| -> f64 {
        match fam {
            Fam::Gaussian => y.iter().zip(mu.iter()).map(|(a, b)| (a - b).powi(2)).sum(),
            Fam::Binomial => {
                let mut d = 0.0;
                for (a, b) in y.iter().zip(mu.iter()) {
                    let m = b.clamp(1e-12, 1.0 - 1e-12);
                    let t1 = if *a > 0.0 { a * (a / m).ln() } else { 0.0 };
                    let t2 = if *a < 1.0 { (1.0 - a) * ((1.0 - a) / (1.0 - m)).ln() } else { 0.0 };
                    d += 2.0 * (t1 + t2);
                }
                d
            }
            Fam::Poisson => {
                let mut d = 0.0;
                for (a, b) in y.iter().zip(mu.iter()) {
                    let m = b.max(1e-12);
                    let t = if *a > 0.0 { a * (a / m).ln() } else { 0.0 };
                    d += 2.0 * (t - (a - m));
                }
                d
            }
            Fam::Gamma => {
                let mut d = 0.0;
                for (a, b) in y.iter().zip(mu.iter()) {
                    let (aa, mm) = (a.max(1e-12), b.max(1e-12));
                    d += 2.0 * ((aa - mm) / mm - (aa / mm).ln());
                }
                d
            }
        }
    };
    let mut prev_dev = deviance(y, &mu, fam);
    for _ in 0..100 {
        let w: Vec<f64> = mu.iter().map(|m| fam.irls_weight(*m)).collect();
        let z: Vec<f64> = eta
            .iter()
            .zip(mu.iter())
            .zip(y.iter())
            .map(|((e, m), yi)| e + fam.link_deriv(*m) * (yi - m))
            .collect();
        // Weighted LS via SVD on sqrt(w) * X.
        let mut mat = DMatrix::<f64>::zeros(n, k);
        let mut rhs = DVector::<f64>::zeros(n);
        for i in 0..n {
            let sw = w[i].max(0.0).sqrt();
            for j in 0..k {
                mat[(i, j)] = x[i][j] * sw;
            }
            rhs[i] = z[i] * sw;
        }
        let svd = mat.clone().svd(true, true);
        let beta_new = match (svd.u, svd.v_t) {
            (Some(u), Some(vt)) => {
                let s = &svd.singular_values;
                let smax = s.iter().cloned().fold(0.0f64, f64::max);
                let tol = smax * 1e-12 * (n.max(k) as f64);
                let uty = u.transpose() * &rhs;
                let mut tmp = DVector::<f64>::zeros(k);
                for i in 0..k {
                    if s[i] > tol {
                        tmp[i] = uty[i] / s[i];
                    }
                }
                vt.transpose() * tmp
            }
            _ => DVector::<f64>::zeros(k),
        };
        beta = beta_new.iter().cloned().collect();
        eta = x
            .iter()
            .map(|row| row.iter().zip(beta.iter()).map(|(a, b)| a * b).sum())
            .collect();
        mu = eta.iter().map(|e| fam.mu(*e)).collect();
        let dev = deviance(y, &mu, fam);
        if (prev_dev - dev).abs() <= 1e-8 {
            break;
        }
        prev_dev = dev;
    }
    beta
}

#[derive(Clone, Copy, PartialEq)]
pub enum GeeMode {
    Plain,
    Nominal { ncut: usize },
}

pub struct GeeOut {
    pub beta: Vec<f64>,
    pub se: Vec<Option<f64>>,
    pub scale: f64,
    pub converged: bool,
    pub mu: Vec<f64>, // per expanded row, in cluster order
    pub y: Vec<f64>,
}

fn cluster_mu_dmat(
    cl: &Cluster,
    beta: &[f64],
    fam: Fam,
    mode: GeeMode,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = cl.y.len();
    let k = beta.len();
    let eta: Vec<f64> = cl
        .x
        .iter()
        .map(|row| row.iter().zip(beta.iter()).map(|(a, b)| a * b).sum())
        .collect();
    match mode {
        GeeMode::Plain => {
            let mu: Vec<f64> = eta.iter().map(|e| fam.mu(*e)).collect();
            let dmat: Vec<Vec<f64>> = cl
                .x
                .iter()
                .zip(mu.iter())
                .map(|(row, m)| {
                    let d = fam.dmu_deta(*m);
                    row.iter().map(|v| v * d).collect()
                })
                .collect();
            (mu, dmat)
        }
        GeeMode::Nominal { ncut } => {
            // Multinomial probabilities per block of ncut rows, and
            // statsmodels' mean_deriv (first term minus the per-row
            // second term): row k -> prob_k * (1 - prob_k) * x_row.
            let mut mu = vec![0.0; n];
            let mut dmat = vec![vec![0.0; k]; n];
            let mut i = 0;
            while i < n {
                let block = (i..(i + ncut).min(n)).collect::<Vec<_>>();
                let mut denom = 1.0;
                for &r in &block {
                    denom += eta[r].exp().min(1e300);
                }
                for &r in &block {
                    let p = eta[r].exp().min(1e300) / denom;
                    mu[r] = p;
                    let f = p * (1.0 - p);
                    for j in 0..k {
                        dmat[r][j] = cl.x[r][j] * f;
                    }
                }
                i += ncut;
            }
            (mu, dmat)
        }
    }
}

/// Working covariance solve: returns V^{-1} * rhs for each rhs column
/// set (rhs given as rows x cols). R is the working correlation.
fn vinv_apply(
    r: &DMatrix<f64>,
    sdev: &[f64],
    rhs: &DMatrix<f64>,
) -> Option<DMatrix<f64>> {
    let n = sdev.len();
    let mut v = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            v[(i, j)] = r[(i, j)] * sdev[i] * sdev[j];
        }
    }
    solve_mat_or_pinv(&v, rhs)
}

fn gee_fit(
    clusters: &[Cluster],
    fam: Fam,
    cov: CovKind,
    mode: GeeMode,
    start: Vec<f64>,
) -> Result<GeeOut, String> {
    let k = start.len();
    let n_total: usize = clusters.iter().map(|c| c.y.len()).sum();
    if k == 0 || n_total == 0 {
        return Err("no data for GEE fit".to_string());
    }
    let max_group = clusters.iter().map(|c| c.y.len()).max().unwrap_or(0);
    let update_dep = max_group > 1;
    // Variance function: nominal uses binary variance, like _Multinomial.
    let varf = |mu: f64| -> f64 {
        match mode {
            GeeMode::Nominal { .. } => (mu * (1.0 - mu)).max(1e-12),
            _ => fam.var(mu),
        }
    };
    let estimate_scale = |mus: &[Vec<f64>]| -> f64 {
        match mode {
            GeeMode::Nominal { .. } => 1.0,
            _ => match fam {
                Fam::Binomial | Fam::Poisson => 1.0,
                _ => {
                    let mut ss = 0.0;
                    for (ci, cl) in clusters.iter().enumerate() {
                        for (j, yi) in cl.y.iter().enumerate() {
                            let r = (yi - mus[ci][j]) / varf(mus[ci][j]).sqrt();
                            ss += r * r;
                        }
                    }
                    if n_total > k {
                        ss / (n_total - k) as f64
                    } else {
                        ss / n_total as f64
                    }
                }
            },
        }
    };
    // Working correlation matrix for one cluster at the current dep.
    let corr_mat = |cl: &Cluster, dep: f64| -> DMatrix<f64> {
        let n = cl.y.len();
        let mut r = DMatrix::<f64>::zeros(n, n);
        match cov {
            CovKind::Independence => {
                for i in 0..n {
                    r[(i, i)] = 1.0;
                }
            }
            CovKind::Exchangeable => {
                for i in 0..n {
                    for j in 0..n {
                        r[(i, j)] = if i == j { 1.0 } else { dep };
                    }
                }
            }
            CovKind::Ar1 => {
                for i in 0..n {
                    for j in 0..n {
                        let d = (cl.time[i] - cl.time[j]).abs();
                        r[(i, j)] = if d == 0.0 { 1.0 } else { dep.powf(d) };
                    }
                }
            }
        }
        r
    };
    let mut beta = start;
    let mut dep = 0.0f64;
    let mut num_assoc = 0usize;
    let mut del = f64::INFINITY;
    let mut converged = false;
    // Cached means/dmat per cluster.
    let mut mus: Vec<Vec<f64>> = Vec::new();
    let mut dmats: Vec<Vec<Vec<f64>>> = Vec::new();
    let refresh = |beta: &[f64]| -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let mut ms = Vec::new();
        let mut ds = Vec::new();
        for cl in clusters {
            let (m, d) = cluster_mu_dmat(cl, beta, fam, mode);
            ms.push(m);
            ds.push(d);
        }
        (ms, ds)
    };
    let (m0, d0) = refresh(&beta);
    mus = m0;
    dmats = d0;
    for _itr in 0..60 {
        // _update_mean_params
        let mut bmat = DMatrix::<f64>::zeros(k, k);
        let mut score = DVector::<f64>::zeros(k);
        for (ci, cl) in clusters.iter().enumerate() {
            let mu = &mus[ci];
            let dmat = &dmats[ci];
            let n = cl.y.len();
            let sdev: Vec<f64> = mu.iter().map(|m| varf(*m).sqrt()).collect();
            let rmat = corr_mat(cl, dep);
            let mut rhs = DMatrix::<f64>::zeros(n, k + 1);
            for i in 0..n {
                for j in 0..k {
                    rhs[(i, j)] = dmat[i][j];
                }
                rhs[(i, k)] = cl.y[i] - mu[i];
            }
            let sol = match vinv_apply(&rmat, &sdev, &rhs) {
                Some(s) => s,
                None => return Err("singular working covariance in GEE update".to_string()),
            };
            // bmat += D' V^{-1} D ; score += D' V^{-1} resid
            for a in 0..k {
                for b in 0..k {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += dmat[i][a] * sol[(i, b)];
                    }
                    bmat[(a, b)] += s;
                }
                let mut s = 0.0;
                for i in 0..n {
                    s += dmat[i][a] * sol[(i, k)];
                }
                score[a] += s;
            }
        }
        // pinv, not LU: on the rank-deficient ordinal/nominal designs
        // bmat is singular and an LU "solution" drifts the unidentifiable
        // component arbitrarily (observed: 5e7 blow-up, then 0.23 shifts).
        // pinv gives min-norm updates, preserving the GLM start's
        // null component; for well-conditioned systems it equals solve.
        let update = pinv(&bmat) * &score;
        if !update.iter().all(|v| v.is_finite()) {
            return Err("singular matrix in GEE update".to_string());
        }
        for i in 0..k {
            beta[i] += update[i];
        }
        let (m1, d1) = refresh(&beta);
        mus = m1;
        dmats = d1;
        del = score.iter().map(|v| v * v).sum::<f64>().sqrt();
        if del < 1e-6 && (num_assoc > 0 || !update_dep) {
            converged = true;
            break;
        }
        // _update_assoc
        if cov != CovKind::Independence {
            dep = update_dep_param(clusters, &mus, &varf, cov, k, n_total, &estimate_scale(&mus));
            num_assoc += 1;
        } else if update_dep {
            num_assoc += 1;
        }
    }
    if !converged && del < 1e-6 {
        converged = true;
    }
    // _covmat (robust sandwich)
    let mut bmat = DMatrix::<f64>::zeros(k, k);
    let mut cmat = DMatrix::<f64>::zeros(k, k);
    for (ci, cl) in clusters.iter().enumerate() {
        let mu = &mus[ci];
        let dmat = &dmats[ci];
        let n = cl.y.len();
        let sdev: Vec<f64> = mu.iter().map(|m| varf(*m).sqrt()).collect();
        let rmat = corr_mat(cl, dep);
        let mut rhs = DMatrix::<f64>::zeros(n, k + 1);
        for i in 0..n {
            for j in 0..k {
                rhs[(i, j)] = dmat[i][j];
            }
            rhs[(i, k)] = cl.y[i] - mu[i];
        }
        let sol = match vinv_apply(&rmat, &sdev, &rhs) {
            Some(s) => s,
            None => return Err("singular working covariance in GEE covariance".to_string()),
        };
        for a in 0..k {
            for b in 0..k {
                let mut s = 0.0;
                for i in 0..n {
                    s += dmat[i][a] * sol[(i, b)];
                }
                bmat[(a, b)] += s;
            }
        }
        let mut dv = DVector::<f64>::zeros(k);
        for a in 0..k {
            let mut s = 0.0;
            for i in 0..n {
                s += dmat[i][a] * sol[(i, k)];
            }
            dv[a] = s;
        }
        for a in 0..k {
            for b in 0..k {
                cmat[(a, b)] += dv[a] * dv[b];
            }
        }
    }
    let bmati = pinv(&bmat);
    let cov_robust = &bmati * &cmat * &bmati;
    let mut se = Vec::new();
    for i in 0..k {
        let v = cov_robust[(i, i)];
        se.push(if v.is_finite() && v >= 0.0 { Some(v.sqrt()) } else { None });
    }
    let scale = estimate_scale(&mus);
    let mut mu_flat = Vec::new();
    let mut y_flat = Vec::new();
    for (ci, cl) in clusters.iter().enumerate() {
        mu_flat.extend(mus[ci].iter().cloned());
        y_flat.extend(cl.y.iter().cloned());
    }
    Ok(GeeOut {
        beta,
        se,
        scale,
        converged,
        mu: mu_flat,
        y: y_flat,
    })
}

/// Association-parameter update, replicating statsmodels cov_struct:
/// Exchangeable moment estimator; AR(1) weighted nonlinear least
/// squares (nogrid, Brent bracketing as in statsmodels).
fn update_dep_param(
    clusters: &[Cluster],
    mus: &[Vec<f64>],
    varf: &dyn Fn(f64) -> f64,
    cov: CovKind,
    p: usize,
    n_total: usize,
    scale_est: &f64,
) -> f64 {
    match cov {
        CovKind::Independence => 0.0,
        CovKind::Exchangeable => {
            let mut ss = 0.0;
            let mut cross = 0.0;
            let mut npairs = 0.0;
            for (ci, cl) in clusters.iter().enumerate() {
                let r: Vec<f64> = cl
                    .y
                    .iter()
                    .zip(mus[ci].iter())
                    .map(|(yi, mi)| (yi - mi) / varf(*mi).sqrt())
                    .collect();
                let s: f64 = r.iter().sum();
                let ssr: f64 = r.iter().map(|v| v * v).sum();
                ss += ssr;
                cross += (s * s - ssr) / 2.0;
                let n = r.len() as f64;
                npairs += 0.5 * n * (n - 1.0);
            }
            if npairs == 0.0 {
                return 0.0;
            }
            let scale = if n_total > p {
                ss / (n_total - p) as f64
            } else {
                ss / n_total as f64
            };
            if scale == 0.0 {
                return 0.0;
            }
            let denom = npairs * (npairs - p as f64) / npairs;
            if denom == 0.0 {
                return 0.0;
            }
            (cross / scale) / denom
        }
        CovKind::Ar1 => {
            // Pairwise distances and standardised residuals.
            let mut dists: Vec<f64> = Vec::new();
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for (ci, cl) in clusters.iter().enumerate() {
                let r: Vec<f64> = cl
                    .y
                    .iter()
                    .zip(mus[ci].iter())
                    .map(|(yi, mi)| (yi - mi) / (*scale_est * varf(*mi)).sqrt().max(1e-12))
                    .collect();
                for j1 in 0..cl.y.len() {
                    for j2 in 0..j1 {
                        dists.push((cl.time[j1] - cl.time[j2]).abs());
                        pairs.push((r[j1], r[j2]));
                    }
                }
            }
            if pairs.is_empty() {
                return 0.0;
            }
            let fitfunc = |a: f64| -> f64 {
                let mut wts: Vec<f64> = Vec::new();
                for &d in &dists {
                    let var = if d == 0.0 {
                        1.0
                    } else if (1.0 - a * a).abs() < 1e-12 {
                        1e12
                    } else {
                        (1.0 - a.powf(2.0 * d)) / (1.0 - a * a)
                    };
                    wts.push(if var > 0.0 { 1.0 / var } else { 0.0 });
                }
                let wsum: f64 = wts.iter().sum();
                if wsum == 0.0 {
                    return f64::INFINITY;
                }
                let mut s = 0.0;
                for (i, (r1, r2)) in pairs.iter().enumerate() {
                    let dif = r1 - a.powf(dists[i]) * r2;
                    s += dif * dif * wts[i] / wsum;
                }
                s
            };
            let b_lft = 0.0f64;
            let f_lft = fitfunc(b_lft);
            let mut b_ctr = 0.5f64;
            let mut f_ctr = fitfunc(b_ctr);
            while f_ctr > f_lft {
                b_ctr /= 2.0;
                f_ctr = fitfunc(b_ctr);
                if b_ctr < 1e-8 {
                    return 0.0;
                }
            }
            let mut b_rgt = 0.75f64;
            let mut f_rgt = fitfunc(b_rgt);
            while f_rgt < f_ctr {
                b_rgt += (1.0 - b_rgt) / 2.0;
                f_rgt = fitfunc(b_rgt);
                if b_rgt > 1.0 - 1e-6 {
                    return 0.0;
                }
            }
            // Brent minimisation on [b_lft, b_rgt].
            brent_min(&fitfunc, b_lft, b_rgt)
        }
    }
}

/// Brent's method for univariate minimisation (scipy-brent equivalent
/// tolerance; used for the AR(1) dependence parameter).
fn brent_min(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let tol = 2e-12f64;
    let cgold = 0.3819660112501051f64;
    let mut x = a + cgold * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut d = 0.0f64;
    let mut e = 0.0f64;
    for _ in 0..500 {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-12;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            break;
        }
        let mut p = 0.0;
        let mut q = 0.0;
        let mut r = 0.0;
        if (e).abs() > tol1 {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() >= (0.5 * q * etemp).abs() || p <= q * (a - x) || p >= q * (b - x) {
                e = if x >= xm { a - x } else { b - x };
                d = cgold * e;
            } else {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = if xm - x >= 0.0 { tol1 } else { -tol1 };
                }
            }
        } else {
            e = if x >= xm { a - x } else { b - x };
            d = cgold * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + if d >= 0.0 { tol1 } else { -tol1 }
        };
        let fu = f(u);
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    x
}

/// Build clusters from cleaned rows, preserving input order within a
/// cluster; clusters sorted by group value (statsmodels np.unique order).
fn make_clusters(clean: &Clean) -> Vec<Cluster> {
    let mut keys: Vec<f64> = Vec::new();
    for g in &clean.groups {
        if !keys.contains(g) {
            keys.push(*g);
        }
    }
    keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out = Vec::new();
    for gk in keys {
        let mut cl = Cluster { y: vec![], x: vec![], time: vec![] };
        let mut pos = 0.0;
        for i in 0..clean.n {
            if clean.groups[i] == gk {
                cl.y.push(clean.y[i]);
                cl.x.push(clean.x[i].clone());
                cl.time.push(pos);
                pos += 1.0;
            }
        }
        out.push(cl);
    }
    out
}

/// GEE log-likelihood / AIC / BIC, matching GEEResults (GLMResults):
/// Gaussian uses the concentrated scale RSS/N; other families use the
/// family loglike at the estimated scale. Returns (ll, aic, bic).
fn gee_ll(fam: Fam, y: &[f64], mu: &[f64], scale: f64, p: usize, n: usize) -> (f64, f64, f64) {
    let ll = match fam {
        Fam::Gaussian => {
            let rss: f64 = y.iter().zip(mu.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            let sc = if n > 0 { rss / n as f64 } else { scale };
            let sc = if sc > 0.0 { sc } else { 1e-12 };
            fam.loglike(y, mu, sc)
        }
        _ => fam.loglike(y, mu, scale),
    };
    let aic = -2.0 * ll + 2.0 * p as f64;
    let bic = -2.0 * ll + (n as f64).ln() * p as f64;
    (ll, aic, bic)
}

fn gee_output(
    names: &[String],
    out: &GeeOut,
    fam: Fam,
    ll_info: Option<(f64, f64, f64)>,
    n_orig: usize,
) -> Value {
    let mut m = Map::new();
    if let Some((ll, aic, bic)) = ll_info {
        m.insert("ll".to_string(), f6(ll));
        m.insert("aic".to_string(), f6(aic));
        m.insert("bic".to_string(), f6(bic));
    }
    m.insert("v".to_string(), f6(out.scale));
    let mut fmap = Map::new();
    for (i, name) in names.iter().enumerate() {
        fmap.insert(name.clone(), coef_row(out.beta[i], out.se[i]));
    }
    m.insert("f".to_string(), Value::Object(fmap));
    m.insert("cnvg".to_string(), json!(out.converged));
    m.insert("n".to_string(), json!(n_orig));
    Value::Object(m)
}

// ------------------------------------------------------------ dispatch

/// Arguments for the statsmodels actions (mirrors statsmodels_run's
/// kwargs; g is the grouping/subject column, w the within factors).
#[derive(Default, Clone)]
pub struct SmArgs {
    pub o: String,
    pub p: Vec<String>,
    pub g: String,
    pub fam: String,
    pub cov: String,
    pub w: Vec<String>,
    pub n_imp: i64,
    pub model: String,
}

pub fn dispatch_sm(action: &str, tab: &Table, a: &SmArgs) -> Result<Value, String> {
    match action {
        "gee" => action_gee(tab, a),
        "gee_ord" => action_gee_ord(tab, a),
        "gee_nom" => action_gee_nom(tab, a),
        "anova_rm" => action_anova_rm(tab, a),
        "mice" => action_mice(tab, a),
        "mlm" => action_mlm(tab, a),
        _ => Err(format!(
            "unknown action '{}'. valid: ['mlm', 'gee', 'gee_ord', 'gee_nom', 'anova_rm', 'mice']",
            action
        )),
    }
}

fn action_gee(tab: &Table, a: &SmArgs) -> Result<Value, String> {
    if a.o.is_empty() {
        return Err("gee requires 'o' (endog/dependent variable)".to_string());
    }
    if a.g.is_empty() {
        return Err("gee requires 'g' (grouping variable)".to_string());
    }
    let fam = Fam::from_str(if a.fam.is_empty() { "gaussian" } else { &a.fam }).ok_or_else(|| {
        format!(
            "unknown family '{}'. valid: ['gaussian', 'binomial', 'poisson', 'gamma']",
            a.fam
        )
    })?;
    let cov = CovKind::from_str(if a.cov.is_empty() { "exchangeable" } else { &a.cov })?;
    let clean = clean_arrays(tab, &a.o, &a.p, Some(&a.g))?;
    let clusters = make_clusters(&clean);
    let start = glm_start(&clean.y, &clean.x, fam);
    let out = gee_fit(&clusters, fam, cov, GeeMode::Plain, start)?;
    let n_rows = out.y.len();
    let ll_info = gee_ll(fam, &out.y, &out.mu, out.scale, clean.names.len(), n_rows);
    Ok(gee_output(&clean.names, &out, fam, Some(ll_info), clean.n))
}

fn action_gee_ord(tab: &Table, a: &SmArgs) -> Result<Value, String> {
    if a.o.is_empty() {
        return Err("gee_ord requires 'o' (endog/dependent variable)".to_string());
    }
    if a.g.is_empty() {
        return Err("gee_ord requires 'g' (grouping variable)".to_string());
    }
    let cov = CovKind::from_str(if a.cov.is_empty() { "exchangeable" } else { &a.cov })?;
    let clean = clean_arrays(tab, &a.o, &a.p, Some(&a.g))?;
    // Integer outcomes, cuts = unique values except the greatest.
    let yint: Vec<f64> = clean.y.iter().map(|v| v.trunc()).collect();
    let mut vals: Vec<f64> = yint.clone();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    vals.dedup_by(|x, y| *x == *y);
    if vals.len() < 2 {
        return Err("gee_ord requires at least two outcome levels".to_string());
    }
    let cuts = &vals[..vals.len() - 1];
    let ncut = cuts.len();
    let mut names: Vec<String> = cuts.iter().map(|c| format!("I(y>{:.1})", c)).collect();
    names.extend(clean.names.iter().cloned());
    // Expanded clusters, preserving cluster order and within-cluster
    // position as time (statsmodels repeats the original time per cut).
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut keys: Vec<f64> = clean.groups.clone();
    keys.sort_by(|x, y| x.partial_cmp(y).unwrap());
    keys.dedup_by(|x, y| *x == *y);
    let mut y_exp_all: Vec<f64> = Vec::new();
    let mut x_exp_all: Vec<Vec<f64>> = Vec::new();
    for gk in &keys {
        let mut cl = Cluster { y: vec![], x: vec![], time: vec![] };
        let mut pos = 0.0;
        for i in 0..clean.n {
            if clean.groups[i] != *gk {
                continue;
            }
            for (ci, cut) in cuts.iter().enumerate() {
                let mut row = vec![0.0; ncut];
                row[ci] = 1.0;
                row.extend(clean.x[i].iter().cloned());
                let ev = if yint[i] > *cut { 1.0 } else { 0.0 };
                cl.y.push(ev);
                cl.x.push(row.clone());
                cl.time.push(pos);
                y_exp_all.push(ev);
                x_exp_all.push(row);
            }
            pos += 1.0;
        }
        clusters.push(cl);
    }
    let start = glm_start(&y_exp_all, &x_exp_all, Fam::Binomial);
    let out = gee_fit(&clusters, Fam::Binomial, cov, GeeMode::Plain, start)?;
    let n_exp = out.y.len();
    let ll_info = gee_ll(Fam::Binomial, &out.y, &out.mu, out.scale, names.len(), n_exp);
    // NOTE: no "th" key, matching the Python tool (dead hasattr check).
    Ok(gee_output(&names, &out, Fam::Binomial, Some(ll_info), clean.n))
}

fn action_gee_nom(tab: &Table, a: &SmArgs) -> Result<Value, String> {
    if a.o.is_empty() {
        return Err("gee_nom requires 'o' (endog/dependent variable)".to_string());
    }
    if a.g.is_empty() {
        return Err("gee_nom requires 'g' (grouping variable)".to_string());
    }
    let cov = CovKind::from_str(if a.cov.is_empty() { "exchangeable" } else { &a.cov })?;
    let clean = clean_arrays(tab, &a.o, &a.p, Some(&a.g))?;
    let yint: Vec<f64> = clean.y.iter().map(|v| v.trunc()).collect();
    let mut vals: Vec<f64> = yint.clone();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    vals.dedup_by(|x, y| *x == *y);
    if vals.len() < 2 {
        return Err("gee_nom requires at least two outcome levels".to_string());
    }
    let cuts = &vals[..vals.len() - 1];
    let ncut = cuts.len();
    let p0 = clean.names.len();
    let mut names: Vec<String> = Vec::new();
    for cut in cuts {
        for nm in &clean.names {
            names.push(format!("{}[{:.1}]", nm, cut));
        }
    }
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut keys: Vec<f64> = clean.groups.clone();
    keys.sort_by(|x, y| x.partial_cmp(y).unwrap());
    keys.dedup_by(|x, y| *x == *y);
    let mut y_exp_all: Vec<f64> = Vec::new();
    let mut x_exp_all: Vec<Vec<f64>> = Vec::new();
    for gk in &keys {
        let mut cl = Cluster { y: vec![], x: vec![], time: vec![] };
        let mut pos = 0.0;
        for i in 0..clean.n {
            if clean.groups[i] != *gk {
                continue;
            }
            for (ci, cut) in cuts.iter().enumerate() {
                let mut row = vec![0.0; ncut * p0];
                for j in 0..p0 {
                    row[ci * p0 + j] = clean.x[i][j];
                }
                let ev = if yint[i] == *cut { 1.0 } else { 0.0 };
                cl.y.push(ev);
                cl.x.push(row.clone());
                cl.time.push(pos);
                y_exp_all.push(ev);
                x_exp_all.push(row);
            }
            pos += 1.0;
        }
        clusters.push(cl);
    }
    let start = glm_start(&y_exp_all, &x_exp_all, Fam::Binomial);
    let out = gee_fit(&clusters, Fam::Binomial, cov, GeeMode::Nominal { ncut }, start)?;
    // NominalGEE llf raises in Python -> no ll/aic/bic keys.
    Ok(gee_output(&names, &out, Fam::Binomial, None, clean.n))
}

// ------------------------------------------------------------ anova_rm

/// Helmert contrast column j (0-based) for L levels: level j+1 gets 1,
/// levels 0..=j get -1/(j+1), later levels 0. Orthogonal, sums to 0.
fn helmert_col(l: usize, j: usize) -> f64 {
    if l == j + 1 {
        1.0
    } else if l <= j {
        -1.0 / (j + 1) as f64
    } else {
        0.0
    }
}

fn action_anova_rm(tab: &Table, a: &SmArgs) -> Result<Value, String> {
    if a.o.is_empty() {
        return Err("anova_rm requires 'o' (dependent variable)".to_string());
    }
    if a.g.is_empty() {
        return Err("anova_rm requires 'g' (subject identifier)".to_string());
    }
    if a.w.is_empty() {
        return Err(
            "anova_rm requires at least one within-subject factor ('w'). For between-subject or mixed designs, use the 'mlm' action."
                .to_string(),
        );
    }
    let mut cols = vec![a.o.clone(), a.g.clone()];
    cols.extend(a.w.iter().cloned());
    let missing: Vec<String> = cols
        .iter()
        .filter(|c| tab.col_idx(c).is_none())
        .cloned()
        .collect();
    if !missing.is_empty() {
        let list = missing
            .iter()
            .map(|m| format!("'{}'", m))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("columns not in data: [{}]", list));
    }
    // Complete cases, in input order.
    let endog = tab.col(&a.o).unwrap();
    let subj = tab.col(&a.g).unwrap();
    let within: Vec<Vec<Cell>> = a.w.iter().map(|w| tab.col(w).unwrap()).collect();
    let mut rows: Vec<(String, Vec<String>, f64)> = Vec::new();
    for i in 0..endog.len() {
        let yv = match &endog[i] {
            Cell::Num(v) if v.is_finite() => *v,
            _ => continue,
        };
        let sv = match &subj[i] {
            Cell::Miss => continue,
            Cell::Num(v) if !v.is_finite() => continue,
            c => c.key(),
        };
        let mut lv = Vec::new();
        let mut ok = true;
        for wcol in &within {
            match &wcol[i] {
                Cell::Miss => {
                    ok = false;
                    break;
                }
                Cell::Num(v) if !v.is_finite() => {
                    ok = false;
                    break;
                }
                c => lv.push(c.key()),
            }
        }
        if !ok {
            continue;
        }
        rows.push((sv, lv, yv));
    }
    if rows.is_empty() {
        return Err("no complete cases after dropping missing values".to_string());
    }
    // Subjects (first-appearance order) and levels per factor.
    let mut subjects: Vec<String> = Vec::new();
    for r in &rows {
        if !subjects.contains(&r.0) {
            subjects.push(r.0.clone());
        }
    }
    let nf = a.w.len();
    let mut levels: Vec<Vec<String>> = Vec::new();
    for f in 0..nf {
        let mut lv: Vec<String> = Vec::new();
        for r in &rows {
            if !lv.contains(&r.1[f]) {
                lv.push(r.1[f].clone());
            }
        }
        levels.push(lv);
    }
    // Cell lookup: (subject, cell index) -> y. Cells enumerate the
    // cartesian product, last factor fastest.
    let mut strides = vec![1usize; nf];
    for f in (0..nf - 1).rev() {
        strides[f] = strides[f + 1] * levels[f + 1].len();
    }
    let n_cells: usize = levels.iter().map(|l| l.len()).product();
    let mut cell_of = |lv: &[String]| -> usize {
        let mut idx = 0;
        for f in 0..nf {
            let li = levels[f].iter().position(|x| x == &lv[f]).unwrap();
            idx += li * strides[f];
        }
        idx
    };
    let s_count = subjects.len();
    let mut ymat: Vec<Vec<Option<f64>>> = vec![vec![None; n_cells]; s_count];
    for r in &rows {
        let si = subjects.iter().position(|x| x == &r.0).unwrap();
        let ci = cell_of(&r.1);
        if ymat[si][ci].is_some() {
            return Err(
                "anova_rm requires a balanced design: each subject must have exactly one observation per within-cell combination"
                    .to_string(),
            );
        }
        ymat[si][ci] = Some(r.2);
    }
    for si in 0..s_count {
        for ci in 0..n_cells {
            if ymat[si][ci].is_none() {
                return Err(
                    "anova_rm requires a balanced design: each subject must have exactly one observation per within-cell combination"
                        .to_string(),
                );
            }
        }
    }
    let ymat: Vec<Vec<f64>> = ymat
        .iter()
        .map(|row| row.iter().map(|v| v.unwrap()).collect())
        .collect();
    // Contrast columns per term, as functions cell -> value.
    // A term is a non-empty subset of factors; terms ordered by size,
    // then by factor-index combination (matches AnovaRM's table).
    let mut terms: Vec<Vec<usize>> = Vec::new();
    for size in 1..=nf {
        let mut combo: Vec<usize> = (0..size).collect();
        loop {
            terms.push(combo.clone());
            // next combination
            let mut i = size;
            loop {
                if i == 0 {
                    break;
                }
                i -= 1;
                if combo[i] < nf - size + i {
                    combo[i] += 1;
                    for j in i + 1..size {
                        combo[j] = combo[j - 1] + 1;
                    }
                    break;
                }
                if i == 0 {
                    // exhausted; signal via sentinel
                    combo = vec![usize::MAX; size];
                    break;
                }
            }
            if combo[0] == usize::MAX {
                break;
            }
        }
    }
    // Cell level vector per cell index.
    let mut cell_levels: Vec<Vec<usize>> = vec![vec![0; nf]; n_cells];
    for ci in 0..n_cells {
        for f in 0..nf {
            cell_levels[ci][f] = (ci / strides[f]) % levels[f].len();
        }
    }
    let mut anova = Map::new();
    for term in &terms {
        // Contrast columns: product of Helmert columns over term factors.
        let mut col_idx: Vec<Vec<usize>> = term
            .iter()
            .map(|&f| (0..levels[f].len() - 1).collect())
            .collect();
        // Cartesian product of contrast indices.
        let mut combos: Vec<Vec<usize>> = vec![Vec::new()];
        for choices in &col_idx {
            let mut next = Vec::new();
            for base in &combos {
                for &ch in choices {
                    let mut b = base.clone();
                    b.push(ch);
                    next.push(b);
                }
            }
            combos = next;
        }
        col_idx.clear();
        let _ = col_idx;
        let df_term: usize = term.iter().map(|&f| levels[f].len() - 1).product();
        let mut ss = 0.0;
        let mut ss_err = 0.0;
        for combo in &combos {
            // Cell contrast vector.
            let mut cvec = vec![0.0; n_cells];
            for ci in 0..n_cells {
                let mut v = 1.0;
                for (ti, &f) in term.iter().enumerate() {
                    v *= helmert_col(cell_levels[ci][f], combo[ti]);
                }
                cvec[ci] = v;
            }
            let cc: f64 = cvec.iter().map(|v| v * v).sum();
            if cc == 0.0 {
                continue;
            }
            // Per-subject projection coefficients.
            let mut b_sub = Vec::new();
            for si in 0..s_count {
                let mut dot = 0.0;
                for ci in 0..n_cells {
                    dot += cvec[ci] * ymat[si][ci];
                }
                b_sub.push(dot / cc);
            }
            let beta = mean(&b_sub);
            ss += cc * (s_count as f64) * beta * beta;
            for si in 0..s_count {
                let d = b_sub[si] - beta;
                ss_err += cc * d * d;
            }
        }
        let df_err = (s_count - 1) * df_term;
        let ms = if df_term > 0 { ss / df_term as f64 } else { 0.0 };
        let ms_err = if df_err > 0 { ss_err / df_err as f64 } else { 0.0 };
        let fval = if ms_err > 0.0 {
            ms / ms_err
        } else if ms > 0.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
        let p = if fval.is_finite() && df_term > 0 && df_err > 0 {
            f_sf(fval, df_term as f64, df_err as f64)
        } else if fval == f64::INFINITY {
            0.0
        } else {
            f64::NAN
        };
        let name = term
            .iter()
            .map(|&f| a.w[f].clone())
            .collect::<Vec<_>>()
            .join(":");
        let mut entry = Map::new();
        entry.insert("f".to_string(), f6(fval));
        entry.insert("ndf".to_string(), json!(df_term));
        entry.insert("ddf".to_string(), json!(df_err));
        entry.insert("p".to_string(), f6(p));
        anova.insert(name, Value::Object(entry));
    }
    let mut out = Map::new();
    out.insert("anova".to_string(), Value::Object(anova));
    out.insert("n".to_string(), json!(rows.len()));
    Ok(Value::Object(out))
}

// ------------------------------------------------------------ mlm

/// Random-intercept linear mixed model by profile REML (statsmodels
/// MixedLM default: reml=True, random intercept only when exog_re is
/// not supplied). One-dimensional optimisation over
/// lambda = sigma_b^2 / sigma_e^2; beta and sigma_e^2 profiled out in
/// closed form per group via the Sherman-Morrison form.
fn action_mlm(tab: &Table, a: &SmArgs) -> Result<Value, String> {
    if a.o.is_empty() {
        return Err("mlm requires 'o' (endog/dependent variable)".to_string());
    }
    if a.g.is_empty() {
        return Err("mlm requires 'g' (grouping variable)".to_string());
    }
    let clean = clean_arrays(tab, &a.o, &a.p, Some(&a.g))?;
    let k = clean.names.len();
    let n = clean.n;
    if n <= k {
        return Err("mlm requires more observations than parameters".to_string());
    }
    // Groups in sorted order.
    let mut keys: Vec<f64> = clean.groups.clone();
    keys.sort_by(|x, y| x.partial_cmp(y).unwrap());
    keys.dedup_by(|x, y| *x == *y);
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for gk in &keys {
        let idx: Vec<usize> = (0..n).filter(|&i| clean.groups[i] == *gk).collect();
        groups.push(idx);
    }
    // Profile REML loglik for a given lambda.
    // Returns (ll_prof, beta, sigma2_e, logdet_XtWX, XtWX matrix).
    let profile = |lam: f64| -> Option<(f64, Vec<f64>, f64, f64, DMatrix<f64>)> {
        // W_i = I - lam/(1+lam*n_i) * J ; XtWX = sum X' W X.
        let mut xtwx = DMatrix::<f64>::zeros(k, k);
        let mut xtwy = DVector::<f64>::zeros(k);
        for idx in &groups {
            let ni = idx.len() as f64;
            let c = lam / (1.0 + lam * ni);
            // X' W X = X'X - c * (col sums)(col sums)'
            let mut xtx = DMatrix::<f64>::zeros(k, k);
            let mut xty = DVector::<f64>::zeros(k);
            let mut colsum = DVector::<f64>::zeros(k);
            let mut ysum = 0.0;
            for &i in idx {
                let row = &clean.x[i];
                for a2 in 0..k {
                    colsum[a2] += row[a2];
                    xty[a2] += row[a2] * clean.y[i];
                    for b2 in 0..k {
                        xtx[(a2, b2)] += row[a2] * row[b2];
                    }
                }
                ysum += clean.y[i];
            }
            for a2 in 0..k {
                for b2 in 0..k {
                    xtwx[(a2, b2)] += xtx[(a2, b2)] - c * colsum[a2] * colsum[b2];
                }
                xtwy[a2] += xty[a2] - c * colsum[a2] * ysum;
            }
        }
        let beta = solve_or_pinv(&xtwx, &xtwy)?;
        // r' W r
        let mut rwr = 0.0;
        for idx in &groups {
            let ni = idx.len() as f64;
            let c = lam / (1.0 + lam * ni);
            let mut sum_r = 0.0;
            let mut ss = 0.0;
            for &i in idx {
                let fit: f64 = clean.x[i].iter().zip(beta.iter()).map(|(a2, b2)| a2 * b2).sum();
                let r = clean.y[i] - fit;
                sum_r += r;
                ss += r * r;
            }
            rwr += ss - c * sum_r * sum_r;
        }
        let dof = (n - k) as f64;
        if dof <= 0.0 || rwr < 0.0 {
            return None;
        }
        let s2 = rwr / dof;
        if s2 <= 0.0 {
            return None;
        }
        let logdet = logdet_spd(&xtwx)?;
        let sum_log: f64 = groups
            .iter()
            .map(|idx| (1.0 + lam * idx.len() as f64).ln())
            .sum();
        // REML loglik (full constants):
        // -0.5[(N-p)log(2pi) + (N-p)log(s2) + sum log(1+lam n_i)
        //      + log|X'WX| + (N-p)]
        let ll = -0.5
            * (dof * (2.0 * std::f64::consts::PI).ln()
                + dof * s2.ln()
                + sum_log
                + logdet
                + dof);
        Some((ll, beta.iter().cloned().collect(), s2, logdet, xtwx))
    };
    // Grid + golden-section maximisation over log(lambda).
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_loglam = f64::NEG_INFINITY;
    let mut grid: Vec<(f64, f64)> = Vec::new();
    // lambda = 0 endpoint.
    if let Some((ll, _, _, _, _)) = profile(0.0) {
        grid.push((f64::NEG_INFINITY, ll));
        if ll > best_ll {
            best_ll = ll;
        }
    }
    let mut gx: f64 = -10.0;
    let mut grid_vals: Vec<(f64, f64)> = Vec::new();
    while gx <= 8.0001 {
        if let Some((ll, _, _, _, _)) = profile(gx.exp()) {
            grid_vals.push((gx, ll));
            if ll > best_ll {
                best_ll = ll;
                best_loglam = gx;
            }
        }
        gx += 0.2;
    }
    let mut lam_star: f64;
    if best_loglam.is_finite() {
        // Golden-section around the best grid point.
        let mut lo = best_loglam - 0.2;
        let mut hi = best_loglam + 0.2;
        let gr = (5.0f64.sqrt() - 1.0) / 2.0;
        let mut c1 = hi - gr * (hi - lo);
        let mut c2 = lo + gr * (hi - lo);
        let mut f1 = profile(c1.exp()).map(|t| t.0).unwrap_or(f64::NEG_INFINITY);
        let mut f2 = profile(c2.exp()).map(|t| t.0).unwrap_or(f64::NEG_INFINITY);
        for _ in 0..200 {
            if hi - lo < 1e-12 {
                break;
            }
            if f1 > f2 {
                hi = c2;
                c2 = c1;
                f2 = f1;
                c1 = hi - gr * (hi - lo);
                f1 = profile(c1.exp()).map(|t| t.0).unwrap_or(f64::NEG_INFINITY);
            } else {
                lo = c1;
                c1 = c2;
                f1 = f2;
                c2 = lo + gr * (hi - lo);
                f2 = profile(c2.exp()).map(|t| t.0).unwrap_or(f64::NEG_INFINITY);
            }
        }
        let loglam = (lo + hi) / 2.0;
        lam_star = loglam.exp();
        if let Some((ll, _, _, _, _)) = profile(lam_star) {
            if ll > best_ll {
                best_ll = ll;
            } else {
                // Keep the grid optimum if refinement did not improve.
                lam_star = best_loglam.exp();
            }
        }
        let _ = grid;
    } else {
        lam_star = 0.0;
    }
    let (ll, beta, s2, _logdet, xtwx) =
        profile(lam_star).ok_or_else(|| "mlm profile likelihood failed".to_string())?;
    // Fixed-effect covariance: s2 * (X'WX)^{-1} (== (X'V^{-1}X)^{-1}).
    let xtwx_inv = pinv(&xtwx);
    let mut se: Vec<Option<f64>> = Vec::new();
    for i in 0..k {
        let v = s2 * xtwx_inv[(i, i)];
        se.push(if v.is_finite() && v >= 0.0 { Some(v.sqrt()) } else { None });
    }
    // Random intercepts (BLUPs): b_i = lam * sum(r_i) / (1 + lam*n_i).
    let mut blups = Vec::new();
    for idx in &groups {
        let ni = idx.len() as f64;
        let mut sum_r = 0.0;
        for &i in idx {
            let fit: f64 = clean.x[i].iter().zip(beta.iter()).map(|(a2, b2)| a2 * b2).sum();
            sum_r += clean.y[i] - fit;
        }
        blups.push(lam_star * sum_r / (1.0 + lam_star * ni));
    }
    let mut out = Map::new();
    out.insert("ll".to_string(), f6(ll));
    // Python: MixedLM aic/bic are NaN -> formatter emits null.
    out.insert("aic".to_string(), Value::Null);
    out.insert("bic".to_string(), Value::Null);
    out.insert("v".to_string(), f6(s2));
    let mut fmap = Map::new();
    for (i, name) in clean.names.iter().enumerate() {
        fmap.insert(name.clone(), coef_row(beta[i], se[i]));
    }
    out.insert("f".to_string(), Value::Object(fmap));
    let mut rmap = Map::new();
    rmap.insert("mean".to_string(), f6(mean(&blups)));
    rmap.insert("sd".to_string(), f6(sd_ddof1(&blups)));
    rmap.insert("n".to_string(), json!(blups.len()));
    out.insert("r".to_string(), Value::Object(rmap));
    // Python: cov_re is a DataFrame, so rv stays {}.
    out.insert("rv".to_string(), Value::Object(Map::new()));
    out.insert("cnvg".to_string(), json!(true));
    out.insert("n".to_string(), json!(n));
    let _ = best_ll;
    Ok(Value::Object(out))
}

// ------------------------------------------------------------ mice

/// IRLS fit for logit / poisson analysis models: returns (beta, cov).
fn irls_fit(x: &[Vec<f64>], y: &[f64], kind: &str) -> Option<(Vec<f64>, DMatrix<f64>)> {
    let n = y.len();
    let k = if x.is_empty() { 0 } else { x[0].len() };
    if n == 0 || k == 0 {
        return None;
    }
    let fam = if kind == "logit" { Fam::Binomial } else { Fam::Poisson };
    let mut beta = vec![0.0; k];
    let my = mean(y);
    beta[0] = match kind {
        "logit" => {
            let p = my.clamp(1e-6, 1.0 - 1e-6);
            (p / (1.0 - p)).ln()
        }
        _ => my.max(1e-6).ln(),
    };
    for _ in 0..100 {
        let mut xtwx = DMatrix::<f64>::zeros(k, k);
        let mut score = DVector::<f64>::zeros(k);
        for i in 0..n {
            let eta: f64 = x[i].iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
            let mu = fam.mu(eta);
            let d = fam.dmu_deta(mu);
            let v = fam.var(mu);
            let w = d * d / v;
            let resid = y[i] - mu;
            for a2 in 0..k {
                score[a2] += x[i][a2] * d / v * resid;
                for b2 in 0..k {
                    xtwx[(a2, b2)] += w * x[i][a2] * x[i][b2];
                }
            }
        }
        let step = solve_or_pinv(&xtwx, &score)?;
        let mut mx: f64 = 0.0;
        for i in 0..k {
            beta[i] += step[i];
            mx = mx.max(step[i].abs());
        }
        if mx < 1e-10 {
            break;
        }
    }
    // Covariance = (X' W X)^{-1} at the final beta.
    let mut xtwx = DMatrix::<f64>::zeros(k, k);
    for i in 0..n {
        let eta: f64 = x[i].iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        let mu = fam.mu(eta);
        let d = fam.dmu_deta(mu);
        let w = d * d / fam.var(mu);
        for a2 in 0..k {
            for b2 in 0..k {
                xtwx[(a2, b2)] += w * x[i][a2] * x[i][b2];
            }
        }
    }
    let cov = pinv(&xtwx);
    Some((beta, cov))
}

fn action_mice(tab: &Table, a: &SmArgs) -> Result<Value, String> {
    let imp_count = if a.n_imp > 0 { a.n_imp } else { 5 };
    let model_type = if a.model.is_empty() { "ols" } else { a.model.as_str() };
    if a.o.is_empty() {
        return Err("mice requires 'o' (outcome/dependent variable)".to_string());
    }
    if a.p.is_empty() {
        return Err("mice requires 'p' (predictor variables)".to_string());
    }
    if !matches!(model_type, "ols" | "logit" | "poisson") {
        return Err(format!(
            "unknown model type '{}'. valid: ['ols', 'logit', 'poisson']",
            model_type
        ));
    }
    // All columns must be numeric (statsmodels MICEData on text columns
    // is out of scope for this deterministic port).
    let mut cols: Vec<(String, Vec<Option<f64>>)> = Vec::new();
    for name in &tab.headers {
        let cells = tab.col(name).unwrap();
        let mut vals = Vec::new();
        for c in &cells {
            match c {
                Cell::Num(v) if v.is_finite() => vals.push(Some(*v)),
                Cell::Num(_) => vals.push(None),
                Cell::Miss => vals.push(None),
                Cell::Txt(s) => {
                    return Err(format!(
                        "mice: non-numeric column '{}' (value '{}') not supported in the Rust port",
                        name, s
                    ));
                }
            }
        }
        cols.push((name.clone(), vals));
    }
    let n = cols.first().map(|c| c.1.len()).unwrap_or(0);
    if n == 0 {
        return Err("no data for mice".to_string());
    }
    let ncols = cols.len();
    // Rows where every column is missing are dropped by statsmodels
    // MICEData (dropna how='all'); the reported n is the original count.
    let keep_rows: Vec<usize> = (0..n)
        .filter(|&i| cols.iter().any(|(_, vals)| vals[i].is_some()))
        .collect();
    if keep_rows.is_empty() {
        return Err("no data for mice".to_string());
    }
    // Initial imputation (statsmodels _initial_imputation): the observed
    // value closest to the column mean (first on ties).
    let mut data: Vec<Vec<f64>> = vec![vec![0.0; ncols]; n];
    let mut missing: Vec<Vec<bool>> = vec![vec![false; ncols]; n];
    for (j, (_, vals)) in cols.iter().enumerate() {
        let obs: Vec<f64> = vals.iter().filter_map(|v| *v).collect();
        if obs.is_empty() {
            return Err(format!("mice: column '{}' is entirely missing", cols[j].0));
        }
        let m = mean(&obs);
        let mut best = obs[0];
        let mut best_d = f64::INFINITY;
        for &v in &obs {
            let d = (v - m).abs();
            if d < best_d {
                best_d = d;
                best = v;
            }
        }
        for i in 0..n {
            match vals[i] {
                Some(v) => data[i][j] = v,
                None => {
                    data[i][j] = best;
                    missing[i][j] = true;
                }
            }
        }
    }
    // Cycle order: variables with the fewest missing values first
    // (statsmodels _cycle_order; ties keep CSV order).
    let mut cycle: Vec<usize> = (0..ncols)
        .filter(|&j| (0..n).any(|i| missing[i][j]))
        .collect();
    cycle.sort_by_key(|&j| (0..n).filter(|&i| missing[i][j]).count());

    // Seeded RNG (SplitMix64 + Box-Muller). statsmodels MICE is
    // stochastic and unseeded, so no Rust run can match a Python run
    // value-for-value; the fixed seed makes the Rust result a
    // reproducible draw from the same PMM procedure instead.
    let mut rng = MiceRng::new(20261026);
    // One chained-equation cycle over all variables with missing data.
    let mut cycle_once = |data: &mut [Vec<f64>], rng: &mut MiceRng| {
        for &j in &cycle {
            let obs_rows: Vec<usize> = keep_rows
                .iter()
                .cloned()
                .filter(|&i| !missing[i][j])
                .collect();
            let miss_rows: Vec<usize> = keep_rows
                .iter()
                .cloned()
                .filter(|&i| missing[i][j])
                .collect();
            if miss_rows.is_empty() || obs_rows.is_empty() {
                continue;
            }
            let design = |data: &[Vec<f64>], i: usize| -> Vec<f64> {
                let mut row = vec![1.0];
                for jj in 0..ncols {
                    if jj != j {
                        row.push(data[i][jj]);
                    }
                }
                row
            };
            let xobs: Vec<Vec<f64>> =
                obs_rows.iter().map(|&i| design(data, i)).collect();
            let yobs: Vec<f64> = obs_rows.iter().map(|&i| cols[j].1[i].unwrap()).collect();
            let (beta, cov, _) = match ols_cov(&xobs, &yobs) {
                Some(t) => t,
                None => continue,
            };
            // Gaussian perturbation: params ~ MVN(beta_hat, cov_hat).
            let drawn = rng.mvn(&beta, &cov);
            let pred_obs: Vec<f64> = xobs
                .iter()
                .map(|row| row.iter().zip(drawn.iter()).map(|(a2, b2)| a2 * b2).sum())
                .collect();
            // Predictive mean matching (k_pmm = 20, statsmodels
            // impute_pmm): sort observed by predicted value; for each
            // missing case take the 20 nearest predicted values among
            // the +-20 window around the insertion point and pick one
            // at random, imputing its observed value.
            let mut order: Vec<usize> = (0..obs_rows.len()).collect();
            order.sort_by(|&a, &b| {
                pred_obs[a]
                    .partial_cmp(&pred_obs[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let pred_sorted: Vec<f64> = order.iter().map(|&o| pred_obs[o]).collect();
            let y_sorted: Vec<f64> = order.iter().map(|&o| yobs[o]).collect();
            for &i in &miss_rows {
                let row = design(data, i);
                let pm: f64 = row.iter().zip(drawn.iter()).map(|(a2, b2)| a2 * b2).sum();
                let ins = pred_sorted.partition_point(|&v| v < pm);
                let mut cand: Vec<(f64, usize)> = Vec::new();
                for d in -20i32..20 {
                    let idx = ins as i32 + d;
                    if idx < 0 || idx as usize >= pred_sorted.len() {
                        cand.push((f64::INFINITY, 0));
                    } else {
                        cand.push((
                            (pm - pred_sorted[idx as usize]).abs(),
                            idx as usize,
                        ));
                    }
                }
                let mut ord2: Vec<usize> = (0..cand.len()).collect();
                ord2.sort_by(|&a, &b| {
                    cand[a]
                        .0
                        .partial_cmp(&cand[b].0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let pick = ord2[rng.int_below(20)];
                data[i][j] = y_sorted[cand[pick].1];
            }
        }
    };
    // statsmodels MICE.fit: 10 burn-in cycles, then one cycle per
    // imputation, fitting the analysis model to each snapshot.
    for _ in 0..10 {
        cycle_once(&mut data, &mut rng);
    }
    let oi = tab
        .col_idx(&a.o)
        .ok_or_else(|| format!("endog column '{}' not in data", a.o))?;
    let mut pidx = Vec::new();
    for p in &a.p {
        pidx.push(
            tab.col_idx(p)
                .ok_or_else(|| format!("exog column '{}' not in data", p))?,
        );
    }
    let kpar = 1 + pidx.len();
    let mut betas: Vec<Vec<f64>> = Vec::new();
    let mut covs: Vec<DMatrix<f64>> = Vec::new();
    for _ in 0..imp_count {
        cycle_once(&mut data, &mut rng);
        let mut x: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for &i in &keep_rows {
            let mut row = vec![1.0];
            for &pi in &pidx {
                row.push(data[i][pi]);
            }
            x.push(row);
            y.push(data[i][oi]);
        }
        let (beta, cov) = match model_type {
            "ols" => {
                let (b, c, _) =
                    ols_cov(&x, &y).ok_or_else(|| "mice OLS fit failed".to_string())?;
                (b, c)
            }
            "logit" => {
                irls_fit(&x, &y, "logit").ok_or_else(|| "mice logit fit failed".to_string())?
            }
            _ => {
                irls_fit(&x, &y, "poisson").ok_or_else(|| "mice poisson fit failed".to_string())?
            }
        };
        betas.push(beta);
        covs.push(cov);
    }
    // Rubin's pooling (statsmodels MICE.combine): params = mean of
    // per-imputation params; W = mean within covariance; B = sample
    // covariance (ddof=1) across imputations; T = W + (1+1/m) B;
    // fmi = (1+1/m) diag(B) / diag(T).
    let m = imp_count as f64;
    let mut beta = vec![0.0; kpar];
    for b in &betas {
        for i in 0..kpar {
            beta[i] += b[i] / m;
        }
    }
    let mut wmat = DMatrix::<f64>::zeros(kpar, kpar);
    for c in &covs {
        wmat += c * (1.0 / m);
    }
    let mut bmat = DMatrix::<f64>::zeros(kpar, kpar);
    if imp_count > 1 {
        for i in 0..kpar {
            for j in 0..kpar {
                let mut s = 0.0;
                for b in &betas {
                    s += (b[i] - beta[i]) * (b[j] - beta[j]);
                }
                bmat[(i, j)] = s / (imp_count - 1) as f64;
            }
        }
    }
    let f = 1.0 + 1.0 / m;
    let tmat = &wmat + f * &bmat;
    let mut fmi_vals = vec![0.0; kpar];
    for i in 0..kpar {
        let t = tmat[(i, i)];
        fmi_vals[i] = if t > 0.0 { f * bmat[(i, i)] / t } else { 0.0 };
    }
    // Analysis-model names: patsy-style Intercept + predictors.
    let mut names = vec!["Intercept".to_string()];
    names.extend(a.p.iter().cloned());
    let mut fmap = Map::new();
    let mut fmimap = Map::new();
    for (i, name) in names.iter().enumerate() {
        let var = tmat[(i, i)];
        let se = if var.is_finite() && var >= 0.0 {
            Some(var.sqrt())
        } else {
            None
        };
        // MICE rows carry c/s/z/p only (no CI keys in the Python tool).
        let mut row = Map::new();
        row.insert("c".to_string(), f6(beta[i]));
        match se {
            Some(s) if s > 0.0 => {
                let z = beta[i] / s;
                row.insert("s".to_string(), f6(s));
                row.insert("z".to_string(), f6(z));
                // MICEResults p-values are normal (df_resid unset).
                row.insert("p".to_string(), f6(2.0 * norm_sf(z.abs())));
            }
            Some(s) => {
                row.insert("s".to_string(), f6(s));
                row.insert("z".to_string(), Value::Null);
                row.insert("p".to_string(), Value::Null);
            }
            None => {
                row.insert("s".to_string(), Value::Null);
                row.insert("z".to_string(), Value::Null);
                row.insert("p".to_string(), Value::Null);
            }
        }
        fmap.insert(name.clone(), Value::Object(row));
        fmimap.insert(name.clone(), f6(fmi_vals[i]));
    }
    let mut out = Map::new();
    out.insert("imp".to_string(), json!(imp_count));
    out.insert("n".to_string(), json!(n));
    out.insert("f".to_string(), Value::Object(fmap));
    out.insert("fmi".to_string(), Value::Object(fmimap));
    Ok(Value::Object(out))
}

/// SplitMix64 + Box-Muller RNG for the seeded MICE chain.
struct MiceRng {
    state: u64,
    spare: Option<f64>,
}

impl MiceRng {
    fn new(seed: u64) -> Self {
        MiceRng {
            state: seed,
            spare: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * t.sin());
        r * t.cos()
    }
    fn int_below(&mut self, n: usize) -> usize {
        ((self.uniform() * n as f64) as usize).min(n - 1)
    }
    /// Multivariate normal draw via symmetric eigendecomposition
    /// (tolerates PSD covariance from pinv).
    fn mvn(&mut self, mu: &[f64], cov: &DMatrix<f64>) -> Vec<f64> {
        let k = mu.len();
        let eig = nalgebra::SymmetricEigen::new(cov.clone());
        let z: Vec<f64> = (0..k).map(|_| self.normal()).collect();
        let mut out = mu.to_vec();
        for i in 0..k {
            let lam = eig.eigenvalues[i].max(0.0);
            let s = lam.sqrt();
            for r in 0..k {
                out[r] += eig.eigenvectors[(r, i)] * s * z[i];
            }
        }
        out
    }
}

// ------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_csv;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol || (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn gee_gaussian_independence_equals_ols() {
        // Independence Gaussian GEE solves the OLS normal equations.
        let tab = parse_csv("g,x,y\n1,0.5,2.1\n1,-0.3,0.9\n2,1.2,3.4\n2,0.1,1.8\n3,-1.0,-0.5\n3,0.7,2.6").unwrap();
        let a = SmArgs {
            o: "y".into(),
            p: vec!["x".into()],
            g: "g".into(),
            fam: "gaussian".into(),
            cov: "independence".into(),
            ..Default::default()
        };
        let v = dispatch_sm("gee", &tab, &a).unwrap();
        let f = v["f"].as_object().unwrap();
        let cx = f["x"]["c"].as_f64().unwrap();
        // OLS by hand (numpy lstsq): slope 1.743421.
        assert!(close(cx, 1.743421, 0.01), "slope {}", cx);
        assert!(v["cnvg"].as_bool().unwrap());
    }

    #[test]
    fn anova_rm_known_f() {
        // 10 subjects x 3 conditions; Python AnovaRM F = 160.428
        // (verification dataset, see report). Recomputed here from a
        // smaller hand-checkable case instead: 4 subjects, 2 levels,
        // condition means differ by 4, subject effects cancel.
        let csv = "s,c,y\n1,1,10\n1,2,14\n2,1,11\n2,2,15\n3,1,9\n3,2,13\n4,1,12\n4,2,16";
        let tab = parse_csv(csv).unwrap();
        let a = SmArgs {
            o: "y".into(),
            g: "s".into(),
            w: vec!["c".into()],
            ..Default::default()
        };
        let v = dispatch_sm("anova_rm", &tab, &a).unwrap();
        // Perfectly consistent differences (all +4): SS_err = 0 -> F inf
        // -> null, p = 0.
        assert_eq!(v["anova"]["c"]["ndf"], 1);
        assert_eq!(v["anova"]["c"]["ddf"], 3);
        assert_eq!(v["anova"]["c"]["p"], 0.0);
    }

    #[test]
    fn mlm_balanced_matches_profile() {
        // Random-intercept data with strong group effects; the profile
        // REML fit must recover a positive group variance and a slope
        // near 2 (data generated with slope 2).
        let mut rows = String::from("id,x,y\n");
        let mut seed = 123456789u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };
        for g in 1..=12 {
            let re = rnd() * 2.0;
            for _ in 0..4 {
                let x = rnd();
                let y = 3.0 + 2.0 * x + re + rnd() * 0.5;
                rows.push_str(&format!("{},{:.3},{:.3}\n", g, x, y));
            }
        }
        let tab = parse_csv(&rows).unwrap();
        let a = SmArgs {
            o: "y".into(),
            p: vec!["x".into()],
            g: "id".into(),
            ..Default::default()
        };
        let v = dispatch_sm("mlm", &tab, &a).unwrap();
        let slope = v["f"]["x"]["c"].as_f64().unwrap();
        assert!((slope - 2.0).abs() < 0.35, "slope {}", slope);
        assert!(v["v"].as_f64().unwrap() > 0.0);
        assert_eq!(v["r"]["n"], 12);
    }

    #[test]
    fn gee_ord_identifiable_quantities_match_python() {
        // 30 groups x 2, 3-level ordinal (verification dataset, seed
        // 1000). Python OrdinalGEE independence: slope 1.545350,
        // ll -58.948603, x SE 0.276463. The design is rank-deficient
        // by construction (const = sum of threshold dummies), so only
        // the identifiable combinations are asserted: slope, ll, and
        // threshold+const sums (Python: 0.800464, -0.773071). The Rust
        // fit is the min-norm (GLM) resolution of the same system.
        let csv = "id,x,y\n1,0.25,0\n1,0.76,2\n2,-1.21,0\n2,0.52,1\n3,-0.45,0\n3,-0.67,1\n4,-0.36,0\n4,-0.28,1\n5,1.14,2\n5,-1.54,0\n6,-1.81,1\n6,1.87,2\n7,-1.79,0\n7,2.69,2\n8,0.69,2\n8,0.46,2\n9,0.72,2\n9,1.95,2\n10,-0.44,2\n10,-0.63,0\n11,-0.23,2\n11,-0.39,0\n12,-1.55,0\n12,0.57,1\n13,-1.11,0\n13,-0.17,2\n14,0.15,1\n14,0.91,2\n15,0.74,2\n15,0.59,0\n16,-0.47,1\n16,1.03,1\n17,-0.9,0\n17,-0.48,1\n18,0.11,0\n18,0.61,2\n19,0.07,0\n19,0.68,1\n20,1.33,1\n20,-0.76,1\n21,-1.24,0\n21,0.58,2\n22,1.07,2\n22,-1.78,1\n23,2.36,2\n23,1.9,2\n24,0.28,2\n24,0.34,1\n25,0.53,2\n25,0.23,2\n26,-0.21,0\n26,0.92,2\n27,1.19,1\n27,0.57,2\n28,-0.14,0\n28,-1.06,0\n29,0.04,2\n29,0.17,0\n30,-1.06,1\n30,0.4,0";
        let tab = parse_csv(csv).unwrap();
        let a = SmArgs {
            o: "y".into(),
            p: vec!["x".into()],
            g: "id".into(),
            cov: "independence".into(),
            ..Default::default()
        };
        let v = dispatch_sm("gee_ord", &tab, &a).unwrap();
        let f = v["f"].as_object().unwrap();
        let t1 = f["I(y>0.0)"]["c"].as_f64().unwrap();
        let t2 = f["I(y>1.0)"]["c"].as_f64().unwrap();
        let c0 = f["const"]["c"].as_f64().unwrap();
        assert!(close(f["x"]["c"].as_f64().unwrap(), 1.545350, 1e-3));
        assert!(close(t1 + c0, 0.800464, 1e-3), "t1+c {}", t1 + c0);
        assert!(close(t2 + c0, -0.773071, 1e-3), "t2+c {}", t2 + c0);
        assert!(close(v["ll"].as_f64().unwrap(), -58.948603, 1e-3));
        assert!(close(f["x"]["s"].as_f64().unwrap(), 0.276463, 1e-3));
    }

    #[test]
    fn gee_nom_matches_python_exactly() {
        // First 10 groups of the same dataset, 3-level nominal.
        // Python NominalGEE independence params (in order
        // const[0.0], x[0.0], const[1.0], x[1.0]).
        let csv = "id,x,y\n1,0.25,0\n1,0.76,2\n2,-1.21,0\n2,0.52,1\n3,-0.45,0\n3,-0.67,1\n4,-0.36,0\n4,-0.28,1\n5,1.14,2\n5,-1.54,0\n6,-1.81,1\n6,1.87,2\n7,-1.79,0\n7,2.69,2\n8,0.69,2\n8,0.46,2\n9,0.72,2\n9,1.95,2\n10,-0.44,2\n10,-0.63,0";
        let tab = parse_csv(csv).unwrap();
        let a = SmArgs {
            o: "y".into(),
            p: vec!["x".into()],
            g: "id".into(),
            cov: "independence".into(),
            ..Default::default()
        };
        let v = dispatch_sm("gee_nom", &tab, &a).unwrap();
        let f = v["f"].as_object().unwrap();
        for (name, want_c, want_s) in [
            ("const[0.0]", 0.09644, 0.412819),
            ("x[0.0]", -3.254974, 0.697221),
            ("const[1.0]", -0.153493, 0.633243),
            ("x[1.0]", -2.801115, 0.854643),
        ] {
            assert!(
                close(f[name]["c"].as_f64().unwrap(), want_c, 1e-3),
                "{} c {}",
                name,
                f[name]["c"]
            );
            assert!(
                close(f[name]["s"].as_f64().unwrap(), want_s, 1e-3),
                "{} s {}",
                name,
                f[name]["s"]
            );
        }
    }

    #[test]
    fn mice_complete_data_equals_ols() {
        let tab = parse_csv("x1,x2,y\n0.5,1.0,3.1\n-0.3,0.2,1.4\n1.2,-0.5,2.9\n0.1,0.8,2.2\n-1.0,-1.0,-0.4\n0.7,0.3,2.8").unwrap();
        let a = SmArgs {
            o: "y".into(),
            p: vec!["x1".into(), "x2".into()],
            n_imp: 5,
            model: "ols".into(),
            ..Default::default()
        };
        let v = dispatch_sm("mice", &tab, &a).unwrap();
        assert_eq!(v["imp"], 5);
        assert_eq!(v["n"], 6);
        // With no missing values the pooled fit is the OLS fit; the
        // x1 coefficient is positive and near the OLS value ~1.2.
        let c = v["f"]["x1"]["c"].as_f64().unwrap();
        assert!(c > 0.5 && c < 2.0, "x1 {}", c);
        assert_eq!(v["fmi"]["x1"], 0.0);
    }
}
