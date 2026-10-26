// NEW SEM actions for the Rust port: sem_fit, sem_inspect.
// Faithful re-implementation of ~/workspace/.rs-orig-stat/tools/sem_tool.py
// (semopy >= 2.3) from first principles: lavaan-style syntax
// (latent =~ indicators, regressions ~, covariances ~~), ML estimation
// of the model-implied covariance (semopy's MLW objective), expected
// Fisher-information SEs, and semopy's calc_stats fit indices.
//
// MERGE NOTE (for the parent): add to src/lib.rs:
//     pub mod actions_sem_new;
// and in src/main.rs, in the "sem" match arm, replace the Not-yet error
// with:
//     "sem" => {
//         let desc = params.get("desc").and_then(|v| v.as_str()).unwrap_or("").to_string();
//         let mode = params.get("mode").and_then(|v| v.as_str()).unwrap_or("raw").to_string();
//         let tab = match parse_csv(csv_text) {
//             Ok(t) => t, Err(e) => return err(&e),
//         };
//         match hermes_statistical_tools_rs::actions_sem_new::dispatch_sem(&action, &tab, &desc, &mode) {
//             Ok(v) => v, Err(e) => err(&e),
//         }
//     }
// (desc/mode travel in --params JSON, as in the Python tool's arguments.)
//
// Behavioural notes / deviations from the Python tool:
//  * sem_inspect in Python reads an in-process cache: it only works in
//    the same session as a previous sem_fit with the same desc+data.
//    A CLI invocation is a fresh process, so this port REFITS from the
//    supplied CSV when data is present (identical output, since the
//    fit is deterministic) and errors without data, with Python's
//    message. This is the only sane CLI semantics; flag it to the user
//    if strict cache semantics are ever required.
//  * sem_inspect mode "std_est" returns the same "c" values as "raw":
//    the Python tool reads only semopy's "Estimate" column, and
//    semopy's std_est output keeps raw values in "Estimate" (the
//    standardised values live in a separate "Est. Std" column the tool
//    never reads). This port reproduces the tool's output, not
//    semopy's full table.
//  * semopy's "ll" fit value is NOT a log-likelihood: calc_stats'
//    LogLik is obj_mlw, the MLW discrepancy F itself, so AIC =
//    2*(k - F) and BIC = log(n)*k - 2*F, and chi2 = n*F. Reproduced.
//  * First loading of each latent is fixed to 1.0 (no SE/z/p keys),
//    exactly as semopy reports it.
//  * Auto-parameters (semopy rules, verified against semopy 2.3.11):
//    variances for every latent and every non-exogenous observed
//    variable; covariances between every pair of exogenous latents
//    (stored as later~~earlier); exogenous observed predictors get NO
//    variance/covariance parameters, and their Psi block is fixed to
//    the sample covariance among exogenous observed variables.
//  * Output row order (verified against semopy inspect): "~" rows =
//    latent-lhs rows in description order, then observed-lhs rows in
//    description order (measurement rows have lv=indicator, rv=latent,
//    op="~"); "~~" rows grouped by lval in [latents in description
//    order, then observed in CSV order], each group = explicit
//    covariances (description order), then auto covariances, then the
//    variance.
use crate::{norm_sf, Cell, Table};
use nalgebra::{DMatrix, DVector};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

fn f6(v: f64) -> Value {
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
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

// ------------------------------------------------------------ parsing

#[derive(Clone, Debug)]
struct Param {
    kind: ParamKind,
    /// For B params: (row, col) in variable space. For Psi params:
    /// (i, j) with i <= j in the canonical storage below.
    row: usize,
    col: usize,
    fixed: Option<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum ParamKind {
    B,       // regression / loading coefficient
    PsiVar,  // variance (diagonal of Psi)
    PsiCov,  // covariance (off-diagonal of Psi)
}

#[derive(Clone, Debug)]
struct OutRow {
    lv: String,
    op: &'static str,
    rv: String,
    /// Index into the free-parameter vector, or fixed value.
    val: RowVal,
}
#[derive(Clone, Debug)]
enum RowVal {
    Free(usize),
    Fixed(f64),
}

pub struct SemModel {
    /// Variable order for B/Psi: latents (desc order) then observed (CSV order).
    vars: Vec<String>,
    latents: Vec<String>,
    observed: Vec<String>, // observed vars used by the model, CSV order
    is_latent: Vec<bool>,
    is_exog_obs: Vec<bool>,
    /// Free parameters (in creation order) and fixed values.
    b_free: Vec<Param>,
    psi_free: Vec<Param>,
    /// Output rows in semopy inspect order.
    rows_reg: Vec<OutRow>, // op "~"
    rows_cov: Vec<OutRow>, // op "~~"
    /// Map (row, col) -> free index for B and Psi, for building matrices.
    b_map: BTreeMap<(usize, usize), RowVal>,
    psi_map: BTreeMap<(usize, usize), RowVal>,
}

fn parse_term(t: &str) -> Option<(String, Option<f64>)> {
    let t = t.trim();
    if t.is_empty() || t == "1" {
        return None; // intercepts / means are not modelled
    }
    if let Some((a, b)) = t.split_once('*') {
        let a = a.trim();
        let b = b.trim();
        if let Ok(v) = a.parse::<f64>() {
            return Some((b.to_string(), Some(v)));
        }
        // A label, not a number: semopy labels are ignored here.
        return Some((b.to_string(), None));
    }
    Some((t.to_string(), None))
}

struct RawStmt {
    lhs: String,
    op: String,
    terms: Vec<(String, Option<f64>)>,
}

fn parse_desc(desc: &str) -> Result<Vec<RawStmt>, String> {
    let mut out = Vec::new();
    for line in desc.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Operators in order of length: =~ first, then ~~, then ~.
        let (lhs, op, rhs) = if let Some(p) = line.find("=~") {
            (&line[..p], "=~", &line[p + 2..])
        } else if let Some(p) = line.find("~~") {
            (&line[..p], "~~", &line[p + 2..])
        } else if let Some(p) = line.find('~') {
            (&line[..p], "~", &line[p + 1..])
        } else {
            return Err(format!("cannot parse model statement '{}'", line));
        };
        let lhs = lhs.trim().to_string();
        let mut terms = Vec::new();
        for t in rhs.split('+') {
            if let Some(term) = parse_term(t) {
                terms.push(term);
            }
        }
        if lhs.is_empty() || terms.is_empty() {
            return Err(format!("cannot parse model statement '{}'", line));
        }
        out.push(RawStmt { lhs, op: op.to_string(), terms });
    }
    if out.is_empty() {
        return Err("empty model description".to_string());
    }
    Ok(out)
}

fn build_model(desc: &str, csv_cols: &[String]) -> Result<SemModel, String> {
    let stmts = parse_desc(desc)?;
    let colset: BTreeSet<&str> = csv_cols.iter().map(|s| s.as_str()).collect();
    // Latents: lhs of =~, in first-appearance order. Any other name
    // that is not a CSV column and appears in the model is also latent.
    let mut latents: Vec<String> = Vec::new();
    for s in &stmts {
        if s.op == "=~" && !latents.contains(&s.lhs) {
            latents.push(s.lhs.clone());
        }
    }
    for s in &stmts {
        for name in std::iter::once(&s.lhs).chain(s.terms.iter().map(|t| &t.0)) {
            if !colset.contains(name.as_str()) && !latents.contains(name) {
                latents.push(name.clone());
            }
        }
    }
    // Observed vars used by the model, in CSV order.
    let mut observed: Vec<String> = Vec::new();
    for c in csv_cols {
        let used = stmts.iter().any(|s| {
            s.lhs == *c || s.terms.iter().any(|t| &t.0 == c)
        });
        if used {
            observed.push(c.clone());
        }
    }
    if observed.is_empty() {
        return Err("model description references no data columns".to_string());
    }
    let mut vars: Vec<String> = latents.clone();
    vars.extend(observed.iter().cloned());
    let idx = |name: &str| -> Option<usize> { vars.iter().position(|v| v == name) };
    let is_latent: Vec<bool> = vars.iter().map(|v| latents.contains(v)).collect();
    // lhs sets: which variables are lhs of ~ or =~ (endogenous).
    let mut lhs_of_reg: BTreeSet<String> = BTreeSet::new();
    for s in &stmts {
        if s.op == "~" || s.op == "=~" {
            lhs_of_reg.insert(s.lhs.clone());
        }
    }
    // Exogenous observed: rhs of ~, never lhs of ~ or =~.
    let mut rhs_obs: BTreeSet<String> = BTreeSet::new();
    for s in &stmts {
        if s.op == "~" {
            for (t, _) in &s.terms {
                if colset.contains(t.as_str()) {
                    rhs_obs.insert(t.clone());
                }
            }
        }
    }
    let is_exog_obs: Vec<bool> = vars
        .iter()
        .map(|v| {
            !is_latent[vars.iter().position(|x| x == v).unwrap()]
                && rhs_obs.contains(v)
                && !lhs_of_reg.contains(v)
        })
        .collect();
    // ---- Collect parameters.
    let mut b_map: BTreeMap<(usize, usize), RowVal> = BTreeMap::new();
    let mut psi_map: BTreeMap<(usize, usize), RowVal> = BTreeMap::new();
    let mut b_free: Vec<Param> = Vec::new();
    let mut psi_free: Vec<Param> = Vec::new();
    let mut free_counter = 0usize; // shared free index space is NOT used;
    // B and Psi free params are indexed separately below.
    let _ = &mut free_counter;
    let mut b_next = 0usize;
    let mut psi_next = 0usize;
    let mut reg_rows_raw: Vec<(String, String, RowVal, bool)> = Vec::new(); // (lv, rv, val, lv_is_latent)
    let mut cov_explicit: Vec<(String, String, RowVal)> = Vec::new();
    let mut var_explicit: BTreeMap<String, RowVal> = BTreeMap::new();
    let mut first_loading_done: BTreeSet<String> = BTreeSet::new();
    for s in &stmts {
        match s.op.as_str() {
            "=~" => {
                let lat = idx(&s.lhs).ok_or_else(|| format!("unknown variable '{}'", s.lhs))?;
                for (ind, fixed) in &s.terms {
                    let ind_i = idx(ind).ok_or_else(|| format!("unknown variable '{}'", ind))?;
                    let is_first = !first_loading_done.contains(&s.lhs);
                    first_loading_done.insert(s.lhs.clone());
                    let val = if let Some(v) = fixed {
                        RowVal::Fixed(*v)
                    } else if is_first {
                        RowVal::Fixed(1.0)
                    } else {
                        let fi = b_next;
                        b_next += 1;
                        b_free.push(Param { kind: ParamKind::B, row: ind_i, col: lat, fixed: None });
                        RowVal::Free(fi)
                    };
                    b_map.insert((ind_i, lat), val.clone());
                    // Output row is flipped: indicator ~ latent.
                    reg_rows_raw.push((ind.clone(), s.lhs.clone(), val, false));
                }
            }
            "~" => {
                let lhs_i = idx(&s.lhs).ok_or_else(|| format!("unknown variable '{}'", s.lhs))?;
                let lhs_lat = is_latent[lhs_i];
                for (rhs, fixed) in &s.terms {
                    let rhs_i = idx(rhs).ok_or_else(|| format!("unknown variable '{}'", rhs))?;
                    let val = if let Some(v) = fixed {
                        RowVal::Fixed(*v)
                    } else {
                        let fi = b_next;
                        b_next += 1;
                        b_free.push(Param { kind: ParamKind::B, row: lhs_i, col: rhs_i, fixed: None });
                        RowVal::Free(fi)
                    };
                    b_map.insert((lhs_i, rhs_i), val.clone());
                    reg_rows_raw.push((s.lhs.clone(), rhs.clone(), val, lhs_lat));
                }
            }
            "~~" => {
                for (rhs, fixed) in &s.terms {
                    // Note: semopy ~~ statements in the tool's usage
                    // have a single term; a ~~ b + c is not standard
                    // and is treated as separate pairs with the lhs.
                    let a_i = idx(&s.lhs).ok_or_else(|| format!("unknown variable '{}'", s.lhs))?;
                    let b_i = idx(rhs).ok_or_else(|| format!("unknown variable '{}'", rhs))?;
                    if s.lhs == *rhs {
                        let val = if let Some(v) = fixed {
                            RowVal::Fixed(*v)
                        } else {
                            let fi = psi_next;
                            psi_next += 1;
                            psi_free.push(Param { kind: ParamKind::PsiVar, row: a_i, col: a_i, fixed: None });
                            RowVal::Free(fi)
                        };
                        psi_map.insert((a_i, a_i), val.clone());
                        var_explicit.insert(s.lhs.clone(), val);
                    } else {
                        let (r, c) = if a_i <= b_i { (a_i, b_i) } else { (b_i, a_i) };
                        let val = if let Some(v) = fixed {
                            RowVal::Fixed(*v)
                        } else {
                            let fi = psi_next;
                            psi_next += 1;
                            psi_free.push(Param { kind: ParamKind::PsiCov, row: r, col: c, fixed: None });
                            RowVal::Free(fi)
                        };
                        psi_map.insert((r, c), val.clone());
                        cov_explicit.push((s.lhs.clone(), rhs.clone(), val));
                    }
                }
            }
            _ => {}
        }
    }
    // ---- Auto parameters.
    // Variances: every latent + every non-exogenous observed variable.
    for (vi, vname) in vars.iter().enumerate() {
        if is_latent[vi] || !is_exog_obs[vi] {
            if !psi_map.contains_key(&(vi, vi)) {
                let fi = psi_next;
                psi_next += 1;
                psi_free.push(Param { kind: ParamKind::PsiVar, row: vi, col: vi, fixed: None });
                psi_map.insert((vi, vi), RowVal::Free(fi));
            }
        }
    }
    // Covariances between exogenous latents (never lhs of ~).
    let mut exog_latents: Vec<String> = Vec::new();
    for lat in &latents {
        let is_lhs = stmts.iter().any(|s| s.op == "~" && s.lhs == *lat);
        if !is_lhs {
            exog_latents.push(lat.clone());
        }
    }
    let mut auto_covs: Vec<(String, String, RowVal)> = Vec::new();
    for ii in 0..exog_latents.len() {
        for jj in 0..ii {
            let a = &exog_latents[jj];
            let b = &exog_latents[ii];
            // Skip if the user already specified this pair either way.
            let already = cov_explicit.iter().any(|(x, y, _)| {
                (x == a && y == b) || (x == b && y == a)
            });
            if already {
                continue;
            }
            let ai = idx(a).unwrap();
            let bi = idx(b).unwrap();
            let (r, c) = if ai <= bi { (ai, bi) } else { (bi, ai) };
            if psi_map.contains_key(&(r, c)) {
                continue;
            }
            let fi = psi_next;
            psi_next += 1;
            psi_free.push(Param { kind: ParamKind::PsiCov, row: r, col: c, fixed: None });
            let val = RowVal::Free(fi);
            psi_map.insert((r, c), val.clone());
            // Stored as later~~earlier, as semopy reports it.
            auto_covs.push((b.clone(), a.clone(), val));
        }
    }
    // ---- Output row order.
    // "~": latent-lhs rows first (desc order), then observed-lhs rows.
    let mut rows_reg: Vec<OutRow> = Vec::new();
    for latent_first in [true, false] {
        for (lv, rv, val, lv_lat) in &reg_rows_raw {
            if *lv_lat == latent_first {
                rows_reg.push(OutRow { lv: lv.clone(), op: "~", rv: rv.clone(), val: val.clone() });
            }
        }
    }
    // "~~": grouped by lval in [latents desc order, observed CSV order];
    // within a group: explicit covariances, auto covariances, variance.
    let mut v_order: Vec<String> = latents.clone();
    for o in &observed {
        if !v_order.contains(o) {
            v_order.push(o.clone());
        }
    }
    let mut rows_cov: Vec<OutRow> = Vec::new();
    for v in &v_order {
        for (lv, rv, val) in &cov_explicit {
            if lv == v {
                rows_cov.push(OutRow { lv: lv.clone(), op: "~~", rv: rv.clone(), val: val.clone() });
            }
        }
        for (lv, rv, val) in &auto_covs {
            if lv == v {
                rows_cov.push(OutRow { lv: lv.clone(), op: "~~", rv: rv.clone(), val: val.clone() });
            }
        }
        if let Some(val) = psi_map.get(&(idx(v).unwrap(), idx(v).unwrap())) {
            // Only emit the variance if it is a parameter (explicit or
            // auto); exogenous observed variables have none.
            let vi = idx(v).unwrap();
            if is_latent[vi] || !is_exog_obs[vi] {
                rows_cov.push(OutRow {
                    lv: v.clone(),
                    op: "~~",
                    rv: v.clone(),
                    val: val.clone(),
                });
            }
        }
    }
    let _ = var_explicit;
    let _ = b_free.len();
    Ok(SemModel {
        vars,
        latents,
        observed,
        is_latent,
        is_exog_obs,
        b_free,
        psi_free,
        rows_reg,
        rows_cov,
        b_map,
        psi_map,
    })
}

// ------------------------------------------------------------ fitting

struct FitData {
    /// Sample covariance (ddof=0, ML divisor n, as semopy uses) over
    /// observed vars, in model order.
    s: DMatrix<f64>,
    /// Indices of observed vars in variable space.
    obs_idx: Vec<usize>,
    /// Sample covariance block among exogenous observed vars (for the
    /// fixed Psi block), indexed by variable-space index.
    s_map: BTreeMap<(usize, usize), f64>,
    n: usize,
}

fn sample_cov(tab: &Table, observed: &[String]) -> Result<FitData, String> {
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for name in observed {
        let cells = tab.col(name).ok_or_else(|| format!("column '{}' not in data", name))?;
        let mut vals = Vec::new();
        let mut ok_rows: Vec<bool> = Vec::new();
        // Rows must be complete over all observed vars (semopy drops
        // incomplete rows via its data handling for MLW).
        let _ = &mut ok_rows;
        for c in &cells {
            match c {
                Cell::Num(v) if v.is_finite() => vals.push(*v),
                _ => vals.push(f64::NAN),
            }
        }
        cols.push(vals);
    }
    let nrows = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut keep: Vec<usize> = Vec::new();
    for i in 0..nrows {
        if cols.iter().all(|c| c[i].is_finite()) {
            keep.push(i);
        }
    }
    if keep.len() < 2 {
        return Err("SEM fit requires at least two complete observations".to_string());
    }
    let p = observed.len();
    let mut means = vec![0.0; p];
    for j in 0..p {
        means[j] = mean(&keep.iter().map(|&i| cols[j][i]).collect::<Vec<_>>());
    }
    let mut s = DMatrix::<f64>::zeros(p, p);
    for a2 in 0..p {
        for b2 in 0..p {
            let mut sum = 0.0;
            for &i in &keep {
                sum += (cols[a2][i] - means[a2]) * (cols[b2][i] - means[b2]);
            }
            // Divide by n (ML), not n-1: semopy's MLW sample covariance
            // uses ddof=0 (verified: variance params were systematically
            // n/(n-1) too large with ddof=1, loadings inside tolerance).
            s[(a2, b2)] = sum / keep.len() as f64;
        }
    }
    Ok(FitData {
        s,
        obs_idx: Vec::new(), // filled by caller
        s_map: BTreeMap::new(),
        n: keep.len(),
    })
}

struct SemFit {
    theta_b: Vec<f64>, // free B values, in b_free order
    theta_p: Vec<f64>, // free Psi values, in psi_free order
    fun: f64,
    sigma_obs: DMatrix<f64>,
}

/// Build B and Psi from free-parameter vectors, then Sigma_obs.
fn build_sigma(
    model: &SemModel,
    fit: &FitData,
    theta_b: &[f64],
    theta_p: &[f64],
) -> Option<DMatrix<f64>> {
    let nv = model.vars.len();
    let mut b = DMatrix::<f64>::zeros(nv, nv);
    for (fi, prm) in model.b_free.iter().enumerate() {
        b[(prm.row, prm.col)] = theta_b[fi];
    }
    for ((r, c), val) in &model.b_map {
        if let RowVal::Fixed(v) = val {
            b[(*r, *c)] = *v;
        }
    }
    let mut psi = DMatrix::<f64>::zeros(nv, nv);
    for (fi, prm) in model.psi_free.iter().enumerate() {
        psi[(prm.row, prm.col)] = theta_p[fi];
        psi[(prm.col, prm.row)] = theta_p[fi];
    }
    for ((r, c), val) in &model.psi_map {
        if let RowVal::Fixed(v) = val {
            psi[(*r, *c)] = *v;
            psi[(*c, *r)] = *v;
        }
    }
    // Fixed Psi block for exogenous observed variables: sample
    // covariance among them (semopy treats them as fixed).
    for (vi, _) in model.vars.iter().enumerate() {
        if model.is_exog_obs[vi] {
            for (vj, _) in model.vars.iter().enumerate() {
                if model.is_exog_obs[vj] {
                    if let Some(v) = fit.s_map.get(&(vi, vj)) {
                        psi[(vi, vj)] = *v;
                    }
                }
            }
        }
    }
    let mut amat = DMatrix::<f64>::identity(nv, nv) - &b;
    let _ = &mut amat;
    let ainv = amat.clone().lu().solve(&DMatrix::<f64>::identity(nv, nv))?;
    let sigma_full = &ainv * &psi * &ainv.transpose();
    let p = model.observed.len();
    let mut so = DMatrix::<f64>::zeros(p, p);
    for (oi, &vi) in fit.obs_idx.iter().enumerate() {
        for (oj, &vj) in fit.obs_idx.iter().enumerate() {
            so[(oi, oj)] = sigma_full[(vi, vj)];
        }
    }
    Some(so)
}

fn mlw_objective(
    model: &SemModel,
    fit: &FitData,
    theta_b: &[f64],
    theta_p: &[f64],
) -> Option<(f64, DMatrix<f64>)> {
    let sigma = build_sigma(model, fit, theta_b, theta_p)?;
    let p = model.observed.len() as f64;
    // log|Sigma| + tr(S Sigma^{-1}) - log|S| - p
    let chol = sigma.clone().cholesky()?;
    let mut logdet_s = 0.0;
    for i in 0..sigma.nrows() {
        let d = chol.l()[(i, i)];
        if d <= 0.0 {
            return None;
        }
        logdet_s += d.ln();
    }
    logdet_s *= 2.0;
    let s_chol = fit.s.clone().cholesky()?;
    let mut logdet_samp = 0.0;
    for i in 0..fit.s.nrows() {
        logdet_samp += s_chol.l()[(i, i)].ln();
    }
    logdet_samp *= 2.0;
    let sig_inv = chol.inverse();
    let tr = (&fit.s * &sig_inv).trace();
    let f = logdet_s + tr - logdet_samp - p;
    if !f.is_finite() {
        return None;
    }
    Some((f, sigma))
}

/// BFGS with central-difference gradients and Armijo backtracking.
fn bfgs(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    maxiter: usize,
) -> (Vec<f64>, f64, bool) {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    if !fx.is_finite() {
        return (x, fx, false);
    }
    let grad = |x: &[f64]| -> Vec<f64> {
        let mut g = vec![0.0; n];
        for i in 0..n {
            let h = 1e-6 * (1.0 + x[i].abs());
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += h;
            xm[i] -= h;
            let fp = f(&xp);
            let fm = f(&xm);
            if fp.is_finite() && fm.is_finite() {
                g[i] = (fp - fm) / (2.0 * h);
            } else if fp.is_finite() {
                g[i] = (fp - f(x)) / h;
            } else if fm.is_finite() {
                g[i] = (f(x) - fm) / h;
            } else {
                g[i] = 0.0;
            }
        }
        g
    };
    let mut g = grad(&x);
    let mut hinv = DMatrix::<f64>::identity(n, n);
    let gnorm = |g: &[f64]| g.iter().map(|v| v * v).sum::<f64>().sqrt();
    for _ in 0..maxiter {
        if gnorm(&g) < 1e-7 {
            return (x, fx, true);
        }
        let gv = DVector::<f64>::from_column_slice(&g);
        let mut dir = -(&hinv * &gv);
        // Ensure a descent direction.
        let gd: f64 = g.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
        if !gd.is_finite() || gd >= 0.0 {
            hinv = DMatrix::<f64>::identity(n, n);
            dir = -gv.clone();
        }
        let gd: f64 = g.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..60 {
            let xn: Vec<f64> = x.iter().zip(dir.iter()).map(|(a, b)| a + step * b).collect();
            let fnx = f(&xn);
            if fnx.is_finite() && fnx <= fx + 1e-4 * step * gd {
                let gn = grad(&xn);
                let s: Vec<f64> = dir.iter().map(|d| step * d).collect();
                let yv: Vec<f64> = gn.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
                let sy: f64 = s.iter().zip(yv.iter()).map(|(a, b)| a * b).sum();
                if sy > 1e-12 {
                    // BFGS update of the inverse Hessian.
                    let sv = DVector::<f64>::from_column_slice(&s);
                    let yvec = DVector::<f64>::from_column_slice(&yv);
                    let rho = 1.0 / sy;
                    let i_mat = DMatrix::<f64>::identity(n, n);
                    let lhs = &i_mat - rho * &sv * yvec.transpose();
                    let rhs = &i_mat - rho * yvec * sv.transpose();
                    hinv = &lhs * &hinv * &rhs + rho * &sv * sv.transpose();
                }
                x = xn;
                fx = fnx;
                g = gn;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    (x, fx, gnorm(&g) < 1e-5)
}

fn fit_sem(model: &SemModel, fit: &FitData) -> Result<SemFit, String> {
    let nb = model.b_free.len();
    let np = model.psi_free.len();
    // Starting values: loadings 0.9, structural 0.0, variances from
    // the sample variances (observed) / 1.0 (latent), covariances 0.
    let obs_var = |vi: usize| -> f64 {
        // Sample variance of observed var vi, else 1.0.
        if let Some(oi) = fit.obs_idx.iter().position(|&x| x == vi) {
            fit.s[(oi, oi)].max(0.05)
        } else {
            1.0
        }
    };
    let start_b: Vec<f64> = model
        .b_free
        .iter()
        .map(|p| {
            // Loading (row observed, col latent) vs structural.
            if model.is_latent[p.col] && !model.is_latent[p.row] {
                0.9
            } else {
                0.0
            }
        })
        .collect();
    let start_p: Vec<f64> = model
        .psi_free
        .iter()
        .map(|p| match p.kind {
            ParamKind::PsiVar => {
                if model.is_latent[p.row] {
                    1.0
                } else {
                    0.5 * obs_var(p.row)
                }
            }
            ParamKind::PsiCov => 0.0,
            ParamKind::B => 0.0,
        })
        .collect();
    // Variance parameters are optimised as u with var = u^2 (semopy
    // bounds variances >= 0 via SLSQP; unconstrained BFGS reached a
    // negative residual variance, -0.27, where semopy stops at 0.0).
    // Covariances and B parameters stay raw. SEs are computed in
    // theta-space afterwards, so this changes only the search path.
    let is_var: Vec<bool> = {
        let mut v = vec![false; nb];
        v.extend(model.psi_free.iter().map(|p| p.kind == ParamKind::PsiVar));
        v
    };
    let to_theta = |u: &[f64]| -> Vec<f64> {
        u.iter()
            .zip(is_var.iter())
            .map(|(ui, iv)| if *iv { ui * ui } else { *ui })
            .collect()
    };
    let obj = |u: &[f64]| -> f64 {
        let theta = to_theta(u);
        let (tb, tp) = theta.split_at(nb);
        match mlw_objective(model, fit, tb, tp) {
            Some((f, _)) => f,
            None => 1e100,
        }
    };
    let to_u = |theta: &[f64]| -> Vec<f64> {
        theta
            .iter()
            .zip(is_var.iter())
            .map(|(t, iv)| if *iv { t.max(0.0).sqrt() } else { *t })
            .collect()
    };
    let mut x0: Vec<f64> = Vec::new();
    x0.extend(start_b.iter().cloned());
    x0.extend(start_p.iter().cloned());
    let x0 = to_u(&x0);
    let (u_opt, fun, _conv) = bfgs(&obj, &x0, 800);
    let theta = to_theta(&u_opt);
    // Second start if the first landed badly (non-finite or huge).
    let (theta, fun) = if !fun.is_finite() || fun > 1e50 {
        let alt_b: Vec<f64> = model
            .b_free
            .iter()
            .map(|p| {
                if model.is_latent[p.col] && !model.is_latent[p.row] {
                    0.5
                } else {
                    0.2
                }
            })
            .collect();
        let alt_p: Vec<f64> = model
            .psi_free
            .iter()
            .map(|p| match p.kind {
                ParamKind::PsiVar => obs_var(p.row),
                ParamKind::PsiCov => 0.0,
                ParamKind::B => 0.0,
            })
            .collect();
        let mut xa: Vec<f64> = Vec::new();
        xa.extend(alt_b);
        xa.extend(alt_p);
        let xa = to_u(&xa);
        let (u2, f2, _) = bfgs(&obj, &xa, 800);
        let t2 = to_theta(&u2);
        if f2 < fun {
            (t2, f2)
        } else {
            (theta, fun)
        }
    } else {
        (theta, fun)
    };
    if !fun.is_finite() || fun >= 1e99 {
        return Err("SEM fit failed: optimiser did not converge to a positive-definite solution".to_string());
    }
    let (tb, tp) = theta.split_at(nb);
    let (_, sigma) = mlw_objective(model, fit, tb, tp)
        .ok_or_else(|| "SEM fit failed: could not evaluate fitted covariance".to_string())?;
    Ok(SemFit {
        theta_b: tb.to_vec(),
        theta_p: tp.to_vec(),
        fun,
        sigma_obs: sigma,
    })
}

/// Expected Fisher-information SEs: I_ij = (n/2) tr(W S_i W S_j),
/// W = Sigma_obs^{-1}, S_i = d Sigma_obs / d theta_i (analytic via
/// the RAM form). Returns SEs aligned with (b_free, psi_free).
fn sem_ses(model: &SemModel, fit: &FitData, semfit: &SemFit) -> Vec<Option<f64>> {
    let nv = model.vars.len();
    let nb = model.b_free.len();
    let np = model.psi_free.len();
    let npar = nb + np;
    // Rebuild B, Psi, A^{-1}.
    let mut b = DMatrix::<f64>::zeros(nv, nv);
    for (fi, prm) in model.b_free.iter().enumerate() {
        b[(prm.row, prm.col)] = semfit.theta_b[fi];
    }
    for ((r, c), val) in &model.b_map {
        if let RowVal::Fixed(v) = val {
            b[(*r, *c)] = *v;
        }
    }
    let mut psi = DMatrix::<f64>::zeros(nv, nv);
    for (fi, prm) in model.psi_free.iter().enumerate() {
        psi[(prm.row, prm.col)] = semfit.theta_p[fi];
        psi[(prm.col, prm.row)] = semfit.theta_p[fi];
    }
    for ((r, c), val) in &model.psi_map {
        if let RowVal::Fixed(v) = val {
            psi[(*r, *c)] = *v;
            psi[(*c, *r)] = *v;
        }
    }
    for (vi, _) in model.vars.iter().enumerate() {
        if model.is_exog_obs[vi] {
            for (vj, _) in model.vars.iter().enumerate() {
                if model.is_exog_obs[vj] {
                    if let Some(v) = fit.s_map.get(&(vi, vj)) {
                        psi[(vi, vj)] = *v;
                    }
                }
            }
        }
    }
    let amat = DMatrix::<f64>::identity(nv, nv) - &b;
    let ainv = match amat.clone().lu().solve(&DMatrix::<f64>::identity(nv, nv)) {
        Some(m) => m,
        None => return vec![None; npar],
    };
    let sigma_full = &ainv * &psi * &ainv.transpose();
    let w_full = match semfit.sigma_obs.clone().cholesky() {
        Some(ch) => ch.inverse(),
        None => return vec![None; npar],
    };
    // Derivatives of Sigma_obs per free parameter.
    let mut ders: Vec<DMatrix<f64>> = Vec::new();
    let p_obs = model.observed.len();
    let obs_pos = |vi: usize| -> Option<usize> { fit.obs_idx.iter().position(|&x| x == vi) };
    // B params: dSigma = A^{-1} E_rc Sigma + Sigma E_rc' A^{-T}
    // (since dA = -E_rc for a B param).
    for prm in &model.b_free {
        let mut e = DMatrix::<f64>::zeros(nv, nv);
        e[(prm.row, prm.col)] = 1.0;
        let d_full = &ainv * &e * &sigma_full + &sigma_full * e.transpose() * &ainv.transpose();
        let mut d = DMatrix::<f64>::zeros(p_obs, p_obs);
        for (oi, &vi) in fit.obs_idx.iter().enumerate() {
            for (oj, &vj) in fit.obs_idx.iter().enumerate() {
                d[(oi, oj)] = d_full[(vi, vj)];
            }
        }
        ders.push(d);
        let _ = obs_pos;
    }
    // Psi params: dSigma = A^{-1} dPsi A^{-T}.
    for prm in &model.psi_free {
        let mut dpsi = DMatrix::<f64>::zeros(nv, nv);
        dpsi[(prm.row, prm.col)] += 1.0;
        if prm.row != prm.col {
            dpsi[(prm.col, prm.row)] += 1.0;
        }
        let d_full = &ainv * &dpsi * &ainv.transpose();
        let mut d = DMatrix::<f64>::zeros(p_obs, p_obs);
        for (oi, &vi) in fit.obs_idx.iter().enumerate() {
            for (oj, &vj) in fit.obs_idx.iter().enumerate() {
                d[(oi, oj)] = d_full[(vi, vj)];
            }
        }
        ders.push(d);
    }
    let n = fit.n as f64;
    let mut fim = DMatrix::<f64>::zeros(npar, npar);
    for i in 0..npar {
        let mi = &w_full * &ders[i];
        for j in 0..npar {
            fim[(i, j)] = (n / 2.0) * (&mi * &w_full * &ders[j]).trace();
            // trace(W D_i W D_j) = trace((W D_i)(W D_j))
            fim[(i, j)] = (n / 2.0) * (mi.clone() * &w_full * &ders[j]).trace();
        }
    }
    // Invert (pinv fallback for boundary solutions).
    let cov = match fim.clone().cholesky() {
        Some(ch) => ch.inverse(),
        None => {
            // SVD pinv
            let svd = fim.clone().svd(true, true);
            match (svd.u, svd.v_t) {
                (Some(u), Some(vt)) => {
                    let s = &svd.singular_values;
                    let smax = s.iter().cloned().fold(0.0f64, f64::max);
                    let tol = smax * 1e-12 * (npar.max(1) as f64);
                    let mut sinv = DMatrix::<f64>::zeros(npar, npar);
                    for i in 0..s.len().min(npar) {
                        if s[i] > tol {
                            sinv[(i, i)] = 1.0 / s[i];
                        }
                    }
                    vt.transpose() * sinv * u.transpose()
                }
                _ => return vec![None; npar],
            }
        }
    };
    (0..npar)
        .map(|i| {
            let v = cov[(i, i)];
            if v.is_finite() && v >= 0.0 {
                Some(v.sqrt())
            } else {
                None
            }
        })
        .collect()
}

fn fit_indices(
    model: &SemModel,
    fit: &FitData,
    semfit: &SemFit,
) -> Map<String, Value> {
    let p = model.observed.len();
    let n = fit.n as f64;
    let k = model.b_free.len() + model.psi_free.len();
    let chi2 = n * semfit.fun;
    let dof = (p * (p + 1) / 2) as i64 - k as i64;
    // Baseline (independence) model: Sigma = diag(S).
    let mut sum_log_diag = 0.0;
    for i in 0..p {
        sum_log_diag += fit.s[(i, i)].ln();
    }
    let logdet_s = {
        let ch = fit.s.clone().cholesky().unwrap();
        let mut s = 0.0;
        for i in 0..p {
            s += ch.l()[(i, i)].ln();
        }
        2.0 * s
    };
    let f_base = sum_log_diag - logdet_s;
    let chi2_base = n * f_base;
    let dof_base = (p * (p + 1) / 2 - p) as i64;
    let gfi = if chi2_base != 0.0 { 1.0 - chi2 / chi2_base } else { f64::NAN };
    let agfi = if dof != 0 {
        1.0 - (dof_base as f64 / dof as f64) * (1.0 - gfi)
    } else {
        f64::NAN
    };
    let nfi = if chi2_base != 0.0 { (chi2_base - chi2) / chi2_base } else { f64::NAN };
    let tli = if dof != 0 && dof_base != 0 {
        let a = chi2 / dof as f64;
        let bb = chi2_base / dof_base as f64;
        (bb - a) / (bb - 1.0)
    } else {
        f64::NAN
    };
    let cfi = if chi2_base - dof_base as f64 != 0.0 {
        1.0 - (chi2 - dof as f64) / (chi2_base - dof_base as f64)
    } else {
        f64::NAN
    };
    let rmsea = if chi2 < dof as f64 {
        0.0
    } else if dof > 0 {
        ((chi2 / dof as f64 - 1.0) / (n - 1.0)).max(0.0).sqrt()
    } else {
        f64::NAN
    };
    let lh = semfit.fun;
    let aic = 2.0 * (k as f64 - lh);
    let bic = n.ln() * k as f64 - 2.0 * lh;
    let mut m = Map::new();
    m.insert("chi2".to_string(), f6(chi2));
    m.insert("df".to_string(), json!(dof as f64));
    m.insert("gfi".to_string(), f6(gfi));
    m.insert("agfi".to_string(), f6(agfi));
    m.insert("cfi".to_string(), f6(cfi));
    m.insert("nfi".to_string(), f6(nfi));
    m.insert("tli".to_string(), f6(tli));
    m.insert("rmsea".to_string(), f6(rmsea));
    m.insert("aic".to_string(), f6(aic));
    m.insert("bic".to_string(), f6(bic));
    m.insert("ll".to_string(), f6(lh));
    m
}

fn params_json(model: &SemModel, semfit: &SemFit, ses: &[Option<f64>]) -> Value {
    let nb = model.b_free.len();
    let mut arr = Vec::new();
    for row in model.rows_reg.iter().chain(model.rows_cov.iter()) {
        let mut e = Map::new();
        e.insert("lv".to_string(), json!(row.lv));
        e.insert("op".to_string(), json!(row.op));
        e.insert("rv".to_string(), json!(row.rv));
        match &row.val {
            RowVal::Fixed(v) => {
                e.insert("c".to_string(), f6(*v));
            }
            RowVal::Free(fi) => {
                // Free-index spaces: B rows use b_free indices, Psi
                // rows use psi_free indices. Rows know their kind by
                // op: "~" rows are B, "~~" rows are Psi.
                let (val, se) = if row.op == "~" {
                    (semfit.theta_b[*fi], ses[*fi])
                } else {
                    (semfit.theta_p[*fi], ses[nb + *fi])
                };
                e.insert("c".to_string(), f6(val));
                match se {
                    Some(s) if s.is_finite() && s > 0.0 => {
                        let z = val / (s + 1e-12);
                        e.insert("s".to_string(), f6(s));
                        e.insert("z".to_string(), f6(z));
                        e.insert("p".to_string(), f6(2.0 * norm_sf(z.abs())));
                    }
                    Some(s) => {
                        e.insert("s".to_string(), f6(s));
                    }
                    None => {}
                }
            }
        }
        arr.push(Value::Object(e));
    }
    Value::Array(arr)
}

fn prepare(tab: &Table, desc: &str) -> Result<(SemModel, FitData), String> {
    let model = build_model(desc, &tab.headers)?;
    let mut fit = sample_cov(tab, &model.observed)?;
    // Variable-space indices of observed vars.
    fit.obs_idx = model
        .observed
        .iter()
        .map(|o| model.vars.iter().position(|v| v == o).unwrap())
        .collect();
    // Sample covariance map keyed by variable-space index (for the
    // exogenous-observed Psi block).
    for (oi, &vi) in fit.obs_idx.iter().enumerate() {
        for (oj, &vj) in fit.obs_idx.iter().enumerate() {
            fit.s_map.insert((vi, vj), fit.s[(oi, oj)]);
        }
    }
    Ok((model, fit))
}

pub fn dispatch_sem(action: &str, tab: &Table, desc: &str, mode: &str) -> Result<Value, String> {
    match action {
        "sem_fit" => {
            if desc.trim().is_empty() {
                return Err(
                    "sem_fit requires 'desc' (model description in lavaan-like syntax)"
                        .to_string(),
                );
            }
            if tab.headers.is_empty() {
                return Err("sem_fit requires 'd' (CSV data)".to_string());
            }
            let (model, fit) = prepare(tab, desc)?;
            let semfit = fit_sem(&model, &fit)?;
            let ses = sem_ses(&model, &fit, &semfit);
            let mut out = Map::new();
            out.insert("params".to_string(), params_json(&model, &semfit, &ses));
            out.insert("fit".to_string(), Value::Object(fit_indices(&model, &fit, &semfit)));
            out.insert("n".to_string(), json!(fit.n));
            Ok(Value::Object(out))
        }
        "sem_inspect" => {
            if desc.trim().is_empty() {
                return Err(
                    "sem_inspect requires 'desc' (same model description as used in fit)"
                        .to_string(),
                );
            }
            if tab.headers.is_empty() {
                // No data in this (stateless CLI) invocation: mirror
                // the Python tool's cache-miss error.
                return Err(
                    "No fitted model cached for this description. Run sem_fit first with the same 'desc' in this session."
                        .to_string(),
                );
            }
            // CLI semantics: refit deterministically from the data.
            let (model, fit) = prepare(tab, desc)?;
            let semfit = fit_sem(&model, &fit)?;
            let ses = sem_ses(&model, &fit, &semfit);
            let mut out = Map::new();
            out.insert("mode".to_string(), json!(if mode.is_empty() { "raw" } else { mode }));
            out.insert("params".to_string(), params_json(&model, &semfit, &ses));
            Ok(Value::Object(out))
        }
        _ => Err(format!(
            "unknown action '{}'. valid: ['sem_fit', 'sem_inspect']",
            action
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_csv;

    fn one_factor_csv() -> String {
        // 3 indicators, just-identified (df=0): chi2 must be ~0.
        let mut s = String::from("x1,x2,x3\n");
        let mut seed = 987654321u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64) * 2.0 - 1.0
        };
        for _ in 0..200 {
            let f = rnd();
            s.push_str(&format!(
                "{:.3},{:.3},{:.3}\n",
                f + rnd() * 0.5,
                0.9 * f + rnd() * 0.5,
                1.1 * f + rnd() * 0.5
            ));
        }
        s
    }

    #[test]
    fn sem_one_factor_just_identified() {
        let tab = parse_csv(&one_factor_csv()).unwrap();
        let v = dispatch_sem("sem_fit", &tab, "f1 =~ x1 + x2 + x3", "raw").unwrap();
        let chi2 = v["fit"]["chi2"].as_f64().unwrap();
        assert!(chi2.abs() < 0.05, "chi2 {}", chi2);
        assert_eq!(v["fit"]["df"], 0.0);
        let params = v["params"].as_array().unwrap();
        // First loading fixed at 1.0 with no SE.
        let x1 = params.iter().find(|p| p["lv"] == "x1").unwrap();
        assert_eq!(x1["c"], 1.0);
        assert!(x1.get("s").is_none());
        let x2 = params.iter().find(|p| p["lv"] == "x2").unwrap();
        let c2 = x2["c"].as_f64().unwrap();
        assert!((c2 - 0.9).abs() < 0.2, "x2 loading {}", c2);
    }

    #[test]
    fn sem_parse_errors() {
        let tab = parse_csv("x1,x2\n1,2\n3,4").unwrap();
        assert!(dispatch_sem("sem_fit", &tab, "", "raw").is_err());
        assert!(dispatch_sem("sem_inspect", &Table { headers: vec![], rows: vec![] }, "f =~ x1", "raw").is_err());
    }
}
