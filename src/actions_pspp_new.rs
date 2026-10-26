// NEW PSPP actions (dca, quantreg, negbin, mediation, pscore, gam, zip, lasso,
// adf, beta) — faithful ports of the Python originals in pspp_tool.py, including
// their quirks (see notes per function). They mirror numpy lstsq via an SVD
// pseudo-inverse solve, so coefficients match to ~1e-9 before 4dp rounding.
//
// Params come from actions_pspp::P, extended with:
//   tau      (quantreg, default 0.5; guarded to 0.5 if outside (0,1))
//   lam      (lasso, default 0.1)
//   df_knots (gam knots per term, default 5; <=0 guarded to 5)
//   lags     (adf, default 1)
// The parent merge must add those four fields to P and wire --tau/--lam/--df/
// --lags + --params overrides in main.rs (exact lines in the report).
use crate::actions_pspp::P;
use crate::*;
use nalgebra::{DMatrix, DVector};
use serde_json::{json, Value};

// ---------- helpers ----------

// numpy lstsq via SVD pseudo-inverse (same truncation rule as lib::ols).
fn lstsq(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<Vec<f64>> {
    let (m, k) = (a.nrows(), a.ncols());
    if k == 0 {
        return Some(vec![]);
    }
    // nalgebra panics on the SVD of an empty (0-row) matrix; surface it as
    // a clean error instead (all callers map None to "Matrix singular").
    if m == 0 {
        return None;
    }
    let svd = a.clone().svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let s = svd.singular_values;
    if s[0] == 0.0 {
        return Some(vec![0.0; k]);
    }
    let tol = f64::EPSILON * (m.max(k) as f64) * s[0];
    let utb = u.transpose() * b;
    let mut tmp = DVector::zeros(k);
    for i in 0..k {
        if s[i] > tol {
            tmp[i] = utb[i] / s[i];
        }
    }
    let beta = vt.transpose() * tmp;
    Some(beta.iter().cloned().collect())
}

// Solve a (possibly square, possibly singular) normal system the numpy way:
// beta = lstsq(A, b). Returns None only if SVD itself fails.
fn solve_normal(ata: &DMatrix<f64>, atb: &DVector<f64>) -> Option<Vec<f64>> {
    lstsq(ata, atb)
}

fn design(xcols: &[Vec<f64>], n: usize) -> DMatrix<f64> {
    let k = xcols.len() + 1;
    let mut m = DMatrix::zeros(n, k);
    for i in 0..n {
        m[(i, 0)] = 1.0;
        for (j, c) in xcols.iter().enumerate() {
            m[(i, j + 1)] = c[i];
        }
    }
    m
}

fn mat_vec(x: &DMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let bv = DVector::from_column_slice(b);
    (x * bv).iter().cloned().collect()
}

// X' diag(w) X and X' (w .* z), accumulated directly (no sqrt trick, so the
// weights match numpy's diag formulation bit-for-bit up to fp summation order).
fn xtwx_xtwz(x: &DMatrix<f64>, w: &[f64], z: &[f64]) -> (DMatrix<f64>, DVector<f64>) {
    let (n, k) = (x.nrows(), x.ncols());
    let mut ata = DMatrix::zeros(k, k);
    let mut atb = DVector::zeros(k);
    for i in 0..n {
        for a in 0..k {
            atb[a] += x[(i, a)] * w[i] * z[i];
            for b in 0..k {
                ata[(a, b)] += x[(i, a)] * w[i] * x[(i, b)];
            }
        }
    }
    (ata, atb)
}

fn xtx(x: &DMatrix<f64>) -> DMatrix<f64> {
    let (n, k) = (x.nrows(), x.ncols());
    let mut ata = DMatrix::zeros(k, k);
    for i in 0..n {
        for a in 0..k {
            for b in 0..k {
                ata[(a, b)] += x[(i, a)] * x[(i, b)];
            }
        }
    }
    ata
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn sigmoid(e: f64) -> f64 {
    if e >= 0.0 {
        1.0 / (1.0 + (-e).exp())
    } else {
        let ex = e.exp();
        ex / (1.0 + ex)
    }
}

fn std_pop(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

// Sample (ddof=1) variance that propagates NaN for n<2, like numpy.
fn var_ddof1_nan(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return f64::NAN;
    }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

fn mean_or_nan(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        mean(v)
    }
}

// Outcome + predictors, truncated to the shortest column (numpy zip semantics).
fn get_xy(tab: &Table, o: &str, preds: &[String]) -> Result<(Vec<f64>, Vec<Vec<f64>>, usize), String> {
    if o.is_empty() || preds.is_empty() {
        return Err("o (outcome), p (predictors) required".into());
    }
    if tab.col_idx(o).is_none() {
        return Err(format!("Outcome '{}' not found", o));
    }
    for pv in preds {
        if tab.col_idx(pv).is_none() {
            return Err(format!("Predictor '{}' not found", pv));
        }
    }
    let y_full = tab.numeric(o)?;
    let x_full: Vec<Vec<f64>> = preds.iter().map(|pv| tab.numeric(pv).unwrap_or_default()).collect();
    let n = y_full.len().min(x_full.iter().map(|c| c.len()).min().unwrap_or(0));
    let y = y_full[..n].to_vec();
    let xcols: Vec<Vec<f64>> = x_full.iter().map(|c| c[..n].to_vec()).collect();
    Ok((y, xcols, n))
}

fn coef_names(preds: &[String]) -> Vec<String> {
    std::iter::once("(Intercept)".to_string()).chain(preds.iter().cloned()).collect()
}

// MT19937, plain init_genrand(seed) — verified bit-exact against
// numpy RandomState(42) (choice == randint == masked-rejection draws).
struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}
impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: 624 }
    }
    fn gen(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    self.mt[i] ^= 2567483615;
                }
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
    // numpy legacy rk_interval: mask up to the next power of two, reject >= n.
    fn randint(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let mut mask = (n - 1) as u32;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.gen() & mask;
            if (v as usize) < n {
                return v as usize;
            }
        }
    }
}

pub fn dispatch_new(t: &str, tab: &Table, p: &P) -> Result<Value, String> {
    match t {
        "dca" => dca(tab, p),
        "quantreg" => quantreg(tab, p),
        "negbin" => negbin(tab, p),
        "mediation" => mediation(tab, p),
        "pscore" => pscore(tab, p),
        "gam" => gam(tab, p),
        "zip" => zip(tab, p),
        "lasso" => lasso(tab, p),
        "adf" => adf(tab, p),
        "beta" => beta_reg(tab, p),
        _ => Err(format!("Unknown test '{}'", t)),
    }
}

// ---------- dca ----------
fn dca(tab: &Table, p: &P) -> Result<Value, String> {
    let a = &p.a;
    let v = p.vs();
    if a.is_empty() || v.is_empty() || tab.col_idx(a).is_none() || tab.col_idx(&v).is_none() {
        return Err("a (binary outcome), v (predicted probability) required".into());
    }
    let y_full = tab.numeric(a)?;
    let ph_full = tab.numeric(&v)?;
    let n = y_full.len().min(ph_full.len());
    let (y, ph) = (&y_full[..n], &ph_full[..n]);
    let prev = if n > 0 {
        y.iter().filter(|&&yi| yi >= 0.5).count() as f64 / n as f64
    } else {
        0.0
    };
    let mut thresholds = Vec::new();
    let mut nb_out = Vec::new();
    let mut ta_out = Vec::new();
    // numpy linspace: pt = i * step + start with step = (stop-start)/49, and
    // the last point forced to stop exactly. The multiply-then-add order
    // matters at the ulp level: scores equal to a threshold (e.g. 0.29) flip
    // classification if pt is 1 ulp on the other side.
    let step = (0.99 - 0.01) / 49.0;
    for i in 0..50 {
        let pt = if i == 49 { 0.99 } else { i as f64 * step + 0.01 };
        let tp = ph.iter().zip(y.iter()).filter(|(pp, yi)| **pp >= pt && **yi >= 0.5).count() as f64;
        let fp = ph.iter().zip(y.iter()).filter(|(pp, yi)| **pp >= pt && **yi < 0.5).count() as f64;
        let w = pt / (1.0 - pt);
        let nb = if n > 0 { (tp - fp * w) / n as f64 } else { 0.0 };
        thresholds.push(r4(pt));
        nb_out.push(r4(nb));
        ta_out.push(r4(prev - (1.0 - prev) * w));
    }
    Ok(json!({"thresholds": thresholds, "net_benefit": nb_out, "treat_all": ta_out}))
}

// ---------- quantreg ----------
fn quantreg(tab: &Table, p: &P) -> Result<Value, String> {
    let (y, xcols, n) = get_xy(tab, &p.o, &p.p)?;
    let tau = if p.tau > 0.0 && p.tau < 1.0 { p.tau } else { 0.5 };
    let x = design(&xcols, n);
    let yv = DVector::from_column_slice(&y);
    let mut beta = lstsq(&x, &yv).ok_or("Matrix singular")?;
    for _ in 0..100 {
        let pred = mat_vec(&x, &beta);
        let resid: Vec<f64> = y.iter().zip(pred.iter()).map(|(a, b)| a - b).collect();
        let w: Vec<f64> = resid
            .iter()
            .map(|r| {
                let base = 1.0 / (r.abs() + 1e-6);
                if *r > 0.0 { tau * base } else { (1.0 - tau) * base }
            })
            .collect();
        let (ata, atb) = xtwx_xtwz(&x, &w, &y);
        let beta_new = solve_normal(&ata, &atb).ok_or("Matrix singular")?;
        let done = max_abs_diff(&beta_new, &beta) < 1e-6;
        beta = beta_new;
        if done {
            break;
        }
    }
    let names = coef_names(&p.p);
    let coefs: Vec<Value> = names
        .iter()
        .enumerate()
        .map(|(i, name)| json!({"n": name, "B": r4(beta[i]), "tau": tau}))
        .collect();
    Ok(json!({"tau": tau, "n": n, "coef": coefs}))
}

// ---------- negbin ----------
// Faithful to Python: alternating IRLS-beta / moments-alpha. Note the alpha
// update uses the mu from the START of the iteration (before the beta update),
// and on beta convergence the loop breaks BEFORE the alpha update.
fn negbin(tab: &Table, p: &P) -> Result<Value, String> {
    let (y, xcols, n) = get_xy(tab, &p.o, &p.p)?;
    let x = design(&xcols, n);
    let k = x.ncols();
    let logy: Vec<f64> = y.iter().map(|v| v.max(0.1).ln()).collect();
    let mut beta = lstsq(&x, &DVector::from_column_slice(&logy)).ok_or("Matrix singular")?;
    let mut alpha = 1.0f64;
    for _ in 0..30 {
        let eta = mat_vec(&x, &beta);
        let mu: Vec<f64> = eta.iter().map(|e| e.exp().clamp(1e-6, 1e6)).collect();
        let w: Vec<f64> = mu.iter().map(|m| m / (1.0 + alpha * m)).collect();
        let z: Vec<f64> = mu
            .iter()
            .zip(y.iter())
            .map(|(m, yi)| m.max(1e-6).ln() + (yi - m) / m)
            .collect();
        let (ata, atb) = xtwx_xtwz(&x, &w, &z);
        let beta_new = solve_normal(&ata, &atb).ok_or("Matrix singular")?;
        let done = max_abs_diff(&beta_new, &beta) < 1e-6;
        beta = beta_new;
        if done {
            break;
        }
        let pearson: f64 = y
            .iter()
            .zip(mu.iter())
            .map(|(yi, m)| (yi - m).powi(2) / (m + alpha * m * m))
            .sum();
        let sum_mu: f64 = mu.iter().sum();
        alpha = if sum_mu > 0.0 {
            (pearson - (n - k) as f64) / sum_mu
        } else {
            0.001
        };
        alpha = alpha.max(0.001);
    }
    let eta = mat_vec(&x, &beta);
    let mu: Vec<f64> = eta.iter().map(|e| e.exp().clamp(1e-6, 1e6)).collect();
    let w: Vec<f64> = mu.iter().map(|m| m / (1.0 + alpha * m)).collect();
    let (ata, _) = xtwx_xtwz(&x, &w, &vec![0.0; n]);
    let cov = mat_inverse(&ata);
    let names = coef_names(&p.p);
    let mut coefs = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let se = match &cov {
            Some(c) => c[(i, i)].max(0.0).sqrt(),
            None => f64::NAN,
        };
        let z = beta[i] / se;
        // Python: p uses 0 in place of a NaN z (=> p = 1.0); inf z => p = 0.
        let pval = if z.is_nan() { 1.0 } else { 2.0 * norm_sf(z.abs()) };
        coefs.push(json!({"n": name, "B": r4(beta[i]), "SE": r4(se), "z": r4(z), "p": r4(pval), "IRR": r4(beta[i].exp())}));
    }
    Ok(json!({"n": n, "alpha": r4(alpha), "coef": coefs}))
}

// ---------- mediation ----------
// Baron-Kenny OLS paths + 500 bootstrap resamples, numpy RandomState(42)
// reproduced bit-exactly (MT19937 + masked rejection). CI indices 12 and 487
// of the sorted bootstrap distribution (int(500*0.025), int(500*0.975)).
// Covariates in p are IGNORED by Python (unadjusted paths); only the note
// string changes.
fn mediation(tab: &Table, p: &P) -> Result<Value, String> {
    let (o, a, v) = (&p.o, &p.a, p.vs());
    if o.is_empty() || a.is_empty() || v.is_empty() {
        return Err("o (outcome Y), a (treatment X), v (mediator M) required".into());
    }
    if tab.col_idx(o).is_none() || tab.col_idx(a).is_none() || tab.col_idx(&v).is_none() {
        return Err("Variables not found".into());
    }
    let y_full = tab.numeric(o)?;
    let x_full = tab.numeric(a)?;
    let m_full = tab.numeric(&v)?;
    let n = y_full.len().min(x_full.len()).min(m_full.len());
    let (y, xv, mv) = (&y_full[..n], &x_full[..n], &m_full[..n]);
    let ols_slope = |xcol: &[f64], target: &[f64]| -> Option<Vec<f64>> {
        let xm = design(&[xcol.to_vec()], xcol.len());
        lstsq(&xm, &DVector::from_column_slice(target))
    };
    let beta_a = ols_slope(xv, mv).ok_or("Matrix singular")?;
    let a_coef = beta_a[1];
    let xm_mat = design(&[xv.to_vec(), mv.to_vec()], n);
    let beta_bc = lstsq(&xm_mat, &DVector::from_column_slice(y)).ok_or("Matrix singular")?;
    let b_coef = beta_bc[2];
    let c_prime = beta_bc[1];
    let beta_c = ols_slope(xv, y).ok_or("Matrix singular")?;
    let c_total = beta_c[1];
    let indirect = a_coef * b_coef;
    let mut rng = Mt19937::new(42);
    let mut boot: Vec<f64> = Vec::with_capacity(500);
    for _ in 0..500 {
        let idx: Vec<usize> = (0..n).map(|_| rng.randint(n)).collect();
        let xb: Vec<f64> = idx.iter().map(|&i| xv[i]).collect();
        let mb: Vec<f64> = idx.iter().map(|&i| mv[i]).collect();
        let yb: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let a_b = ols_slope(&xb, &mb).ok_or("Matrix singular")?[1];
        let xmb = design(&[xb, mb], n);
        let b_b = lstsq(&xmb, &DVector::from_column_slice(&yb)).ok_or("Matrix singular")?[2];
        boot.push(a_b * b_b);
    }
    boot.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_low = boot[12];
    let ci_high = boot[487];
    let prop_med = if c_total.abs() > 1e-10 { indirect / c_total } else { 0.0 };
    let note = if !p.p.is_empty() {
        "unadjusted OLS paths — covariates in p are ignored"
    } else {
        "unadjusted OLS paths"
    };
    Ok(json!({
        "n": n,
        "total_effect": r4(c_total),
        "direct_effect": r4(c_prime),
        "indirect_effect": r4(indirect),
        "prop_mediated": r4(prop_med),
        "indirect_ci": [r4(ci_low), r4(ci_high)],
        "paths": {"a": r4(a_coef), "b": r4(b_coef), "c": r4(c_total), "c_prime": r4(c_prime)},
        "note": note,
    }))
}

// ---------- pscore ----------
// Logistic IRLS, exactly 50 iterations (no convergence check in Python),
// ps clipped to [0.01, 0.99] after fitting; SMD before/after ATE weighting.
fn pscore(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.p.is_empty() {
        return Err("a (treatment), p (covariates) required".into());
    }
    if tab.col_idx(&p.a).is_none() {
        return Err(format!("Treatment '{}' not found", p.a));
    }
    for pv in &p.p {
        if tab.col_idx(pv).is_none() {
            return Err(format!("Covariate '{}' not found", pv));
        }
    }
    let treat_full = tab.numeric(&p.a)?;
    let x_full: Vec<Vec<f64>> = p.p.iter().map(|pv| tab.numeric(pv).unwrap_or_default()).collect();
    let n = treat_full.len().min(x_full.iter().map(|c| c.len()).min().unwrap_or(0));
    let treat = &treat_full[..n];
    let xcols: Vec<Vec<f64>> = x_full.iter().map(|c| c[..n].to_vec()).collect();
    let x = design(&xcols, n);
    let k = x.ncols();
    let mut beta = vec![0.0; k];
    for _ in 0..50 {
        let eta_raw = mat_vec(&x, &beta);
        let eta: Vec<f64> = eta_raw.iter().map(|e| e.clamp(-30.0, 30.0)).collect();
        let mu: Vec<f64> = eta
            .iter()
            .map(|e| sigmoid(*e).clamp(1e-10, 1.0 - 1e-10))
            .collect();
        let w: Vec<f64> = mu.iter().map(|m| m * (1.0 - m)).collect();
        let z: Vec<f64> = eta
            .iter()
            .zip(mu.iter())
            .zip(treat.iter())
            .map(|((e, m), t)| e + (t - m) / (m * (1.0 - m)))
            .collect();
        let (ata, atb) = xtwx_xtwz(&x, &w, &z);
        beta = solve_normal(&ata, &atb).ok_or("Matrix singular")?;
    }
    let eta = mat_vec(&x, &beta);
    let ps: Vec<f64> = eta.iter().map(|e| sigmoid(*e).clamp(0.01, 0.99)).collect();
    let w_ate: Vec<f64> = treat
        .iter()
        .zip(ps.iter())
        .map(|(t, s)| if *t >= 0.5 { 1.0 / s } else { 1.0 / (1.0 - s) })
        .collect();
    let treated: Vec<bool> = treat.iter().map(|t| *t >= 0.5).collect();
    let ps_t: Vec<f64> = ps.iter().zip(treated.iter()).filter(|(_, &t)| t).map(|(s, _)| *s).collect();
    let ps_c: Vec<f64> = ps.iter().zip(treated.iter()).filter(|(_, &t)| !t).map(|(s, _)| *s).collect();
    let mut smd_before = serde_json::Map::new();
    let mut smd_after = serde_json::Map::new();
    for (idx, pv) in p.p.iter().enumerate() {
        let col = &xcols[idx];
        let t1: Vec<f64> = col.iter().zip(treated.iter()).filter(|(_, &t)| t).map(|(v, _)| *v).collect();
        let t0: Vec<f64> = col.iter().zip(treated.iter()).filter(|(_, &t)| !t).map(|(v, _)| *v).collect();
        let sd_pool = ((var_ddof1_nan(&t1) + var_ddof1_nan(&t0)) / 2.0).sqrt();
        let smd_b = if sd_pool > 0.0 { (mean_or_nan(&t1) - mean_or_nan(&t0)).abs() / sd_pool } else { 0.0 };
        let w1: Vec<f64> = w_ate.iter().zip(treated.iter()).filter(|(_, &t)| t).map(|(w, _)| *w).collect();
        let w0: Vec<f64> = w_ate.iter().zip(treated.iter()).filter(|(_, &t)| !t).map(|(w, _)| *w).collect();
        let m1_w = if w1.is_empty() {
            0.0
        } else {
            let sw: f64 = w1.iter().sum();
            t1.iter().zip(w1.iter()).map(|(v, w)| v * w).sum::<f64>() / sw
        };
        let m0_w = if w0.is_empty() {
            0.0
        } else {
            let sw: f64 = w0.iter().sum();
            t0.iter().zip(w0.iter()).map(|(v, w)| v * w).sum::<f64>() / sw
        };
        let smd_a = if sd_pool > 0.0 { (m1_w - m0_w).abs() / sd_pool } else { 0.0 };
        smd_before.insert(pv.clone(), r4(smd_b));
        smd_after.insert(pv.clone(), r4(smd_a));
    }
    Ok(json!({
        "n": n,
        "n_treated": treated.iter().filter(|&&t| t).count(),
        "n_control": treated.iter().filter(|&&t| !t).count(),
        "ps_mean_treated": r4(mean_or_nan(&ps_t)),
        "ps_mean_control": r4(mean_or_nan(&ps_c)),
        "smd_before": Value::Object(smd_before),
        "smd_after": Value::Object(smd_after),
        "note": "smd_after uses ATE (inverse-probability) weights for the average treatment effect in the whole sample. For the ATT (effect on the treated), reweight controls by ps/(1-ps) with treated weight 1.",
    }))
}

// ---------- gam ----------
// Truncated-power cubic basis per term: [1, x, (x-k)^3+ ...], knots =
// linspace(min, max, df+2)[1:-1]. NOTE the intercept column is repeated once
// per term (Python column_stacks each term's basis); only the FIRST column
// escapes the 0.001 ridge penalty. Solved as one lstsq system.
fn gam(tab: &Table, p: &P) -> Result<Value, String> {
    let (y, xcols, n) = get_xy(tab, &p.o, &p.p)?;
    let n_knots = if p.df_knots > 0 { p.df_knots as usize } else { 5 };
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for x in &xcols {
        let (xmin, xmax) = (
            x.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        );
        cols.push(vec![1.0; n]);
        cols.push(x.clone());
        for j in 0..n_knots {
            let knot = xmin + (j + 1) as f64 * (xmax - xmin) / (n_knots + 1) as f64;
            cols.push(x.iter().map(|v| (v - knot).max(0.0).powi(3)).collect());
        }
    }
    let ncols = cols.len();
    let mut xf = DMatrix::zeros(n, ncols);
    for (j, c) in cols.iter().enumerate() {
        for i in 0..n {
            xf[(i, j)] = c[i];
        }
    }
    let mut ata = xtx(&xf);
    for j in 0..ncols {
        if j != 0 {
            ata[(j, j)] += 0.001;
        }
    }
    let yv = DVector::from_column_slice(&y);
    let aty = xf.transpose() * &yv;
    let beta = solve_normal(&ata, &aty).ok_or("Matrix singular")?;
    let pred = mat_vec(&xf, &beta);
    let ss_res: f64 = y.iter().zip(pred.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let ym = mean(&y);
    let ss_tot: f64 = y.iter().map(|a| (a - ym).powi(2)).sum();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    Ok(json!({"n": n, "R2": r4(r2), "smooth_terms": p.p.len(), "knots_per_term": n_knots}))
}

// ---------- zip ----------
// EM, exactly 5 iterations. QUIRKS (faithful): the Poisson M-step is a plain
// weighted linear regression of y on X (not a log-link IRLS), and the
// zero-inflation M-step is a plain linear regression of tau_z on X (not a
// logistic IRLS). Structural-zero posterior for y>0 is 1e-10, not 0.
fn zip(tab: &Table, p: &P) -> Result<Value, String> {
    let (y, xcols, n) = get_xy(tab, &p.o, &p.p)?;
    let x = design(&xcols, n);
    let k = x.ncols();
    let logy: Vec<f64> = y.iter().map(|v| v.max(0.1).ln()).collect();
    let mut beta_p = lstsq(&x, &DVector::from_column_slice(&logy)).ok_or("Matrix singular")?;
    let mut gamma = vec![0.0; k];
    for _ in 0..5 {
        let mu_p: Vec<f64> = mat_vec(&x, &beta_p).iter().map(|e| e.exp().clamp(1e-10, 1e10)).collect();
        let pi_z: Vec<f64> = mat_vec(&x, &gamma)
            .iter()
            .map(|e| sigmoid(*e).clamp(1e-10, 1.0 - 1e-10))
            .collect();
        let tau_z: Vec<f64> = y
            .iter()
            .zip(pi_z.iter())
            .zip(mu_p.iter())
            .map(|((yi, pi), mu)| {
                if *yi < 0.5 {
                    pi / (pi + (1.0 - pi) * (-mu).exp())
                } else {
                    1e-10
                }
            })
            .collect();
        let w_p: Vec<f64> = tau_z.iter().map(|t| (1.0 - t).clamp(0.01, 1.0)).collect();
        let (ata, atb) = xtwx_xtwz(&x, &w_p, &y);
        beta_p = solve_normal(&ata, &atb).ok_or("Matrix singular")?;
        let xtx_m = xtx(&x);
        let xty = x.transpose() * DVector::from_column_slice(&tau_z);
        gamma = solve_normal(&xtx_m, &xty).ok_or("Matrix singular")?;
    }
    let pi_final: Vec<f64> = mat_vec(&x, &gamma).iter().map(|e| sigmoid(*e)).collect();
    let zero_prob = mean(&pi_final);
    let names = coef_names(&p.p);
    let poisson: Vec<Value> = names
        .iter()
        .enumerate()
        .map(|(i, nm)| json!({"n": nm, "B": r4(beta_p[i]), "IRR": r4(beta_p[i].exp())}))
        .collect();
    let zi: Vec<Value> = names
        .iter()
        .enumerate()
        .map(|(i, nm)| json!({"n": nm, "B": r4(gamma[i]), "OR": r4(gamma[i].exp())}))
        .collect();
    Ok(json!({"n": n, "zero_prob": r4(zero_prob), "poisson": poisson, "zero_infl": zi}))
}

// ---------- lasso ----------
// Coordinate descent on population-standardized (ddof=0) variables, threshold
// lam/2 (NOT lam/n), denominator Xj'Xj, 200 sweeps, tol 1e-6. Selection uses
// the SCALED coefficients; reported B are unscaled.
fn lasso(tab: &Table, p: &P) -> Result<Value, String> {
    let (y, xcols, n) = get_xy(tab, &p.o, &p.p)?;
    let lam = p.lam;
    let y_mean = mean(&y);
    let y_std = std_pop(&y);
    let y_scaled: Vec<f64> = if y_std > 0.0 {
        y.iter().map(|v| (v - y_mean) / y_std).collect()
    } else {
        y.iter().map(|v| v - y_mean).collect()
    };
    let k = p.p.len();
    let x_means: Vec<f64> = xcols.iter().map(|c| mean(c)).collect();
    let x_stds: Vec<f64> = xcols.iter().map(|c| std_pop(c)).collect();
    let x_scaled: Vec<Vec<f64>> = xcols
        .iter()
        .enumerate()
        .map(|(j, c)| {
            if x_stds[j] > 0.0 {
                c.iter().map(|v| (v - x_means[j]) / x_stds[j]).collect()
            } else {
                c.iter().map(|v| v - x_means[j]).collect()
            }
        })
        .collect();
    let mut beta = vec![0.0; k];
    for _ in 0..200 {
        let beta_old = beta.clone();
        for j in 0..k {
            let xj = &x_scaled[j];
            let denom: f64 = xj.iter().map(|v| v * v).sum();
            if denom == 0.0 {
                beta[j] = 0.0;
                continue;
            }
            let mut r = y_scaled.clone();
            for i in 0..n {
                let mut pred = 0.0;
                for q in 0..k {
                    pred += x_scaled[q][i] * beta[q];
                }
                r[i] = y_scaled[i] - pred + xj[i] * beta[j];
            }
            let rho: f64 = xj.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
            if rho < -lam / 2.0 {
                beta[j] = (rho + lam / 2.0) / denom;
            } else if rho > lam / 2.0 {
                beta[j] = (rho - lam / 2.0) / denom;
            } else {
                beta[j] = 0.0;
            }
        }
        if max_abs_diff(&beta, &beta_old) < 1e-6 {
            break;
        }
    }
    let mut beta_unscaled = vec![0.0; k];
    for j in 0..k {
        if x_stds[j] > 0.0 {
            beta_unscaled[j] = beta[j] * y_std / x_stds[j];
        }
    }
    let intercept = y_mean - (0..k).map(|j| beta_unscaled[j] * x_means[j]).sum::<f64>();
    let selected: Vec<String> = p
        .p
        .iter()
        .enumerate()
        .filter(|(j, _)| beta[*j].abs() > 1e-6)
        .map(|(_, name)| name.clone())
        .collect();
    let mut coefs = vec![json!({"n": "(Intercept)", "B": r4(intercept)})];
    for (j, name) in p.p.iter().enumerate() {
        coefs.push(json!({"n": name, "B": r4(beta_unscaled[j])}));
    }
    Ok(json!({"lam": lam, "n": n, "selected": selected.len(), "vars": selected, "coef": coefs}))
}

// ---------- adf ----------
// Regression of dy on [1, trend, y_lag, lagged dy...]; statistic = gamma/SE.
// Critical values are FIXED (-3.43/-2.86/-2.57) regardless of n in Python.
fn adf(tab: &Table, p: &P) -> Result<Value, String> {
    let v = p.vs();
    if v.is_empty() || tab.col_idx(&v).is_none() {
        return Err("v (time series variable) required".into());
    }
    let y = tab.numeric(&v)?;
    let n = y.len();
    if n < 10 {
        return Err(format!("Need ≥10 observations, got {}", n));
    }
    let lags = if p.lags < 0 { 0 } else { p.lags as usize };
    let dy: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    if lags + 1 >= n {
        return Err("lags too large for series length".into());
    }
    let y_lag = y[lags..n - 1].to_vec();
    let dy_use = dy[lags..].to_vec();
    let m = y_lag.len();
    let mut cols: Vec<Vec<f64>> = vec![vec![1.0; m], (1..=m).map(|i| i as f64).collect(), y_lag];
    for pi in 1..=lags {
        let start = lags - pi;
        let end = dy.len() - pi;
        cols.push(dy[start..end].to_vec());
    }
    let ncols = cols.len();
    let mut xm = DMatrix::zeros(m, ncols);
    for (j, c) in cols.iter().enumerate() {
        for i in 0..m {
            xm[(i, j)] = c[i];
        }
    }
    let beta = lstsq(&xm, &DVector::from_column_slice(&dy_use)).ok_or("Matrix singular")?;
    let pred = mat_vec(&xm, &beta);
    let ss_res: f64 = dy_use.iter().zip(pred.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let denom = (m as f64) - (ncols as f64);
    let sigma2 = ss_res / denom;
    let xtx_m = xtx(&xm);
    let se_gamma = match mat_inverse(&xtx_m) {
        Some(cov) => (sigma2 * cov[(2, 2)]).sqrt(),
        None => f64::NAN,
    };
    let gamma = beta[2];
    let adf_stat = if se_gamma > 0.0 { gamma / se_gamma } else { 0.0 };
    Ok(json!({
        "adf": r4(adf_stat), "n": n, "lags": lags,
        "crit_1pct": -3.43, "crit_5pct": -2.86, "crit_10pct": -2.57,
        "stationary_5pct": adf_stat < -2.86,
    }))
}

// ---------- beta ----------
// QUIRKS (faithful): the working response uses the FIXED logit of the
// clipped y (eta), not the current linear predictor; SEs come from
// inv(X' W X) with W = mu(1-mu) and no precision parameter; McFadden uses
// Bernoulli log-likelihoods of the clipped y. Boundary y are clipped to
// (0.001, 0.999) and counted in warn_clip.
fn beta_reg(tab: &Table, p: &P) -> Result<Value, String> {
    let (y_raw, xcols, n) = get_xy(tab, &p.o, &p.p)?;
    let n_clip = y_raw.iter().filter(|v| **v <= 0.0 || **v >= 1.0).count();
    let y: Vec<f64> = y_raw.iter().map(|v| v.clamp(0.001, 0.999)).collect();
    let x = design(&xcols, n);
    let eta: Vec<f64> = y.iter().map(|v| (v / (1.0 - v)).ln()).collect();
    let mut beta = lstsq(&x, &DVector::from_column_slice(&eta)).ok_or("Matrix singular")?;
    for _ in 0..30 {
        let mu: Vec<f64> = mat_vec(&x, &beta).iter().map(|e| sigmoid(*e).clamp(0.001, 0.999)).collect();
        let w: Vec<f64> = mu.iter().map(|m| m * (1.0 - m)).collect();
        let z: Vec<f64> = eta
            .iter()
            .zip(y.iter())
            .zip(mu.iter())
            .map(|((e, yi), m)| e + (yi - m) / (m * (1.0 - m)))
            .collect();
        let (ata, atb) = xtwx_xtwz(&x, &w, &z);
        let beta_new = solve_normal(&ata, &atb).ok_or("Matrix singular")?;
        let done = max_abs_diff(&beta_new, &beta) < 1e-6;
        beta = beta_new;
        if done {
            break;
        }
    }
    let mu: Vec<f64> = mat_vec(&x, &beta).iter().map(|e| sigmoid(*e).clamp(0.001, 0.999)).collect();
    let w: Vec<f64> = mu.iter().map(|m| m * (1.0 - m)).collect();
    let (ata, _) = xtwx_xtwz(&x, &w, &vec![0.0; n]);
    let cov = mat_inverse(&ata);
    let names = coef_names(&p.p);
    let mut coefs = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let se = match &cov {
            Some(c) => c[(i, i)].max(0.0).sqrt(),
            None => f64::NAN,
        };
        let z = beta[i] / se;
        let pval = if z.is_nan() { 1.0 } else { 2.0 * norm_sf(z.abs()) };
        coefs.push(json!({"n": name, "B": r4(beta[i]), "SE": r4(se), "z": r4(z), "p": r4(pval), "OR": r4(beta[i].exp())}));
    }
    let ll_full: f64 = y
        .iter()
        .zip(mu.iter())
        .map(|(yi, m)| yi * m.ln() + (1.0 - yi) * (1.0 - m).ln())
        .sum();
    let ym = mean(&y);
    let ll_null: f64 = y.iter().map(|yi| yi * ym.ln() + (1.0 - yi) * (1.0 - ym).ln()).sum();
    let mcf = if ll_null != 0.0 { 1.0 - ll_full / ll_null } else { 0.0 };
    let mut out = json!({
        "n": n, "R2_mcfadden": r4(mcf), "coef": coefs,
        "note": "logit link: exp(B) = odds ratio for mean proportion mu/(1-mu). Quasi-MLE via IRLS (no precision submodel); SEs are approximate.",
    });
    if n_clip > 0 {
        out.as_object_mut().unwrap().insert(
            "warn_clip".into(),
            json!(format!(
                "{}/{} outcomes on the [0,1] boundary were clipped to (0.001,0.999). Beta regression requires y in (0,1); consider zero/one-inflated beta regression if boundary mass is substantial.",
                n_clip, n
            )),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_numpy_randomstate42() {
        // First 10 randint(0,30) draws from numpy RandomState(42).
        let mut rng = Mt19937::new(42);
        let draws: Vec<usize> = (0..10).map(|_| rng.randint(30)).collect();
        assert_eq!(draws, vec![6, 19, 28, 14, 10, 7, 28, 20, 6, 25]);
    }
}
