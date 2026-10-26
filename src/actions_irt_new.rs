// New IRT model fits + EAP/MAP scoring for the Rust port.
// Mirrors the Python original (tools/irt_tool.py, girth 0.8.0) in inputs
// and output keys; see the header notes per action for girth quirks.
//
// Quadrature: girth does NOT use Gauss-Hermite. All girth MML/EAP paths use
// Gauss-Legendre nodes, 41 points on [-4.5, 4.5], with a standard-normal
// prior density times the Legendre weights as integration weights. This
// module uses exactly that grid, so marginal log-likelihoods and EAP
// abilities are directly comparable with girth's.
//
// Estimator: Bock-Aitkin EM on the marginal likelihood. E-step posteriors
// over the 41 nodes; M-step Fisher scoring per item (full multinomial
// Fisher matrix, line-searched, parameters clipped to girth's bounds:
// discrimination [0.2, 5] for 2PL/GRM, [0.25, 4] for 3PL/PCM,
// difficulty/thresholds [-6, 6], guessing [0, 0.33]).
// Empirically (see verification notes) this lands on the same optimum as
// girth's alternating integral-equation / SLSQP scheme for 2PL, GRM and
// PCM to ~1e-3, and at a higher marginal LL than girth for 3PL (see
// irt_3pl note). Rasch JML is a separate, faithful port of girth's
// _jml_abstract (theta recentered, unweighted over unique patterns).
//
// Output keys mirror Python's _build_model_out exactly:
//   rasch: n_items, n_people, d, disc, method   (no ability, no AIC/BIC)
//   2pl/grm/pcm: n_items, n_people, d, disc, a (EAP), aic, bic
//   3pl: n_items, n_people, d, disc, g          (no ability, no AIC/BIC)
// AIC/BIC quirk (girth): with estimate_distribution=False, girth's
// compute_metrics is called with k_params = 0 for both null and final
// models, so aic = bic = -2*LL and delta = 0. That is reproduced here
// deliberately so the numbers match the Python tool; it is NOT a
// penalised AIC/BIC. "ll" is intentionally not added, to keep the JSON
// identical to the Python tool's keys.

use crate::actions_med;
use crate::{r6, Cell, Table};
use serde_json::{json, Value};

// ---------- quadrature (Gauss-Legendre, n=41, on [-4.5, 4.5]) ----------

fn legendre_pn(n: usize, x: f64) -> (f64, f64) {
    // Returns (P_n(x), P_n'(x)) by recurrence.
    let mut p0 = 1.0;
    let mut p1 = x;
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }
    for k in 2..=n {
        let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = p2;
    }
    // P_n' = n (x P_n - P_{n-1}) / (x^2 - 1)
    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

/// (theta nodes, prior*weight) for girth's grid: GL-41 on [-4.5, 4.5],
/// weights = phi(theta) * legendre_weight * 4.5.
pub fn girth_quadrature() -> (Vec<f64>, Vec<f64>) {
    let n = 41usize;
    let mut theta = Vec::with_capacity(n);
    let mut qw = Vec::with_capacity(n);
    for i in 1..=n {
        // Standard initial guess for the i-th root.
        let mut x = (std::f64::consts::PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = legendre_pn(n, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1e-14 {
                break;
            }
        }
        let (_, dp) = legendre_pn(n, x);
        let w_leg = 2.0 / ((1.0 - x * x) * dp * dp);
        let th = 4.5 * x;
        let w = 4.5 * w_leg;
        let phi = (-0.5 * th * th).exp() / (2.0 * std::f64::consts::PI).sqrt();
        theta.push(th);
        qw.push(phi * w);
    }
    // Roots come out descending (cos decreases); sort ascending to match
    // girth's node order (ascending from -4.5).
    let mut pairs: Vec<(f64, f64)> = theta.into_iter().zip(qw.into_iter()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    (pairs.iter().map(|p| p.0).collect(), pairs.iter().map(|p| p.1).collect())
}

pub fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

// Golden-section minimisation on [a, b] (mirrors scipy fminbound's
// bracketed search to ~1e-5 in x; used for Rasch JML steps).
pub fn golden_min<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64) -> f64 {
    let gr = (5.0f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - gr * (b - a), a + gr * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..300 {
        if (b - a).abs() < tol {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

// Small dense linear solve (Gaussian elimination, partial pivoting).
fn solve_lin(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-14 {
            return None;
        }
        m.swap(col, piv);
        x.swap(col, piv);
        for r in col + 1..n {
            let f = m[r][col] / m[col][col];
            for c in col..n {
                m[r][c] -= f * m[col][c];
            }
            x[r] -= f * x[col];
        }
    }
    let mut out = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = x[r];
        for c in r + 1..n {
            s -= m[r][c] * out[c];
        }
        out[r] = s / m[r][r];
    }
    Some(out)
}

// Fisher-scoring step for one item's parameter vector. `exp_ll`,
// `grad_fisher` are closures over the fixed posterior weights. Returns
// the new parameter vector (old vector if no improving step found).
//
// Active-set handling of box bounds: a plain joint step followed by
// coordinate-wise clipping is WRONG when a coordinate is pinned at a
// bound — the other coordinates' step components were computed assuming
// the pinned coordinate moves, so clipping leaves them pointing the
// wrong way and the line search rejects everything (EM stalls at a
// non-stationary point; seen on SMALL 2PL with discrimination at 0.2).
// Instead, coordinates at a bound whose gradient pushes outward are
// fixed, the reduced system is re-solved for the free coordinates, and
// any coordinate whose step would newly cross a bound is fixed in turn.
fn fisher_step<F, G>(params: &[f64], lo: &[f64], hi: &[f64], exp_ll: F, grad_fisher: G) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> (Vec<f64>, Vec<Vec<f64>>),
{
    let cur = exp_ll(params);
    if !cur.is_finite() {
        return params.to_vec();
    }
    let (grad, fisher) = grad_fisher(params);
    let d = params.len();
    let mut fixed = vec![false; d];
    for i in 0..d {
        if (params[i] <= lo[i] && grad[i] < 0.0) || (params[i] >= hi[i] && grad[i] > 0.0) {
            fixed[i] = true;
        }
    }
    let mut step = vec![0.0; d];
    loop {
        let free: Vec<usize> = (0..d).filter(|i| !fixed[*i]).collect();
        if free.is_empty() {
            break;
        }
        let mat: Vec<Vec<f64>> = free
            .iter()
            .map(|&r| {
                free.iter()
                    .map(|&c| fisher[r][c] + if r == c { 1e-8 } else { 0.0 })
                    .collect()
            })
            .collect();
        let rhs: Vec<f64> = free.iter().map(|&r| grad[r]).collect();
        let sol = match solve_lin(&mat, &rhs) {
            Some(s) => s,
            None => free
                .iter()
                .map(|&r| grad[r] / (fisher[r][r].abs() + 1.0))
                .collect(),
        };
        for (k, &i) in free.iter().enumerate() {
            step[i] = sol[k];
        }
        let mut violated: Option<usize> = None;
        for i in 0..d {
            if !fixed[i] {
                let cand = params[i] + step[i];
                if cand < lo[i] || cand > hi[i] {
                    violated = Some(i);
                    break;
                }
            }
        }
        match violated {
            Some(i) => {
                // Step exactly onto the bound, then re-solve the rest.
                step[i] = if params[i] + step[i] < lo[i] {
                    lo[i] - params[i]
                } else {
                    hi[i] - params[i]
                };
                fixed[i] = true;
            }
            None => break,
        }
    }
    if step.iter().all(|s| *s == 0.0) || step.iter().any(|s| !s.is_finite()) {
        return params.to_vec();
    }
    // Cap the step: with weak items the Fisher step can be enormous
    // (tiny effective information), and then even the smallest line
    // search fraction overshoots. Scale so no coordinate moves > 1.0.
    let max_step = step.iter().map(|s| s.abs()).fold(0.0, f64::max);
    if max_step > 1.0 {
        for s in step.iter_mut() {
            *s /= max_step;
        }
    }
    for alpha in [
        1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        0.001953125, 0.0009765625, 0.00048828125, 0.000244140625,
    ] {
        let cand: Vec<f64> = params
            .iter()
            .zip(step.iter())
            .enumerate()
            .map(|(i, (p, s))| (p + alpha * s).clamp(lo[i], hi[i]))
            .collect();
        let v = exp_ll(&cand);
        if v.is_finite() && v > cur + 1e-12 {
            return cand;
        }
    }
    params.to_vec()
}

// ---------- CSV parsing ----------

// Binary matrix: rows = people, cols = items. Listwise deletion of rows
// with any non-finite cell; global 1/2 -> 0/1 conversion (Python
// _validate_binary); anything else non-0/1 is an error (girth would fail
// on it too).
fn binary_matrix(tab: &Table) -> Result<Vec<Vec<f64>>, String> {
    let n_items = tab.headers.len();
    if n_items == 0 {
        return Err("no columns".into());
    }
    let mut data: Vec<Vec<f64>> = Vec::new();
    for row in &tab.rows {
        let mut r = Vec::new();
        let mut ok = true;
        for c in row.iter().take(n_items) {
            match c {
                Cell::Num(x) if x.is_finite() => r.push(x.trunc()),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            data.push(r);
        }
    }
    if data.is_empty() {
        return Err("no complete cases after dropping missing values".into());
    }
    let mut uniq: Vec<f64> = data.iter().flatten().cloned().collect();
    uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
    uniq.dedup();
    if uniq.len() == 2 && uniq[0] == 1.0 && uniq[1] == 2.0 {
        for r in data.iter_mut() {
            for x in r.iter_mut() {
                *x -= 1.0;
            }
        }
    }
    for r in &data {
        for x in r {
            if *x != 0.0 && *x != 1.0 {
                return Err(format!(
                    "binary response values must be 0/1 (or 1/2), got {}",
                    x
                ));
            }
        }
    }
    Ok(data)
}

// Polytomous matrix with girth's per-item rank recoding
// (condition_polytomous_response, trim_ends=False): each item's observed
// values are mapped to 0..K-1 by sorted rank; a single-valued item maps
// to 0 if it equals the global minimum, else 1, with K = 2 (girth's
// single-value branch). Returns (recoded rows, K per item).
fn poly_matrix(tab: &Table) -> Result<(Vec<Vec<usize>>, Vec<usize>), String> {
    let n_items = tab.headers.len();
    if n_items == 0 {
        return Err("no columns".into());
    }
    let mut raw: Vec<Vec<f64>> = Vec::new();
    for row in &tab.rows {
        let mut r = Vec::new();
        let mut ok = true;
        for c in row.iter().take(n_items) {
            match c {
                Cell::Num(x) if x.is_finite() => r.push(x.trunc()),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            raw.push(r);
        }
    }
    if raw.is_empty() {
        return Err("no complete cases after dropping missing values".into());
    }
    let gmin = raw.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
    let mut ks = Vec::new();
    let mut maps: Vec<Vec<f64>> = Vec::new();
    for j in 0..n_items {
        let mut vals: Vec<f64> = raw.iter().map(|r| r[j]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();
        if vals.len() == 1 {
            ks.push(2usize);
            maps.push(vals);
        } else {
            ks.push(vals.len());
            maps.push(vals);
        }
    }
    let mut out = Vec::new();
    for r in &raw {
        let mut row = Vec::new();
        for j in 0..n_items {
            let vals = &maps[j];
            if vals.len() == 1 {
                row.push(if (r[j] - gmin).abs() < 1e-9 { 0 } else { 1 });
            } else {
                row.push(vals.iter().position(|v| *v == r[j]).unwrap());
            }
        }
        out.push(row);
    }
    Ok((out, ks))
}

// Group rows into unique patterns with counts (EM works on patterns).
fn group_patterns<T: Clone + PartialEq>(rows: &[Vec<T>]) -> (Vec<Vec<T>>, Vec<f64>) {
    let mut pats: Vec<Vec<T>> = Vec::new();
    let mut counts: Vec<f64> = Vec::new();
    for r in rows {
        if let Some(i) = pats.iter().position(|p| p == r) {
            counts[i] += 1.0;
        } else {
            pats.push(r.clone());
            counts.push(1.0);
        }
    }
    (pats, counts)
}

// ---------- item response functions ----------

// Dichotomous: returns (P(correct), dP/d(a,b,g)).
fn dich_p_grad(a: f64, b: f64, g: f64, th: f64) -> (f64, [f64; 3]) {
    let q = expit(a * (th - b));
    let p = g + (1.0 - g) * q;
    let dq = q * (1.0 - q);
    let dp = [
        (1.0 - g) * dq * (th - b),
        -(1.0 - g) * dq * a,
        1.0 - q,
    ];
    (p, dp)
}

// GRM category probabilities + gradients wrt (a, b_1..b_{K-1}).
// grads[k] is d log P_k / d params (length K).
fn grm_probs_grads(a: f64, b: &[f64], th: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = b.len() + 1;
    // S_0 = 1, S_k = expit(a(th-b_k)) for 1..K-1, S_K = 0.
    let mut s = vec![0.0; k + 1];
    s[0] = 1.0;
    for (i, bb) in b.iter().enumerate() {
        s[i + 1] = expit(a * (th - bb));
    }
    s[k] = 0.0;
    let mut probs = Vec::with_capacity(k);
    for cat in 0..k {
        probs.push((s[cat] - s[cat + 1]).max(0.0));
    }
    // dS_m/d params: params are (a, b_1..b_{K-1}); S_m depends on a and
    // on b_m (for 1<=m<=K-1) only.
    let mut grads = Vec::with_capacity(k);
    for cat in 0..k {
        let mut dp = vec![0.0; k]; // index 0 = a, index m = b_m
        for m in [cat, cat + 1] {
            if m >= 1 && m <= k - 1 {
                let sm = s[m];
                let dsa = sm * (1.0 - sm) * (th - b[m - 1]);
                let dsb = -a * sm * (1.0 - sm);
                let sign = if m == cat { 1.0 } else { -1.0 };
                dp[0] += sign * dsa;
                dp[m] += sign * dsb;
            }
        }
        let p = probs[cat].max(1e-12);
        grads.push(dp.iter().map(|v| v / p).collect());
    }
    (probs, grads)
}

// PCM category probabilities + gradients wrt (a, d_1..d_{K-1}).
// Step difficulties d_s for categories s>=1; category 0 numerator is 1.
fn pcm_probs_grads(a: f64, d: &[f64], th: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = d.len() + 1;
    // A_0 = 0; A_cat = sum_{s=1..cat} a (th - d_s).
    let mut avals = vec![0.0; k];
    let mut cvals = vec![0.0; k]; // dA_cat/da
    for cat in 1..k {
        avals[cat] = avals[cat - 1] + a * (th - d[cat - 1]);
        cvals[cat] = cvals[cat - 1] + (th - d[cat - 1]);
    }
    let mx = avals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = avals.iter().map(|v| (v - mx).exp()).collect();
    let tot: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|e| e / tot).collect();
    // E[C] and E[1(cat>=s)] under probs.
    let ec: f64 = probs.iter().zip(cvals.iter()).map(|(p, c)| p * c).sum();
    let mut ege = vec![0.0; k]; // ege[s] = P(cat >= s) for s>=1
    for s in 1..k {
        ege[s] = probs.iter().skip(s).sum();
    }
    let mut grads = Vec::with_capacity(k);
    for cat in 0..k {
        let mut g = vec![0.0; k];
        g[0] = cvals[cat] - ec;
        for s in 1..k {
            let ind = if cat >= s { 1.0 } else { 0.0 };
            g[s] = -a * (ind - ege[s]);
        }
        grads.push(g);
    }
    (probs, grads)
}

// ---------- dichotomous EM ----------

#[derive(Clone, Copy, PartialEq)]
enum DichKind {
    RaschMml, // common discrimination, estimated in [0.25, 10]
    TwoPl,    // per-item a in [0.2, 5], g = 0
    ThreePl,  // per-item a in [0.25, 4], g in [0, 0.33]
}

struct DichFit {
    a: Vec<f64>,
    b: Vec<f64>,
    g: Vec<f64>,
    ll: f64,
}

fn fit_dich(pats: &[Vec<f64>], counts: &[f64], kind: DichKind, a_init: f64, g_init: f64) -> DichFit {
    let (theta, qw) = girth_quadrature();
    let q = theta.len();
    let j = pats[0].len();
    // Initial values: a = a_init, b from mml_approx with a = 1, g = g_init
    // (3PL only; other kinds keep g = 0 throughout).
    let mut a = vec![a_init; j];
    let mut b = vec![0.0; j];
    let mut g = vec![if kind == DichKind::ThreePl { g_init } else { 0.0 }; j];
    for jj in 0..j {
        let mut tot = 0.0;
        let mut yes = 0.0;
        for (pi, pat) in pats.iter().enumerate() {
            tot += counts[pi];
            yes += counts[pi] * pat[jj];
        }
        let p = if tot > 0.0 { yes / tot } else { 0.5 };
        let pc = p.clamp(1e-6, 1.0 - 1e-6);
        b[jj] = ((1.0f64 + 1.0 / 3.0).sqrt() * ((1.0 - pc) / pc).ln()).clamp(-6.0, 6.0);
    }
    let (alo, ahi) = match kind {
        DichKind::RaschMml => (0.25, 10.0),
        DichKind::TwoPl => (0.2, 5.0),
        DichKind::ThreePl => (0.25, 4.0),
    };
    let mut prev_ll = f64::NEG_INFINITY;
    for _iter in 0..2000 {
        // E-step: posterior over nodes per pattern, and marginal LL.
        let mut post: Vec<Vec<f64>> = Vec::with_capacity(pats.len());
        let mut ll = 0.0;
        for (pi, pat) in pats.iter().enumerate() {
            let mut logl = Vec::with_capacity(q);
            for qi in 0..q {
                let mut s = 0.0;
                for jj in 0..j {
                    let (p, _) = dich_p_grad(a[jj], b[jj], g[jj], theta[qi]);
                    let pc = p.clamp(1e-12, 1.0 - 1e-12);
                    s += if pat[jj] == 1.0 { pc.ln() } else { (1.0 - pc).ln() };
                }
                logl.push(s + qw[qi].ln());
            }
            let lse = logsumexp(&logl);
            ll += counts[pi] * lse;
            post.push(logl.iter().map(|v| (v - lse).exp()).collect());
        }
        // M-step.
        let mut max_change = 0.0f64;
        if kind == DichKind::RaschMml {
            // b_j updates with a fixed, then the common a update.
            let ac = a[0];
            for jj in 0..j {
                let old = b[jj];
                let nb = dich_item_step(jj, pats, counts, &post, &theta, &[ac], &[old], &[0.0], false, false, alo, ahi);
                b[jj] = nb[1]; // nb is (a, b, g); only b moves here
                max_change = max_change.max((b[jj] - old).abs());
            }
            // Common-a step: aggregate grad/fisher over items.
            let old_a = a[0];
            let na = common_a_step(pats, counts, &post, &theta, &b, old_a, alo, ahi);
            for v in a.iter_mut() {
                *v = na;
            }
            max_change = max_change.max((na - old_a).abs());
        } else {
            for jj in 0..j {
                let old = (a[jj], b[jj], g[jj]);
                let with_g = kind == DichKind::ThreePl;
                let nb = dich_item_step(
                    jj, pats, counts, &post, &theta,
                    &[a[jj]], &[b[jj]], &[g[jj]], true, with_g, alo, ahi,
                );
                a[jj] = nb[0];
                b[jj] = nb[1];
                if with_g {
                    g[jj] = nb[2];
                }
                max_change = max_change
                    .max((a[jj] - old.0).abs())
                    .max((b[jj] - old.1).abs())
                    .max((g[jj] - old.2).abs());
            }
        }
        let dll = (ll - prev_ll).abs();
        prev_ll = ll;
        // Tight stop: EM converges linearly, and with weak items (small
        // discrimination) a loose stop leaves parameters ~0.1 from the
        // optimum while the LL is already flat to ~1e-3. Both criteria
        // must hold so neither a flat LL nor a stalled step alone stops
        // the run.
        if dll < 1e-9 && max_change < 1e-6 {
            break;
        }
    }
    // Final marginal LL (recomputed cleanly).
    let mut ll = 0.0;
    for (pi, pat) in pats.iter().enumerate() {
        let mut logl = Vec::with_capacity(q);
        for qi in 0..q {
            let mut s = 0.0;
            for jj in 0..j {
                let (p, _) = dich_p_grad(a[jj], b[jj], g[jj], theta[qi]);
                let pc = p.clamp(1e-12, 1.0 - 1e-12);
                s += if pat[jj] == 1.0 { pc.ln() } else { (1.0 - pc).ln() };
            }
            logl.push(s + qw[qi].ln());
        }
        ll += counts[pi] * logsumexp(&logl);
    }
    DichFit { a, b, g, ll }
}

// Fisher step for one dichotomous item. If `vary_a` is false, a is held
// fixed and only b is updated (Rasch-MML b-step); the parameter vector
// layout is always (a, b, g) with unused entries ignored.
#[allow(clippy::too_many_arguments)]
fn dich_item_step(
    jj: usize,
    pats: &[Vec<f64>],
    counts: &[f64],
    post: &[Vec<f64>],
    theta: &[f64],
    a0: &[f64],
    b0: &[f64],
    g0: &[f64],
    vary_a: bool,
    vary_g: bool,
    alo: f64,
    ahi: f64,
) -> Vec<f64> {
    // Free-parameter index map into (a, b, g).
    let mut idx: Vec<usize> = vec![1];
    if vary_a {
        idx.insert(0, 0);
    }
    if vary_g {
        idx.push(2);
    }
    let base = [a0[0], b0[0], g0[0]];
    let mut cur: Vec<f64> = idx.iter().map(|&i| base[i]).collect();
    let lo: Vec<f64> = idx
        .iter()
        .map(|&i| match i {
            0 => alo,
            1 => -6.0,
            _ => 0.0,
        })
        .collect();
    let hi: Vec<f64> = idx
        .iter()
        .map(|&i| match i {
            0 => ahi,
            1 => 6.0,
            _ => 0.33,
        })
        .collect();
    let expand = |v: &[f64]| -> [f64; 3] {
        let mut full = base;
        for (k, &i) in idx.iter().enumerate() {
            full[i] = v[k];
        }
        full
    };
    let exp_ll = |v: &[f64]| -> f64 {
        let f = expand(v);
        let mut tot = 0.0;
        for (pi, pat) in pats.iter().enumerate() {
            let y = pat[jj];
            for qi in 0..theta.len() {
                let w = counts[pi] * post[pi][qi];
                if w == 0.0 {
                    continue;
                }
                let (p, _) = dich_p_grad(f[0], f[1], f[2], theta[qi]);
                let pc = p.clamp(1e-12, 1.0 - 1e-12);
                tot += w * if y == 1.0 { pc.ln() } else { (1.0 - pc).ln() };
            }
        }
        tot
    };
    let grad_fisher = |v: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
        let f = expand(v);
        let d = idx.len();
        let mut grad = vec![0.0; d];
        let mut fish = vec![vec![0.0; d]; d];
        for (pi, pat) in pats.iter().enumerate() {
            let y = pat[jj];
            for qi in 0..theta.len() {
                let w = counts[pi] * post[pi][qi];
                if w == 0.0 {
                    continue;
                }
                let (p, dp) = dich_p_grad(f[0], f[1], f[2], theta[qi]);
                let pc = p.clamp(1e-12, 1.0 - 1e-12);
                let dlog: Vec<f64> = idx
                    .iter()
                    .map(|&i| {
                        if y == 1.0 {
                            dp[i] / pc
                        } else {
                            -dp[i] / (1.0 - pc)
                        }
                    })
                    .collect();
                for r in 0..d {
                    grad[r] += w * dlog[r];
                }
                // Full Bernoulli Fisher: (dP)(dP)^T / (P(1-P)).
                for r in 0..d {
                    for c in 0..d {
                        fish[r][c] += w * dp[idx[r]] * dp[idx[c]] / (pc * (1.0 - pc));
                    }
                }
            }
        }
        (grad, fish)
    };
    for _ in 0..30 {
        let next = fisher_step(&cur, &lo, &hi, &exp_ll, &grad_fisher);
        let change = next
            .iter()
            .zip(cur.iter())
            .map(|(n, o)| (n - o).abs())
            .fold(0.0, f64::max);
        cur = next;
        if change < 1e-7 {
            break;
        }
    }
    let full = expand(&cur);
    vec![full[0], full[1], full[2]]
}

// Common-discrimination step for Rasch MML (scalar Fisher step on the
// expected log-likelihood summed over items).
fn common_a_step(
    pats: &[Vec<f64>],
    counts: &[f64],
    post: &[Vec<f64>],
    theta: &[f64],
    b: &[f64],
    a0: f64,
    alo: f64,
    ahi: f64,
) -> f64 {
    let j = b.len();
    let exp_ll = |av: &[f64]| -> f64 {
        let a = av[0];
        let mut tot = 0.0;
        for (pi, pat) in pats.iter().enumerate() {
            for jj in 0..j {
                let y = pat[jj];
                for qi in 0..theta.len() {
                    let w = counts[pi] * post[pi][qi];
                    if w == 0.0 {
                        continue;
                    }
                    let (p, _) = dich_p_grad(a, b[jj], 0.0, theta[qi]);
                    let pc = p.clamp(1e-12, 1.0 - 1e-12);
                    tot += w * if y == 1.0 { pc.ln() } else { (1.0 - pc).ln() };
                }
            }
        }
        tot
    };
    let grad_fisher = |av: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
        let a = av[0];
        let mut grad = 0.0;
        let mut fish = 0.0;
        for (pi, pat) in pats.iter().enumerate() {
            for jj in 0..j {
                let y = pat[jj];
                for qi in 0..theta.len() {
                    let w = counts[pi] * post[pi][qi];
                    if w == 0.0 {
                        continue;
                    }
                    let (p, dp) = dich_p_grad(a, b[jj], 0.0, theta[qi]);
                    let pc = p.clamp(1e-12, 1.0 - 1e-12);
                    let dlog = if y == 1.0 { dp[0] / pc } else { -dp[0] / (1.0 - pc) };
                    grad += w * dlog;
                    fish += w * dp[0] * dp[0] / (pc * (1.0 - pc));
                }
            }
        }
        (vec![grad], vec![vec![fish]])
    };
    let mut cur = vec![a0];
    for _ in 0..30 {
        let next = fisher_step(&cur, &[alo], &[ahi], &exp_ll, &grad_fisher);
        let change = (next[0] - cur[0]).abs();
        cur = next;
        if change < 1e-7 {
            break;
        }
    }
    cur[0]
}

// ---------- polytomous EM (GRM / PCM) ----------

#[derive(Clone, Copy, PartialEq)]
enum PolyKind {
    Grm,
    Pcm,
}

struct PolyFit {
    a: Vec<f64>,
    // GRM: thresholds (ascending); PCM: step difficulties.
    thr: Vec<Vec<f64>>,
    ll: f64,
}

fn fit_poly(pats: &[Vec<usize>], counts: &[f64], ks: &[usize], kind: PolyKind) -> PolyFit {
    let (theta, qw) = girth_quadrature();
    let q = theta.len();
    let j = pats[0].len();
    let (alo, ahi) = match kind {
        PolyKind::Grm => (0.2, 5.0),
        PolyKind::Pcm => (0.25, 4.0),
    };
    // Initial values.
    let mut a = vec![1.0; j];
    let mut thr: Vec<Vec<f64>> = Vec::new();
    for jj in 0..j {
        let k = ks[jj];
        if kind == PolyKind::Grm {
            // Thresholds from marginal cumulative proportions.
            let mut tot = 0.0;
            let mut cum = vec![0.0; k];
            for (pi, pat) in pats.iter().enumerate() {
                tot += counts[pi];
                for cat in 0..k {
                    if pat[jj] >= cat {
                        cum[cat] += counts[pi];
                    }
                }
            }
            let mut b = Vec::new();
            for cat in 1..k {
                let p = (cum[cat] / tot).clamp(0.02, 0.98);
                b.push((-(p / (1.0 - p)).ln()).clamp(-4.0, 4.0));
            }
            // Cumulative proportions decrease, so -logit increases; sort
            // defensively and enforce a small gap.
            b.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for t in 1..b.len() {
                if b[t] < b[t - 1] + 0.05 {
                    b[t] = b[t - 1] + 0.05;
                }
            }
            thr.push(b);
        } else {
            // Girth's PCM init: linspace(-1, 1, K-1).
            let mut d = Vec::new();
            for t in 0..k - 1 {
                d.push(if k - 1 == 1 { 0.0 } else { -1.0 + 2.0 * t as f64 / (k - 2) as f64 });
            }
            thr.push(d);
        }
    }
    let mut prev_ll = f64::NEG_INFINITY;
    for _iter in 0..2000 {
        // E-step.
        let mut post: Vec<Vec<f64>> = Vec::with_capacity(pats.len());
        let mut ll = 0.0;
        for (pi, pat) in pats.iter().enumerate() {
            let mut logl = Vec::with_capacity(q);
            for qi in 0..q {
                let mut s = 0.0;
                for jj in 0..j {
                    let (probs, _) = match kind {
                        PolyKind::Grm => grm_probs_grads(a[jj], &thr[jj], theta[qi]),
                        PolyKind::Pcm => pcm_probs_grads(a[jj], &thr[jj], theta[qi]),
                    };
                    s += probs[pat[jj]].max(1e-12).ln();
                }
                logl.push(s + qw[qi].ln());
            }
            let lse = logsumexp(&logl);
            ll += counts[pi] * lse;
            post.push(logl.iter().map(|v| (v - lse).exp()).collect());
        }
        // M-step per item.
        let mut max_change = 0.0f64;
        for jj in 0..j {
            let old_a = a[jj];
            let old_thr = thr[jj].clone();
            let (na, nthr) = poly_item_step(
                jj, pats, counts, &post, &theta, a[jj], &thr[jj], kind, alo, ahi,
            );
            a[jj] = na;
            thr[jj] = nthr;
            max_change = max_change.max((a[jj] - old_a).abs());
            for (n, o) in thr[jj].iter().zip(old_thr.iter()) {
                max_change = max_change.max((n - o).abs());
            }
        }
        let dll = (ll - prev_ll).abs();
        prev_ll = ll;
        if dll < 1e-9 && max_change < 1e-6 {
            break;
        }
    }
    // Final marginal LL.
    let mut ll = 0.0;
    for (pi, pat) in pats.iter().enumerate() {
        let mut logl = Vec::with_capacity(q);
        for qi in 0..q {
            let mut s = 0.0;
            for jj in 0..j {
                let (probs, _) = match kind {
                    PolyKind::Grm => grm_probs_grads(a[jj], &thr[jj], theta[qi]),
                    PolyKind::Pcm => pcm_probs_grads(a[jj], &thr[jj], theta[qi]),
                };
                s += probs[pat[jj]].max(1e-12).ln();
            }
            logl.push(s + qw[qi].ln());
        }
        ll += counts[pi] * logsumexp(&logl);
    }
    PolyFit { a, thr, ll }
}

// One item's M-step. For GRM the Fisher step runs in t-space
// (t_1 = b_1, t_m = log(b_m - b_{m-1})) so thresholds stay ordered.
#[allow(clippy::too_many_arguments)]
fn poly_item_step(
    jj: usize,
    pats: &[Vec<usize>],
    counts: &[f64],
    post: &[Vec<f64>],
    theta: &[f64],
    a0: f64,
    thr0: &[f64],
    kind: PolyKind,
    alo: f64,
    ahi: f64,
) -> (f64, Vec<f64>) {
    let k = thr0.len() + 1; // categories
    let d = k; // params: a + (K-1) thresholds/steps
    // probs/grads in "b-space" (a, thresholds/steps) for one (pattern, node).
    let bg = |a: f64, t: &[f64], th: f64| -> (Vec<f64>, Vec<Vec<f64>>) {
        match kind {
            PolyKind::Grm => grm_probs_grads(a, t, th),
            PolyKind::Pcm => pcm_probs_grads(a, t, th),
        }
    };
    // t-space <-> b-space conversion (GRM only; PCM is identity).
    let to_b = |u: &[f64]| -> Vec<f64> {
        if kind == PolyKind::Pcm {
            return u[1..].to_vec();
        }
        let mut b = vec![0.0; k - 1];
        b[0] = u[1];
        for m in 1..k - 1 {
            b[m] = b[m - 1] + u[m + 1].exp();
        }
        b
    };
    let to_u = |a: f64, b: &[f64]| -> Vec<f64> {
        let mut u = vec![a, b[0]];
        for m in 1..k - 1 {
            u.push((b[m] - b[m - 1]).max(1e-6).ln());
        }
        u
    };
    let mut cur_u = to_u(a0, thr0);
    let lo_u: Vec<f64> = {
        let mut v = vec![alo, -6.0];
        for _ in 1..k - 1 {
            v.push(-9.0);
        }
        v
    };
    let hi_u: Vec<f64> = {
        let mut v = vec![ahi, 6.0];
        for _ in 1..k - 1 {
            v.push(3.0);
        }
        v
    };
    // Jacobian db/du (rows: b-space params (a, b_1..), cols: u-space).
    // For PCM this is the identity.
    let jacobian = |u: &[f64]| -> Vec<Vec<f64>> {
        let mut jac = vec![vec![0.0; d]; d];
        jac[0][0] = 1.0;
        if kind == PolyKind::Pcm {
            for i in 1..d {
                jac[i][i] = 1.0;
            }
            return jac;
        }
        for r in 1..d {
            jac[r][1] = 1.0; // db_r/dt_1 = 1
            for c in 2..d {
                if c <= r {
                    jac[r][c] = u[c].exp();
                }
            }
        }
        jac
    };
    let exp_ll_u = |u: &[f64]| -> f64 {
        let b = to_b(u);
        // Enforce the [-6, 6] box in b-space (girth's bound).
        let bc: Vec<f64> = b.iter().map(|v| v.clamp(-6.0, 6.0)).collect();
        let mut tot = 0.0;
        for (pi, pat) in pats.iter().enumerate() {
            let y = pat[jj];
            for qi in 0..theta.len() {
                let w = counts[pi] * post[pi][qi];
                if w == 0.0 {
                    continue;
                }
                let (probs, _) = bg(u[0], &bc, theta[qi]);
                tot += w * probs[y].max(1e-12).ln();
            }
        }
        tot
    };
    let grad_fisher_u = |u: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
        let b = to_b(u);
        let bc: Vec<f64> = b.iter().map(|v| v.clamp(-6.0, 6.0)).collect();
        let jac = jacobian(u);
        let mut grad_b = vec![0.0; d];
        let mut fish_b = vec![vec![0.0; d]; d];
        for (pi, pat) in pats.iter().enumerate() {
            let y = pat[jj];
            for qi in 0..theta.len() {
                let w = counts[pi] * post[pi][qi];
                if w == 0.0 {
                    continue;
                }
                let (probs, grads) = bg(u[0], &bc, theta[qi]);
                for r in 0..d {
                    grad_b[r] += w * grads[y][r];
                }
                for cat in 0..k {
                    let p = probs[cat];
                    if p <= 0.0 {
                        continue;
                    }
                    for r in 0..d {
                        for c in 0..d {
                            fish_b[r][c] += w * p * grads[cat][r] * grads[cat][c];
                        }
                    }
                }
            }
        }
        // Transform to u-space: grad_u = J^T grad_b, fish_u = J^T fish_b J.
        let mut grad_u = vec![0.0; d];
        for c in 0..d {
            for r in 0..d {
                grad_u[c] += jac[r][c] * grad_b[r];
            }
        }
        let mut fish_u = vec![vec![0.0; d]; d];
        for r in 0..d {
            for c in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    for j2 in 0..d {
                        s += jac[i][r] * fish_b[i][j2] * jac[j2][c];
                    }
                }
                fish_u[r][c] = s;
            }
        }
        (grad_u, fish_u)
    };
    for _ in 0..30 {
        let next = fisher_step(&cur_u, &lo_u, &hi_u, &exp_ll_u, &grad_fisher_u);
        let change = next
            .iter()
            .zip(cur_u.iter())
            .map(|(n, o)| (n - o).abs())
            .fold(0.0, f64::max);
        cur_u = next;
        if change < 1e-7 {
            break;
        }
    }
    let mut b = to_b(&cur_u);
    // Final box + ordering cleanup in b-space.
    for v in b.iter_mut() {
        *v = v.clamp(-6.0, 6.0);
    }
    if kind == PolyKind::Grm {
        for t in 1..b.len() {
            if b[t] < b[t - 1] + 1e-4 {
                b[t] = b[t - 1] + 1e-4;
            }
        }
        if let Some(last) = b.last_mut() {
            if *last > 6.0 {
                *last = 6.0;
            }
        }
    }
    (cur_u[0].clamp(alo, ahi), b)
}

// ---------- EAP abilities (any fitted model) ----------

fn eap_for_rows_dich(rows: &[Vec<f64>], a: &[f64], b: &[f64], g: &[f64]) -> Vec<f64> {
    let (theta, qw) = girth_quadrature();
    rows.iter()
        .map(|row| {
            let mut num = 0.0;
            let mut den = 0.0;
            for qi in 0..theta.len() {
                let mut s = 0.0;
                for jj in 0..row.len() {
                    let (p, _) = dich_p_grad(a[jj], b[jj], g[jj], theta[qi]);
                    let pc = p.clamp(1e-12, 1.0 - 1e-12);
                    s += if row[jj] == 1.0 { pc.ln() } else { (1.0 - pc).ln() };
                }
                let w = (s + qw[qi].ln()).exp();
                num += theta[qi] * w;
                den += w;
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        })
        .collect()
}

fn eap_for_rows_poly(
    rows: &[Vec<usize>],
    a: &[f64],
    thr: &[Vec<f64>],
    kind: PolyKind,
) -> Vec<f64> {
    let (theta, qw) = girth_quadrature();
    rows.iter()
        .map(|row| {
            let mut num = 0.0;
            let mut den = 0.0;
            for qi in 0..theta.len() {
                let mut s = 0.0;
                for jj in 0..row.len() {
                    let (probs, _) = match kind {
                        PolyKind::Grm => grm_probs_grads(a[jj], &thr[jj], theta[qi]),
                        PolyKind::Pcm => pcm_probs_grads(a[jj], &thr[jj], theta[qi]),
                    };
                    s += probs[row[jj]].max(1e-12).ln();
                }
                let w = (s + qw[qi].ln()).exp();
                num += theta[qi] * w;
                den += w;
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        })
        .collect()
}

// ---------- Rasch JML (faithful port of girth _jml_abstract) ----------

fn rasch_jml(data: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>), String> {
    let j = data[0].len();
    // Unique patterns + counts; trim zero-variance patterns.
    let (pats, counts) = group_patterns(data);
    let mut tpats: Vec<Vec<f64>> = Vec::new();
    let mut tcounts: Vec<f64> = Vec::new();
    for (i, p) in pats.iter().enumerate() {
        let var = p.iter().any(|&y| y != p[0]);
        if var {
            tpats.push(p.clone());
            tcounts.push(counts[i]);
        }
    }
    if tpats.is_empty() {
        return Err("no response patterns with variance (all respondents answered all items the same)".into());
    }
    // Seed: mml_approx with discrimination 1.
    let mut betas = vec![0.0; j];
    for jj in 0..j {
        let mut tot = 0.0;
        let mut yes = 0.0;
        for (i, p) in pats.iter().enumerate() {
            tot += counts[i];
            yes += counts[i] * p[jj];
        }
        let no = tot - yes;
        let scalar = if yes > 0.0 && no > 0.0 {
            (no / yes).ln()
        } else if yes == 0.0 {
            6.0
        } else {
            -6.0
        };
        betas[jj] = ((1.0f64 + 1.0 / 3.0).sqrt() * scalar).clamp(-6.0, 6.0);
    }
    // sign: y=1 -> -1, y=0 -> +1.
    let sign = |y: f64| if y == 1.0 { -1.0 } else { 1.0 };
    for _ in 0..25 {
        let prev = betas.clone();
        // Step 1: theta per trimmed pattern (unweighted recenter below).
        let mut thetas = Vec::with_capacity(tpats.len());
        for pat in &tpats {
            let obj = |th: f64| -> f64 {
                pat.iter()
                    .enumerate()
                    .map(|(jj, &y)| (sign(y) * (th - betas[jj])).exp().ln_1p())
                    .sum()
            };
            thetas.push(golden_min(obj, -6.0, 6.0, 1e-5));
        }
        // Recenter: unweighted mean / sd(ddof=1) over unique patterns.
        let m = thetas.iter().sum::<f64>() / thetas.len() as f64;
        let sd = if thetas.len() > 1 {
            (thetas.iter().map(|t| (t - m).powi(2)).sum::<f64>() / (thetas.len() - 1) as f64).sqrt()
        } else {
            1.0
        };
        if sd > 0.0 {
            for t in thetas.iter_mut() {
                *t = (*t - m) / sd;
            }
        } else {
            for t in thetas.iter_mut() {
                *t -= m;
            }
        }
        // Step 2: beta per item (count-weighted over trimmed patterns).
        for jj in 0..j {
            let obj = |beta: f64| -> f64 {
                tpats
                    .iter()
                    .enumerate()
                    .map(|(i, pat)| {
                        tcounts[i] * (sign(pat[jj]) * (thetas[i] - beta)).exp().ln_1p()
                    })
                    .sum()
            };
            betas[jj] = golden_min(obj, -6.0, 6.0, 1e-5);
        }
        if betas
            .iter()
            .zip(prev.iter())
            .map(|(n, o)| (n - o).abs())
            .fold(0.0, f64::max)
            < 1e-3
        {
            break;
        }
    }
    Ok((vec![1.0; j], betas))
}

// ---------- output helpers ----------

fn aic_bic(ll: f64) -> Value {
    // Girth quirk: k_params = 0 for null and final alike.
    let v = -2.0 * ll;
    json!({"final": r6(v), "null": r6(v), "delta": r6(0.0)})
}

// ---------- actions ----------

pub fn irt_rasch(tab: &Table, params: &serde_json::Map<String, Value>) -> Result<Value, String> {
    let method = params
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("jml")
        .to_string();
    let data = binary_matrix(tab)?;
    let n_items = data[0].len();
    let n_people = data.len();
    if method == "mml" {
        let (pats, counts) = group_patterns(&data);
        let fit = fit_dich(&pats, &counts, DichKind::RaschMml, 1.0, 0.0);
        // onepl_mml returns a scalar discrimination; Python broadcasts it.
        let disc = fit.a[0];
        Ok(json!({
            "n_items": n_items, "n_people": n_people,
            "d": fit.b.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
            "disc": vec![r6(disc); n_items],
            "method": "mml",
        }))
    } else {
        let (disc, betas) = rasch_jml(&data)?;
        Ok(json!({
            "n_items": n_items, "n_people": n_people,
            "d": betas.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
            "disc": disc.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
            "method": "jml",
        }))
    }
}

pub fn irt_2pl(tab: &Table) -> Result<Value, String> {
    let data = binary_matrix(tab)?;
    let n_items = data[0].len();
    let n_people = data.len();
    let (pats, counts) = group_patterns(&data);
    let fit = fit_dich(&pats, &counts, DichKind::TwoPl, 1.0, 0.0);
    let abilities = eap_for_rows_dich(&data, &fit.a, &fit.b, &fit.g);
    Ok(json!({
        "n_items": n_items, "n_people": n_people,
        "d": fit.b.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
        "disc": fit.a.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
        "a": abilities.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
        "aic": aic_bic(fit.ll), "bic": aic_bic(fit.ll),
    }))
}

pub fn irt_3pl(tab: &Table) -> Result<Value, String> {
    let data = binary_matrix(tab)?;
    let n_items = data[0].len();
    let n_people = data.len();
    let (pats, counts) = group_patterns(&data);
    // 3PL marginal likelihood is multimodal and guessing is weakly
    // identified (girth's SLSQP run from (a=1, g=0) lands on one local
    // optimum). Run EM from several starts and keep the highest LL;
    // this is the standard guard against a poor local optimum, not a
    // fit to girth's path — the reported LL is the arbiter.
    let mut best: Option<DichFit> = None;
    for (a_init, g_init) in [(1.0, 0.0), (1.0, 0.15), (0.7, 0.1), (1.5, 0.2)] {
        let fit = fit_dich(&pats, &counts, DichKind::ThreePl, a_init, g_init);
        let better = match &best {
            None => true,
            Some(b) => fit.ll > b.ll,
        };
        if better {
            best = Some(fit);
        }
    }
    let fit = best.unwrap();
    Ok(json!({
        "n_items": n_items, "n_people": n_people,
        "d": fit.b.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
        "disc": fit.a.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
        "g": fit.g.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
    }))
}

fn poly_out(
    data: &[Vec<usize>],
    ks: &[usize],
    fit: &PolyFit,
    kind: PolyKind,
) -> Value {
    let n_items = data[0].len();
    let n_people = data.len();
    let maxk = *ks.iter().max().unwrap_or(&1);
    let mut d: Vec<Value> = Vec::new();
    for jj in 0..n_items {
        let mut row: Vec<Value> = Vec::new();
        for t in 0..maxk - 1 {
            if t < fit.thr[jj].len() {
                row.push(r6(fit.thr[jj][t]));
            } else {
                row.push(Value::Null);
            }
        }
        d.push(Value::Array(row));
    }
    let abilities = eap_for_rows_poly(data, &fit.a, &fit.thr, kind);
    json!({
        "n_items": n_items, "n_people": n_people,
        "d": d,
        "disc": fit.a.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
        "a": abilities.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
        "aic": aic_bic(fit.ll), "bic": aic_bic(fit.ll),
    })
}

pub fn irt_grm(tab: &Table) -> Result<Value, String> {
    let (data, ks) = poly_matrix(tab)?;
    let (pats, counts) = group_patterns(&data);
    let fit = fit_poly(&pats, &counts, &ks, PolyKind::Grm);
    Ok(poly_out(&data, &ks, &fit, PolyKind::Grm))
}

pub fn irt_pcm(tab: &Table) -> Result<Value, String> {
    let (data, ks) = poly_matrix(tab)?;
    let (pats, counts) = group_patterns(&data);
    let fit = fit_poly(&pats, &counts, &ks, PolyKind::Pcm);
    Ok(poly_out(&data, &ks, &fit, PolyKind::Pcm))
}

// ---------- irt_score: mle (delegated) + eap / map ----------

// Parse diff/disc/guess lists the way irt_score does; returns
// (diff, disc, guess, model, method) with Python's normalisation:
// unknown method -> mle; 3PL scoring only when model == "3pl" and some
// guessing parameter is > 0 (then the method is forced to mle-3pl).
fn parse_score_params(
    params: &serde_json::Map<String, Value>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, String, String), String> {
    let diff_s = params
        .get("diff")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if diff_s.is_empty() {
        return Err("irt_score requires 'diff' (item difficulties, comma-separated)".into());
    }
    let parse_list = |s: &str| -> Result<Vec<f64>, String> {
        s.split(',')
            .map(|x| {
                x.trim()
                    .parse::<f64>()
                    .map_err(|_| format!("must be comma-separated numbers, got '{}'", s))
            })
            .collect()
    };
    let diff = parse_list(&diff_s)?;
    let ndiff = diff.len();
    let disc: Vec<f64> = match params.get("disc").and_then(|v| v.as_str()) {
        Some(d) if !d.is_empty() => parse_list(d)?,
        _ => vec![1.0; ndiff],
    };
    let guess: Vec<f64> = match params.get("guess").and_then(|v| v.as_str()) {
        Some(g) if !g.is_empty() => parse_list(g)?,
        _ => vec![0.0; ndiff],
    };
    if disc.len() != ndiff || guess.len() != ndiff {
        return Err("diff/disc/guess lengths must match the number of items".into());
    }
    let model = params
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("2pl")
        .to_string();
    let mut method = params
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("mle")
        .to_string();
    if !matches!(method.as_str(), "mle" | "eap" | "map") {
        method = "mle".to_string();
    }
    Ok((diff, disc, guess, model, method))
}

/// Full irt_score replacement: mle and mle-3pl delegate to the existing
/// actions_med::irt_score; eap and map are implemented here (girth
/// ability_eap / ability_map, 2PL only — exactly as the Python tool,
/// which forces the 3PL path whenever model == "3pl" and max(guess) > 0).
pub fn irt_score_new(tab: &Table, params: &serde_json::Map<String, Value>) -> Result<Value, String> {
    let (diff, disc, guess, model, method) = parse_score_params(params)?;
    let use_3pl = model == "3pl" && guess.iter().cloned().fold(0.0, f64::max) > 0.0;
    if method == "mle" || use_3pl {
        // Existing implementation already handles mle and mle-3pl
        // (including its method: "mle-3pl" output and SE for 3PL).
        let mut p2 = params.clone();
        p2.insert("method".into(), Value::String("mle".into()));
        return actions_med::irt_score(tab, &p2);
    }
    // eap / map, 2PL.
    let data = binary_matrix(tab)?;
    if diff.len() != data[0].len() {
        return Err(format!(
            "diff has {} values but data has {} items",
            diff.len(),
            data[0].len()
        ));
    }
    let g0 = vec![0.0; diff.len()];
    let abilities = if method == "eap" {
        eap_for_rows_dich(&data, &disc, &diff, &g0)
    } else {
        map_for_rows_dich(&data, &disc, &diff)
    };
    Ok(json!({
        "method": method, "model": model, "n": abilities.len(),
        "a": abilities.iter().map(|v| r6(*v)).collect::<Vec<_>>(),
    }))
}

// MAP ability: Newton on log-likelihood + log N(0,1) prior, clipped to
// [-6, 6] like girth's fminbound search. The objective is concave, so
// Newton from 0 converges to girth's mode.
fn map_for_rows_dich(rows: &[Vec<f64>], disc: &[f64], diff: &[f64]) -> Vec<f64> {
    rows.iter()
        .map(|row| {
            let mut th = 0.0f64;
            for _ in 0..200 {
                let mut grad = -th;
                let mut hess = -1.0;
                for jj in 0..row.len() {
                    let p = expit(disc[jj] * (th - diff[jj]));
                    grad += disc[jj] * (row[jj] - p);
                    hess -= disc[jj] * disc[jj] * p * (1.0 - p);
                }
                let step = if hess != 0.0 { -grad / hess } else { 0.0 };
                let step = step.clamp(-1.0, 1.0);
                th = (th + step).clamp(-6.0, 6.0);
                if step.abs() < 1e-10 {
                    break;
                }
            }
            th
        })
        .collect()
}

// ---------- unit tests (known answers) ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn quadrature_moments() {
        // GL-41 on [-4.5, 4.5] with a standard-normal prior: weights sum
        // to ~1 and the first two moments are ~0 and ~1 (truncation at
        // 4.5 sigma costs ~7e-6 of mass and ~1.4e-4 of variance).
        let (theta, qw) = girth_quadrature();
        assert_eq!(theta.len(), 41);
        let s: f64 = qw.iter().sum();
        assert!(approx(s, 1.0, 1e-4), "sum qw = {}", s);
        let m1: f64 = theta.iter().zip(qw.iter()).map(|(t, w)| t * w).sum();
        assert!(approx(m1, 0.0, 1e-10), "mean = {}", m1);
        let m2: f64 = theta.iter().zip(qw.iter()).map(|(t, w)| t * t * w).sum();
        assert!(approx(m2, 1.0, 5e-4), "var = {}", m2);
    }

    #[test]
    fn rasch_jml_small_girth_verified() {
        // 20x4 matrix from verification (girth rasch_jml, 2026-10-26):
        // d = [-0.118407, -0.118407, -0.118407, 0.420479], disc all 1.
        let tab = crate::parse_csv(
            "i1,i2,i3,i4\n1,1,0,0\n1,0,1,0\n0,1,1,1\n1,1,1,0\n0,0,0,1\n1,0,0,1\n0,1,0,0\n1,1,0,1\n0,0,1,1\n1,0,1,1\n0,1,1,0\n1,1,1,1\n0,0,0,0\n1,0,0,0\n0,1,0,1\n1,1,0,0\n0,0,1,0\n1,0,1,0\n0,1,1,1\n1,1,1,0",
        )
        .unwrap();
        let data = binary_matrix(&tab).unwrap();
        let (disc, betas) = rasch_jml(&data).unwrap();
        assert!(disc.iter().all(|&d| d == 1.0));
        for (got, want) in betas.iter().zip([-0.118407, -0.118407, -0.118407, 0.420479]) {
            assert!(approx(*got, want, 5e-4), "beta {} vs {}", got, want);
        }
    }

    #[test]
    fn rasch_mml_runs_and_disc_common() {
        let tab = crate::parse_csv(
            "i1,i2,i3,i4\n1,1,0,0\n1,0,1,0\n0,1,1,1\n1,1,1,0\n0,0,0,1\n1,0,0,1\n0,1,0,0\n1,1,0,1\n0,0,1,1\n1,0,1,1\n0,1,1,0\n1,1,1,1\n0,0,0,0\n1,0,0,0\n0,1,0,1\n1,1,0,0\n0,0,1,0\n1,0,1,0\n0,1,1,1\n1,1,1,0",
        )
        .unwrap();
        let data = binary_matrix(&tab).unwrap();
        let (pats, counts) = group_patterns(&data);
        let fit = fit_dich(&pats, &counts, DichKind::RaschMml, 1.0, 0.0);
        // Girth onepl_mml on this matrix hits the 0.25 lower bound.
        assert!(approx(fit.a[0], 0.25, 1e-3), "common a = {}", fit.a[0]);
        assert!(fit.ll.is_finite());
    }

    #[test]
    fn twopl_fit_improves_ll_and_probs_sane() {
        let tab = crate::parse_csv(
            "i1,i2,i3,i4\n1,1,0,0\n1,0,1,0\n0,1,1,1\n1,1,1,0\n0,0,0,1\n1,0,0,1\n0,1,0,0\n1,1,0,1\n0,0,1,1\n1,0,1,1\n0,1,1,0\n1,1,1,1\n0,0,0,0\n1,0,0,0\n0,1,0,1\n1,1,0,0\n0,0,1,0\n1,0,1,0\n0,1,1,1\n1,1,1,0",
        )
        .unwrap();
        let data = binary_matrix(&tab).unwrap();
        let (pats, counts) = group_patterns(&data);
        let fit = fit_dich(&pats, &counts, DichKind::TwoPl, 1.0, 0.0);
        assert!(fit.ll.is_finite());
        // Must beat the Rasch (a=1) marginal LL on the same data.
        let rasch = fit_dich(&pats, &counts, DichKind::RaschMml, 1.0, 0.0);
        assert!(fit.ll >= rasch.ll - 1e-6, "2pl ll {} vs rasch {}", fit.ll, rasch.ll);
        for &a in &fit.a {
            assert!(a >= 0.2 && a <= 5.0);
        }
    }

    #[test]
    fn eap_map_fixed_items_bruteforce() {
        // Fixed item set; EAP/MAP checked against brute-force Simpson
        // integration over [-6, 6] (independent of the GL-41 grid).
        let rows = vec![
            vec![1.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
        ];
        let disc = vec![1.0, 1.2, 0.8];
        let diff = vec![-0.5, 0.0, 0.5];
        let g0 = vec![0.0; 3];
        let eap = eap_for_rows_dich(&rows, &disc, &diff, &g0);
        let map = map_for_rows_dich(&rows, &disc, &diff);
        // Brute force EAP with Simpson on a fine grid.
        let n = 12001;
        let (lo, hi) = (-6.0f64, 6.0f64);
        let h = (hi - lo) / (n - 1) as f64;
        for (i, row) in rows.iter().enumerate() {
            let mut num = 0.0;
            let mut den = 0.0;
            for k in 0..n {
                let th = lo + k as f64 * h;
                let mut ll = -0.5 * th * th;
                for jj in 0..3 {
                    let p = expit(disc[jj] * (th - diff[jj]));
                    ll += if row[jj] == 1.0 { p.ln() } else { (1.0 - p).ln() };
                }
                let w = ll.exp();
                let wt = if k == 0 || k == n - 1 { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 };
                num += wt * th * w;
                den += wt * w;
            }
            let brute = num / den;
            assert!(approx(eap[i], brute, 2e-3), "eap {} vs brute {}", eap[i], brute);
            // MAP must be the posterior mode: its log-posterior must be
            // >= the brute-force grid maximum's neighbourhood value, and
            // the gradient at MAP must be ~0.
            let mut grad = -map[i];
            for jj in 0..3 {
                let p = expit(disc[jj] * (map[i] - diff[jj]));
                grad += disc[jj] * (row[jj] - p);
            }
            assert!(grad.abs() < 1e-6, "map gradient {}", grad);
        }
        // Ordering sanity: all-correct is highest, all-wrong lowest.
        assert!(eap[2] > eap[0] && eap[2] > eap[3] && eap[0] > eap[1] && eap[3] > eap[1]);
        assert!(map[2] > map[0] && map[2] > map[3] && map[0] > map[1] && map[3] > map[1]);
    }

    #[test]
    fn grm_pcm_probs_sum_to_one_and_fit_runs() {
        // Hand-built 3-category data; probabilities must normalise and
        // the fits must return finite LLs with ordered GRM thresholds.
        for th in [-2.0, -0.5, 0.0, 0.7, 2.0] {
            let (p, _) = grm_probs_grads(1.2, &[-1.0, 0.5], th);
            assert!(approx(p.iter().sum::<f64>(), 1.0, 1e-12));
            let (p2, _) = pcm_probs_grads(1.1, &[-0.5, 1.0], th);
            assert!(approx(p2.iter().sum::<f64>(), 1.0, 1e-12));
        }
        let tab = crate::parse_csv(
            "i1,i2\n0,1\n1,2\n2,2\n1,1\n0,0\n2,1\n1,0\n2,2\n0,2\n1,1\n2,0\n0,1\n1,2\n2,1\n0,0\n1,0",
        )
        .unwrap();
        let (data, ks) = poly_matrix(&tab).unwrap();
        assert_eq!(ks, vec![3, 3]);
        let (pats, counts) = group_patterns(&data);
        let gfit = fit_poly(&pats, &counts, &ks, PolyKind::Grm);
        assert!(gfit.ll.is_finite());
        for t in &gfit.thr {
            assert!(t[0] < t[1], "GRM thresholds ordered: {:?}", t);
        }
        let pfit = fit_poly(&pats, &counts, &ks, PolyKind::Pcm);
        assert!(pfit.ll.is_finite());
    }

    #[test]
    fn threepl_fit_bounds_and_ll() {
        let tab = crate::parse_csv(
            "i1,i2,i3\n1,1,0\n1,0,1\n0,1,1\n1,1,1\n0,0,0\n1,0,0\n0,1,0\n1,1,0\n0,0,1\n1,0,1\n0,1,1\n1,1,1\n0,0,0\n1,1,0\n0,1,1\n1,0,0",
        )
        .unwrap();
        let data = binary_matrix(&tab).unwrap();
        let (pats, counts) = group_patterns(&data);
        let fit = fit_dich(&pats, &counts, DichKind::ThreePl, 1.0, 0.0);
        assert!(fit.ll.is_finite());
        for &a in &fit.a {
            assert!(a >= 0.25 && a <= 4.0);
        }
        for &g in &fit.g {
            assert!(g >= 0.0 && g <= 0.33);
        }
    }
}
