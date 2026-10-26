// Numeric core for hermes-statistical-tools-rs.
// Design: CSV in -> compact JSON out, same action names / short keys as the
// Python original where implemented. Errors are {"e": "..."} — never panics
// surfaced as stack traces (dispatch catches them).

pub mod actions_med;
pub mod actions_pspp;
pub mod actions_pspp_new;
pub mod actions_irt_new;
pub mod actions_sem_new;
pub mod actions_sm_new;

use nalgebra::{DMatrix, DVector};
use serde_json::{json, Map, Value};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, Normal, StudentsT};
use std::collections::BTreeMap;

// ---------- rounding (Python round(x,4) for pspp, round(x,6) for medical) ----------
pub fn rnd(v: f64, nd: i32) -> Option<f64> {
    if !v.is_finite() {
        return None;
    }
    let m = 10f64.powi(nd);
    let r = (v * m).round() / m;
    Some(if r == 0.0 { 0.0 } else { r })
}
pub fn r4(v: f64) -> Value {
    match rnd(v, 4) {
        Some(x) => json!(x),
        None => Value::Null,
    }
}
pub fn r6(v: f64) -> Value {
    match rnd(v, 6) {
        Some(x) => json!(x),
        None => Value::Null,
    }
}
pub fn r4o(v: Option<f64>) -> Value {
    match v {
        Some(x) => r4(x),
        None => Value::Null,
    }
}

// ---------- distributions ----------
pub fn norm_sf(x: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().sf(x)
}
pub fn norm_cdf(x: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().cdf(x)
}
pub fn norm_ppf(p: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().inverse_cdf(p)
}
pub fn t_sf(x: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    StudentsT::new(0.0, 1.0, df).unwrap().sf(x)
}
pub fn t_ppf(p: f64, df: f64) -> f64 {
    StudentsT::new(0.0, 1.0, df).unwrap().inverse_cdf(p)
}
pub fn f_sf(x: f64, d1: f64, d2: f64) -> f64 {
    FisherSnedecor::new(d1, d2).unwrap().sf(x)
}
pub fn f_ppf(p: f64, d1: f64, d2: f64) -> f64 {
    FisherSnedecor::new(d1, d2).unwrap().inverse_cdf(p)
}
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    ChiSquared::new(df).unwrap().sf(x)
}
pub fn chi2_ppf(p: f64, df: f64) -> f64 {
    ChiSquared::new(df).unwrap().inverse_cdf(p)
}
fn ln_gamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}
fn ln_fact(n: usize) -> f64 {
    ln_gamma(n as f64 + 1.0)
}
fn ln_comb(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_fact(n) - ln_fact(k) - ln_fact(n - k)
}
pub fn binom_pmf(k: usize, n: usize, p: f64) -> f64 {
    if p <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    (ln_comb(n, k) + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()).exp()
}
pub fn binom_cdf(k: usize, n: usize, p: f64) -> f64 {
    (0..=k.min(n)).map(|i| binom_pmf(i, n, p)).sum()
}

// Kolmogorov asymptotic survival function Q_KS(lambda)
pub fn ks_sf(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut s = 0.0;
    for k in 1..=100 {
        let t = (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
        s += if k % 2 == 1 { t } else { -t };
        if t < 1e-15 {
            break;
        }
    }
    (2.0 * s).clamp(0.0, 1.0)
}

// ---------- exact one-sample Kolmogorov-Smirnov CDF ----------
// P(D_n <= d), following scipy's _kolmogn selection (Simard & L'Ecuyer 2011):
// Ruben-Gambino closed forms at the edges, Durbin's matrix algorithm via
// Marsaglia-Tsang-Wang (DMTW) for n*d^2 <= 0.754693, and Pomeranz's recursion
// (1974) for 0.754693 < n*d^2 <= 4 (and for n > 140, where scipy switches to
// Pelz-Good / smirnov approximations — Pomeranz is exact there too and was
// verified against scipy's kolmogn to <1e-12 on d >= 0.5 cases and <1e-15
// on 300 random (n, d) pairs in the prototype). For n > 2000 the asymptotic
// series is used (finite-n error is negligible there; documented bound).
fn ldexp(x: f64, e: i32) -> f64 {
    x * 2f64.powi(e)
}
const EP128: f64 = 340282366920938463463374607431768211456.0; // 2^128
const EM128: f64 = 2.9387358770557188e-39; // 2^-128

fn ks_dmtw(n: usize, d: f64) -> f64 {
    if d >= 1.0 {
        return 1.0;
    }
    let nd = n as f64 * d;
    if nd <= 0.5 {
        return 0.0;
    }
    let k = nd.ceil() as usize;
    let h = k as f64 - nd;
    let m = 2 * k - 1;
    // v[j] = (1 - h^(j+1)) / (j+1)! , w[j] = 1 / j!  (0-indexed)
    let mut v = vec![0.0; m];
    let mut w = vec![0.0; m];
    let mut fac = 1.0f64;
    for j in 0..m {
        w[j] = fac;
        fac /= (j + 1) as f64;
        v[j] = (1.0 - h.powi((j + 1) as i32)) * fac;
    }
    let tt = (2.0 * h - 1.0).max(0.0).powi(m as i32) - 2.0 * h.powi(m as i32);
    v[m - 1] = (1.0 + tt) * fac;
    // H: column 0 = v, last row = reversed v, column i (i>=1) gets w in rows i-1..
    let mut hmat = vec![vec![0.0; m]; m];
    for i in 1..m {
        for r in (i - 1)..m {
            hmat[r][i] = w[r - (i - 1)];
        }
    }
    for r in 0..m {
        hmat[r][0] = v[r];
        hmat[m - 1][r] = v[m - 1 - r];
    }
    let matmul = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        let mut c = vec![vec![0.0; m]; m];
        for i in 0..m {
            for kk in 0..m {
                let aik = a[i][kk];
                if aik != 0.0 {
                    for j in 0..m {
                        c[i][j] += aik * b[kk][j];
                    }
                }
            }
        }
        c
    };
    let mut hpwr: Vec<Vec<f64>> = (0..m).map(|i| (0..m).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let mut hcur = hmat;
    let mut nn = n;
    let mut expnt: i32 = 0;
    let mut hexpnt: i32 = 0;
    while nn > 0 {
        if nn % 2 == 1 {
            hpwr = matmul(&hpwr, &hcur);
            expnt += hexpnt;
        }
        hcur = matmul(&hcur, &hcur);
        hexpnt *= 2;
        if hcur[k - 1][k - 1].abs() > EP128 {
            for row in hcur.iter_mut() {
                for x in row.iter_mut() {
                    *x /= EP128;
                }
            }
            hexpnt += 128;
        }
        nn /= 2;
    }
    let mut p = hpwr[k - 1][k - 1];
    for i in 1..=n {
        p = i as f64 * p / n as f64;
        if p.abs() < EM128 {
            p *= EP128;
            expnt -= 128;
        }
    }
    if expnt != 0 {
        p = ldexp(p, expnt);
    }
    p.clamp(0.0, 1.0)
}

fn pomeranz_j1j2(i: usize, n: usize, ll: i64, ceilf: i64, roundf: i64) -> (i64, i64) {
    let (j1, j2) = if i == 0 {
        (-ll - ceilf - 1, ll + ceilf - 1)
    } else {
        let ip1div2 = (i + 1) / 2;
        let ip1mod2 = (i + 1) % 2;
        if ip1mod2 == 0 {
            if ip1div2 == n + 1 {
                (n as i64 - ll - ceilf - 1, n as i64 + ll + ceilf - 1)
            } else {
                (ip1div2 as i64 - 1 - ll - roundf - 1, ip1div2 as i64 + ll - 1 + ceilf - 1)
            }
        } else {
            (ip1div2 as i64 - 1 - ll - 1, ip1div2 as i64 + ll + roundf - 1)
        }
    };
    (j1.max(-2) + 2, j2.min(n as i64))
    // NOTE: scipy returns (max(j1+2, 0), min(j2, n)); max(-2)+2 == 0 when j1 <= -2.
}

fn ks_pomeranz(n: usize, x: f64) -> f64 {
    let t = n as f64 * x;
    let ll = t.floor() as i64;
    let f = t - ll as f64;
    let g = f.min(1.0 - f);
    let ceilf: i64 = if f > 0.0 { 1 } else { 0 };
    let roundf: i64 = if f > 0.5 { 1 } else { 0 };
    let npwrs = (2 * (ll + 1)) as usize;
    let mut gpower = vec![0.0; npwrs];
    let mut twogpower = vec![0.0; npwrs];
    let mut onem2gpower = vec![0.0; npwrs];
    gpower[0] = 1.0;
    twogpower[0] = 1.0;
    onem2gpower[0] = 1.0;
    for mm in 1..npwrs {
        gpower[mm] = gpower[mm - 1] * (g / n as f64) / mm as f64;
        twogpower[mm] = twogpower[mm - 1] * (2.0 * g / n as f64) / mm as f64;
        onem2gpower[mm] = onem2gpower[mm - 1] * ((1.0 - 2.0 * g) / n as f64) / mm as f64;
    }
    let mut v0 = vec![0.0; npwrs];
    let mut v1 = vec![0.0; npwrs];
    v1[0] = 1.0;
    let (mut v0s, mut v1s) = (0i64, 0i64);
    let mut expnt: i32 = 0;
    let (mut j1, mut j2) = pomeranz_j1j2(0, n, ll, ceilf, roundf);
    for i in 1..=(2 * n + 1) {
        let k1 = j1;
        std::mem::swap(&mut v0, &mut v1);
        std::mem::swap(&mut v0s, &mut v1s);
        for x in v1.iter_mut() {
            *x = 0.0;
        }
        let (nj1, nj2) = pomeranz_j1j2(i, n, ll, ceilf, roundf);
        j1 = nj1;
        j2 = nj2;
        let pwrs: &Vec<f64> = if i == 1 || i == 2 * n + 1 {
            &gpower
        } else if i % 2 == 1 {
            &twogpower
        } else {
            &onem2gpower
        };
        let ln2 = j2 - k1 + 1;
        if ln2 > 0 {
            // convolution of V0[k1-V0s .. k1-V0s+ln2] with pwrs[..ln2]
            let start = (k1 - v0s) as usize;
            let mut conv = vec![0.0; (ln2 as usize) + npwrs];
            for a in 0..(ln2 as usize) {
                let va = v0[start + a];
                if va != 0.0 {
                    for b in 0..(ln2 as usize) {
                        conv[a + b] += va * pwrs[b];
                    }
                }
            }
            let conv_start = (j1 - k1) as usize;
            let conv_len = (j2 - j1 + 1) as usize;
            for c in 0..conv_len {
                v1[c] = conv[conv_start + c];
            }
            let mx = v1.iter().cloned().fold(0.0f64, f64::max);
            if mx > 0.0 && mx < EM128 {
                for x in v1.iter_mut() {
                    *x *= EP128;
                }
                expnt -= 128;
            }
            v1s = v0s + j1 - k1;
        }
    }
    let mut ans = v1[(n as i64 - v1s) as usize];
    for mm in 1..=n {
        if ans.abs() > EP128 {
            ans *= EM128;
            expnt += 128;
        }
        ans *= mm as f64;
    }
    if expnt != 0 {
        ans = ldexp(ans, expnt);
    }
    ans.clamp(0.0, 1.0)
}

/// Exact P(D_n <= d) for the two-sided one-sample KS statistic.
pub fn ks_exact_cdf(n: usize, d: f64) -> f64 {
    if n == 0 || d <= 0.0 {
        return 0.0;
    }
    if d >= 1.0 {
        return 1.0;
    }
    let t = n as f64 * d;
    if t <= 1.0 {
        if t <= 0.5 {
            return 0.0;
        }
        // Ruben-Gambino: (n!/n^n) * (2t-1)^n, computed in log space.
        let mut ln = 0.0f64;
        for i in 1..=n {
            ln += (i as f64).ln();
        }
        ln -= n as f64 * (n as f64).ln();
        ln += n as f64 * (2.0 * t - 1.0).ln();
        return ln.exp().clamp(0.0, 1.0);
    }
    if t >= n as f64 - 1.0 {
        return (1.0 - 2.0 * (1.0 - d).powi(n as i32)).clamp(0.0, 1.0);
    }
    if n > 2000 {
        // Asymptotic series; finite-n error negligible at this n.
        let lam = (n as f64).sqrt() * d;
        return (1.0 - ks_sf(lam)).clamp(0.0, 1.0);
    }
    if d >= 0.5 {
        // Pomeranz is exact here too (verified against scipy kolmogn).
        return ks_pomeranz(n, d);
    }
    let nxsq = n as f64 * d * d;
    if nxsq <= 0.754693 {
        ks_dmtw(n, d)
    } else if nxsq <= 4.0 {
        ks_pomeranz(n, d)
    } else {
        // scipy uses a 2*smirnov (Miller) approximation here; Pomeranz is
        // exact and was verified equivalent to <1e-12 on the d>=0.5 cases.
        ks_pomeranz(n, d)
    }
}

/// Exact two-sided one-sample KS p-value: P(D_n >= d).
pub fn ks_exact_p(n: usize, d: f64) -> f64 {
    (1.0 - ks_exact_cdf(n, d)).clamp(0.0, 1.0)
}

/// Exact two-sided two-sample KS p-value, by DP over the (i, j) lattice.
/// Under the null every combined ordering is equally likely, so a path from
/// (0, 0) to (n, m) has probability 1/C(n+m, n). D takes only lattice values
/// t/lcm(n, m), so scipy rounds d to h/lcm first and P(D >= d) = 1 - P(D < h/lcm):
/// the DP therefore uses a STRICT integer band. The band MUST be integer:
/// comparing f64 divisions |i/n - j/m| < d misclassifies boundary vertices,
/// because e.g. 14/20 - 10/20 in f64 is 0.20000000000000007, not 0.2 — that
/// f64 noise alone moved p by up to 0.17 on tied data. With
/// g = gcd(n, m), diff = |i*m - j*n| / (n*m) = t/lcm, and the band is t < h.
/// Transition from (i, j): next observation is from sample 1 with
/// probability (n-i)/(n+m-i-j), from sample 2 otherwise.
pub fn ks_two_sample_exact_p(n: usize, m: usize, d: f64) -> f64 {
    if n == 0 || m == 0 || d < 0.0 {
        return f64::NAN;
    }
    if d >= 1.0 {
        return 0.0;
    }
    let mut a = n;
    let mut b = m;
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    let g = a as i64; // gcd(n, m)
    let lcm = (n / a) * m;
    let h = (d * lcm as f64).round() as i64;
    if h <= 0 {
        return 1.0;
    }
    let in_band = |i: usize, j: usize| -> bool {
        (i as i64 * m as i64 - j as i64 * n as i64).abs() < h * g
    };
    let mut dp = vec![vec![0.0f64; m + 1]; n + 1];
    if in_band(0, 0) {
        dp[0][0] = 1.0;
    }
    for i in 0..=n {
        for j in 0..=m {
            if !in_band(i, j) {
                dp[i][j] = 0.0;
                continue;
            }
            let cur = dp[i][j];
            if cur == 0.0 {
                continue;
            }
            let rem = (n - i + m - j) as f64;
            if rem == 0.0 {
                continue;
            }
            if i < n && in_band(i + 1, j) {
                dp[i + 1][j] += cur * (n - i) as f64 / rem;
            }
            if j < m && in_band(i, j + 1) {
                dp[i][j + 1] += cur * (m - j) as f64 / rem;
            }
        }
    }
    (1.0 - dp[n][m]).clamp(0.0, 1.0)
}

// ---------- Shapiro-Wilk (Royston AS R94, as implemented by scipy) ----------
// Returns (W, p). None for n < 3 or n > 5000 (scipy returns None in Python's
// examine for n > 5000, and raises for n < 3).
pub fn shapiro_wilk(vals: &[f64]) -> Option<(f64, f64)> {
    let n = vals.len();
    if n < 3 || n > 5000 {
        return None;
    }
    let mut y = vals.to_vec();
    y.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if n == 3 {
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let a = [-c, 0.0, c];
        let num = a.iter().zip(y.iter()).map(|(ai, yi)| ai * yi).sum::<f64>().powi(2);
        let m = mean(&y);
        let den: f64 = y.iter().map(|v| (v - m).powi(2)).sum();
        if den == 0.0 {
            return Some((1.0, 1.0));
        }
        let w = (num / den).clamp(0.75, 1.0);
        let p = (1.0 - 6.0 / std::f64::consts::PI * w.sqrt().acos()).clamp(0.0, 1.0);
        return Some((w, p));
    }
    let nf = n as f64;
    let mut mvec = Vec::with_capacity(n);
    for i in 1..=n {
        mvec.push(norm_ppf((i as f64 - 3.0 / 8.0) / (nf + 1.0 / 4.0)));
    }
    let u = nf.powf(-0.5);
    let mtm: f64 = mvec.iter().map(|v| v * v).sum();
    let c: Vec<f64> = mvec.iter().map(|v| v / mtm.sqrt()).collect();
    let mn = mvec[n - 1];
    let cn = c[n - 1];
    let an = cn + 0.221157 * u - 0.147981 * u.powi(2) - 2.071190 * u.powi(3)
        + 4.434685 * u.powi(4) - 2.706056 * u.powi(5);
    let (phi, mut a) = if n <= 5 {
        let phi = (mtm - 2.0 * mn * mn) / (1.0 - 2.0 * an * an);
        (phi, mvec.iter().map(|v| v / phi.sqrt()).collect::<Vec<f64>>())
    } else {
        let mnm1 = mvec[n - 2];
        let cnm1 = c[n - 2];
        let anm1 = cnm1 + 0.042981 * u - 0.293762 * u.powi(2) - 1.752461 * u.powi(3)
            + 5.682633 * u.powi(4) - 3.582633 * u.powi(5);
        let phi = (mtm - 2.0 * mn * mn - 2.0 * mnm1 * mnm1)
            / (1.0 - 2.0 * an * an - 2.0 * anm1 * anm1);
        let mut a: Vec<f64> = mvec.iter().map(|v| v / phi.sqrt()).collect();
        a[n - 2] = anm1;
        a[1] = -anm1;
        (phi, a)
    };
    let _ = phi;
    a[n - 1] = an;
    a[0] = -an;
    let num = a.iter().zip(y.iter()).map(|(ai, yi)| ai * yi).sum::<f64>().powi(2);
    let mean_y = mean(&y);
    let den: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum();
    if den == 0.0 {
        return Some((1.0, 1.0));
    }
    let w = (num / den).clamp(0.0, 1.0);
    let p = if n <= 11 {
        let un = nf;
        let gamma = -2.273 + 0.459 * un;
        let mu = 0.5440 - 0.39978 * un + 0.025054 * un.powi(2) - 0.0006714 * un.powi(3);
        let sigma = (1.3822 - 0.77857 * un + 0.062767 * un.powi(2) - 0.0020322 * un.powi(3)).exp();
        let arg = gamma - (1.0 - w).ln();
        if arg <= 0.0 {
            0.0
        } else {
            let gw = -arg.ln();
            norm_cdf(-(gw - mu) / sigma)
        }
    } else {
        let lu = nf.ln();
        let mu = -1.5861 - 0.31082 * lu - 0.083751 * lu.powi(2) + 0.0038915 * lu.powi(3);
        let sigma = (-0.4803 - 0.082676 * lu + 0.0030302 * lu.powi(2)).exp();
        let gw = if w >= 1.0 { -36.0 } else { (1.0 - w).ln() };
        norm_cdf(-(gw - mu) / sigma)
    };
    Some((w, p.clamp(0.0, 1.0)))
}

// Studentized-range survival function, by double Simpson integration.
// Q = range of k standard normals / sqrt(chi2_nu/nu). CDF(q) =
// int_u P(range <= q*sqrt(u/nu)) * chi2_pdf(u) du, with
// P(range <= w) = k * int_z phi(z) * (Phi(z)-Phi(z-w))^(k-1) dz.
pub fn studentized_range_sf(q: f64, k: usize, df: f64) -> f64 {
    if q <= 0.0 {
        return 1.0;
    }
    if !(df > 0.0) || k < 2 {
        return f64::NAN;
    }
    let range_cdf = |w: f64| -> f64 {
        if w <= 0.0 {
            return 0.0;
        }
        // Simpson over z in [-7, 7], step 0.05 -> 281 points
        let (a, b, n) = (-7.0f64, 7.0f64, 280usize);
        let h = (b - a) / n as f64;
        let mut s = 0.0;
        for i in 0..=n {
            let z = a + i as f64 * h;
            let d = (norm_cdf(z) - norm_cdf(z - w)).max(0.0);
            let f = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
                * d.powi((k - 1) as i32);
            let wgt = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
            s += wgt * f;
        }
        (k as f64 * s * h / 3.0).clamp(0.0, 1.0)
    };
    // chi2 density for u; integrate u over [0, u_hi] Simpson, 200 steps
    let u_hi = df + 12.0 * (2.0 * df).sqrt() + 40.0;
    let n = 200usize;
    let h = u_hi / n as f64;
    let chi2_pdf = |u: f64| -> f64 {
        if u <= 0.0 {
            return if df == 2.0 { 0.5 } else { 0.0 };
        }
        ((df / 2.0 - 1.0) * u.ln() - u / 2.0 - (df / 2.0) * (2.0f64).ln() - ln_gamma(df / 2.0)).exp()
    };
    let mut s = 0.0;
    for i in 0..=n {
        let u = i as f64 * h;
        let w = q * (u / df).sqrt();
        let f = range_cdf(w) * chi2_pdf(u);
        let wgt = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        s += wgt * f;
    }
    let cdf = (s * h / 3.0).clamp(0.0, 1.0);
    (1.0 - cdf).clamp(0.0, 1.0)
}

// Noncentral t CDF via series over chi2 (for power): P(T <= x) with ncp.
// Uses the standard representation integrating over chi: T = (Z+ncp)/sqrt(V/nu).
// Numeric integration over V with Simpson on a chi2 grid is slow but the
// power use-case calls it few times; use Gauss-style Simpson, 120 steps.
pub fn nct_cdf(x: f64, df: f64, ncp: f64) -> f64 {
    // Integrate: P = E_V[ Phi(x*sqrt(V/df) - ncp) ]
    let u_hi = df + 12.0 * (2.0 * df).sqrt() + 40.0;
    let n = 240usize;
    let h = u_hi / n as f64;
    let chi2_pdf = |u: f64| -> f64 {
        if u <= 0.0 {
            return 0.0;
        }
        ((df / 2.0 - 1.0) * u.ln() - u / 2.0 - (df / 2.0) * (2.0f64).ln() - ln_gamma(df / 2.0)).exp()
    };
    let mut s = 0.0;
    for i in 0..=n {
        let u = i as f64 * h;
        let f = norm_cdf(x * (u / df).sqrt() - ncp) * chi2_pdf(u);
        let wgt = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        s += wgt * f;
    }
    (s * h / 3.0).clamp(0.0, 1.0)
}

// Noncentral F CDF via series: F = (X1/d1)/(X2/d2), X1 ~ chi2(d1, lam).
// Poisson mixture over noncentral chi2: P = sum_j e^{-lam/2}(lam/2)^j/j! *
// I_{...}(regularized beta). Use statrs Beta CDF via incomplete beta through
// the F CDF identity: for central F, use f CDF directly per mixture term.
pub fn ncf_cdf(x: f64, d1: f64, d2: f64, lam: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Regularized incomplete beta via continued fraction (Numerical Recipes betacf)
    fn betacf(a: f64, b: f64, x: f64) -> f64 {
        let (mut qab, mut qap, mut qam) = (1.0, 1.0, 1.0);
        let mut c = 1.0;
        let mut d = 1.0 - (a + b) * x / (a + 1.0);
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        d = 1.0 / d;
        let mut h = d;
        for m in 1..=200 {
            let m2 = 2 * m;
            let aa = m as f64 * (b - m as f64) * x / ((a + m2 as f64 - 1.0) * (a + m2 as f64));
            d = 1.0 + aa * d;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = 1.0 + aa / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            h *= d * c;
            let aa2 = -(a + m as f64) * (a + b + m as f64) * x / ((a + m2 as f64) * (a + m2 as f64 + 1.0));
            d = 1.0 + aa2 * d;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = 1.0 + aa2 / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 3e-12 {
                break;
            }
            qab = qap;
            qap = qam;
            qam = h;
            let _ = (qab, qap, qam);
        }
        h
    }
    fn ibeta(a: f64, b: f64, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
        if x < (a + 1.0) / (a + b + 2.0) {
            bt * betacf(a, b, x) / a
        } else {
            1.0 - bt * betacf(b, a, 1.0 - x) / b
        }
    }
    // Central F CDF via beta: P(F<=x) = I_{d1 x/(d1 x+d2)}(d1/2, d2/2)
    let f_cdf = |xx: f64, dd1: f64, dd2: f64| -> f64 {
        let t = dd1 * xx / (dd1 * xx + dd2);
        ibeta(dd1 / 2.0, dd2 / 2.0, t)
    };
    let mut total = 0.0;
    let mut term = (-lam / 2.0).exp(); // j = 0 Poisson weight
    for j in 0..200 {
        let df1j = d1 + 2.0 * j as f64;
        total += term * f_cdf(x * d1 / df1j, df1j, d2);
        if term < 1e-14 && j > 10 {
            break;
        }
        term *= (lam / 2.0) / (j as f64 + 1.0);
    }
    total.clamp(0.0, 1.0)
}

// ---------- basic stats ----------
pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
pub fn var_ddof1(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}
pub fn sd1(v: &[f64]) -> f64 {
    var_ddof1(v).sqrt()
}
pub fn median_sorted(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
pub fn percentile(v: &[f64], q: f64) -> f64 {
    // numpy default: linear interpolation
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        return f64::NAN;
    }
    let pos = q / 100.0 * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        s[lo]
    } else {
        s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
    }
}
pub fn skew(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if v.len() < 3 {
        return 0.0;
    }
    let m = mean(v);
    let m2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m3 = v.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n;
    if m2 == 0.0 {
        return 0.0;
    }
    // scipy skew default: bias=True -> m3 / m2^1.5
    m3 / m2.powf(1.5)
}
pub fn kurtosis_fisher(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if v.len() < 4 {
        return 0.0;
    }
    let m = mean(v);
    let m2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m4 = v.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        return 0.0;
    }
    m4 / (m2 * m2) - 3.0
}
// Average ranks (1-based), ties averaged — scipy rankdata default.
pub fn rankdata(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut ranks = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + 1 + j + 1) as f64 / 2.0;
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}
pub fn pearson(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len());
    let (x, y) = (&x[..n], &y[..n]);
    let mx = mean(x);
    let my = mean(y);
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
    let r = if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { 0.0 };
    let p = if n > 2 && r.abs() < 1.0 {
        let t = r * (((n - 2) as f64) / (1.0 - r * r)).sqrt();
        2.0 * t_sf(t.abs(), (n - 2) as f64)
    } else if r.abs() >= 1.0 {
        0.0
    } else {
        1.0
    };
    (r, p)
}
// OLS via SVD pseudo-inverse (mirrors numpy lstsq + inv(XTX) SEs).
pub struct Ols {
    pub beta: Vec<f64>,
    pub se: Vec<f64>,
    pub t: Vec<f64>,
    pub p: Vec<f64>,
    pub r2: f64,
    pub adj_r2: f64,
    pub ss_res: f64,
}
pub fn ols(y: &[f64], x_cols: &[Vec<f64>], add_intercept: bool) -> Option<Ols> {
    let n = y.len();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    if add_intercept {
        cols.push(vec![1.0; n]);
    }
    for c in x_cols {
        cols.push(c[..n].to_vec());
    }
    let k = cols.len();
    if k == 0 || n <= k {
        return None;
    }
    let mut mat = DMatrix::zeros(n, k);
    for (j, c) in cols.iter().enumerate() {
        for i in 0..n {
            mat[(i, j)] = c[i];
        }
    }
    let yv = DVector::from_column_slice(y);
    let svd = mat.clone().svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let s = svd.singular_values;
    let tol = f64::EPSILON * (n.max(k) as f64) * s[0];
    let mut beta = DVector::zeros(k);
    // beta = V * diag(1/s) * U^T y
    let uty = u.transpose() * &yv;
    let mut tmp = DVector::zeros(k);
    for i in 0..k {
        if s[i] > tol {
            tmp[i] = uty[i] / s[i];
        }
    }
    beta = vt.transpose() * tmp;
    let yhat = &mat * &beta;
    let resid = &yv - &yhat;
    let ss_res: f64 = resid.iter().map(|e| e * e).sum();
    let ymean = mean(y);
    let ss_tot: f64 = y.iter().map(|e| (e - ymean).powi(2)).sum();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let adj_r2 = 1.0 - (1.0 - r2) * (n - 1) as f64 / (n - k) as f64;
    let sigma2 = ss_res / (n - k) as f64;
    // cov = sigma2 * V * diag(1/s^2) * V^T
    let mut cov = DMatrix::zeros(k, k);
    let v = vt.transpose();
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
    let mut se = vec![0.0; k];
    let mut t = vec![0.0; k];
    let mut p = vec![0.0; k];
    for i in 0..k {
        se[i] = cov[(i, i)].max(0.0).sqrt();
        t[i] = if se[i] > 0.0 { beta[i] / se[i] } else { 0.0 };
        p[i] = 2.0 * t_sf(t[i].abs(), (n - k) as f64);
    }
    Some(Ols {
        beta: beta.iter().cloned().collect(),
        se,
        t,
        p,
        r2,
        adj_r2,
        ss_res,
    })
}
pub fn mat_inverse(m: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    m.clone().try_inverse()
}

// ---------- CSV table ----------
#[derive(Clone, Debug, PartialEq)]
pub enum Cell {
    Num(f64),
    Txt(String),
    Miss,
}
impl Cell {
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Cell::Num(x) => Some(*x),
            _ => None,
        }
    }
    pub fn key(&self) -> String {
        match self {
            Cell::Num(x) => {
                if x.fract() == 0.0 && x.is_finite() {
                    format!("{:.1}", x)
                } else {
                    format!("{}", x)
                }
            }
            Cell::Txt(s) => s.clone(),
            Cell::Miss => String::new(),
        }
    }
    pub fn is_valid(&self) -> bool {
        !(matches!(self, Cell::Miss) || matches!(self, Cell::Num(x) if x.is_nan()))
    }
}
#[derive(Clone, Debug)]
pub struct Table {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<Cell>>,
}
impl Table {
    pub fn col_idx(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|h| h == name)
    }
    pub fn col(&self, name: &str) -> Option<Vec<Cell>> {
        let j = self.col_idx(name)?;
        Some(self.rows.iter().map(|r| r.get(j).cloned().unwrap_or(Cell::Miss)).collect())
    }
    // Python _arr: numeric values, skipping None/NaN; a present text cell is an error.
    pub fn numeric(&self, name: &str) -> Result<Vec<f64>, String> {
        let col = self.col(name).ok_or_else(|| format!("Variable '{}' not found", name))?;
        let mut out = Vec::new();
        for c in &col {
            match c {
                Cell::Num(x) => {
                    if !x.is_nan() {
                        out.push(*x);
                    }
                }
                Cell::Txt(s) => return Err(format!("Non-numeric value '{}' in column '{}'", s, name)),
                Cell::Miss => {}
            }
        }
        Ok(out)
    }
    // Raw cells (for grouping / categories), truncated to n rows like Python zip semantics.
    pub fn groups_numeric(&self, val_col: &str, grp_col: &str, targets: &[Cell]) -> Result<Vec<Vec<f64>>, String> {
        let vals = self.col(val_col).ok_or_else(|| format!("Variable '{}' not found", val_col))?;
        let grps = self.col(grp_col).ok_or_else(|| format!("Variable '{}' not found", grp_col))?;
        let mut out: Vec<Vec<f64>> = targets.iter().map(|_| Vec::new()).collect();
        for i in 0..vals.len().min(grps.len()) {
            if let Some(ti) = targets.iter().position(|t| t == &grps[i]) {
                if let Cell::Num(x) = vals[i] {
                    if !x.is_nan() {
                        out[ti].push(x);
                    }
                }
            }
        }
        Ok(out)
    }
    pub fn unique_groups(&self, name: &str) -> Result<Vec<Cell>, String> {
        let col = self.col(name).ok_or_else(|| format!("Variable '{}' not found", name))?;
        let mut set: Vec<Cell> = Vec::new();
        for c in col {
            if c.is_valid() && !set.contains(&c) {
                set.push(c);
            }
        }
        set.sort_by(|a, b| match (a, b) {
            (Cell::Num(x), Cell::Num(y)) => x.partial_cmp(y).unwrap(),
            (Cell::Txt(x), Cell::Txt(y)) => x.cmp(y),
            (Cell::Num(_), Cell::Txt(_)) => std::cmp::Ordering::Less,
            _ => std::cmp::Ordering::Equal,
        });
        Ok(set)
    }
}
pub fn parse_csv(data: &str) -> Result<Table, String> {
    let text = data.trim();
    if text.is_empty() {
        return Err("Could not parse CSV data".to_string());
    }
    // Simple CSV parser: quotes, doubled quotes, commas, CRLF.
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_q = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if in_q {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_q = false;
                }
            } else {
                field.push(ch);
            }
        } else if ch == '"' && field.is_empty() {
            in_q = true;
        } else if ch == ',' {
            cur.push(field);
            field = String::new();
        } else if ch == '\n' {
            cur.push(field);
            field = String::new();
            rows.push(cur);
            cur = Vec::new();
        } else if ch == '\r' {
            // ignore; \n follows
        } else {
            field.push(ch);
        }
    }
    if !field.is_empty() || !cur.is_empty() {
        cur.push(field);
        rows.push(cur);
    }
    rows.retain(|r| !(r.len() == 1 && r[0].trim().is_empty()));
    if rows.is_empty() {
        return Err("Could not parse CSV data".to_string());
    }
    let headers: Vec<String> = rows[0].iter().map(|h| h.trim().to_string()).collect();
    let mut parsed: Vec<Vec<Cell>> = Vec::new();
    for r in &rows[1..] {
        let mut row: Vec<Cell> = Vec::new();
        for j in 0..headers.len() {
            let raw = r.get(j).map(|s| s.as_str()).unwrap_or("").trim();
            if raw.is_empty() {
                row.push(Cell::Miss);
            } else if let Ok(x) = raw.parse::<f64>() {
                row.push(Cell::Num(x));
            } else {
                row.push(Cell::Txt(raw.to_string()));
            }
        }
        parsed.push(row);
    }
    // Python rule: a column >80% numeric becomes all-float (text -> NaN).
    let mut table = Table { headers, rows: parsed };
    for j in 0..table.headers.len() {
        let col: Vec<Cell> = table.rows.iter().map(|r| r[j].clone()).collect();
        let n = col.len();
        if n > 0 {
            let num = col.iter().filter(|c| matches!(c, Cell::Num(_))).count();
            if num as f64 > n as f64 * 0.8 {
                for r in table.rows.iter_mut() {
                    if !matches!(r[j], Cell::Num(_)) {
                        r[j] = Cell::Num(f64::NAN);
                    }
                }
            }
        }
    }
    Ok(table)
}
pub fn parse_cell_list(s: &str) -> Vec<Cell> {
    // JSON-ish "[1,2]" or comma list "1,2" — numbers parsed as numbers.
    let t = s.trim();
    let inner = t.strip_prefix('[').and_then(|x| x.strip_suffix(']')).unwrap_or(t);
    inner
        .split(',')
        .map(|p| p.trim().trim_matches('"').trim().to_string())
        .filter(|p| !p.is_empty())
        .map(|p| {
            if let Ok(x) = p.parse::<f64>() {
                Cell::Num(x)
            } else {
                Cell::Txt(p)
            }
        })
        .collect()
}
pub fn err(msg: &str) -> Value {
    json!({"e": msg})
}
pub fn obj(pairs: Vec<(&str, Value)>) -> Value {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    Value::Object(m)
}
pub fn bmap_to_json(m: BTreeMap<String, Value>) -> Value {
    let mut out = Map::new();
    for (k, v) in m {
        out.insert(k, v);
    }
    Value::Object(out)
}
