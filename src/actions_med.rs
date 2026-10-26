// medical_ext (km, cox, logrank, forest, power ttest/anova/prop) and
// IRT descriptive actions (score, ctt). Rounding here is 6dp like the
// Python medical tool's _f; the PSPP meta/forest path uses 4dp in pspp.
use crate::*;
use nalgebra::DMatrix;
use serde_json::{json, Value};

// KM curve with Greenwood log-log CI (lifelines default transform).
pub struct KmOut {
    pub surv: Vec<(f64, f64, f64, f64)>, // (t, S, lo, hi)
    pub median: Option<f64>,
    pub at_risk_last: i64,
}
pub fn km_fit(times: &[f64], events: &[i64]) -> KmOut {
    // lifelines timeline: 0.0 with S=1, then every distinct time (event or
    // censor); S changes only at events, CI carried forward at censor times.
    // at_risk_last = at-risk count at the last event time (event_table).
    let n = times.len();
    let mut out = vec![(0.0, 1.0, 1.0, 1.0)];
    let mut surv = 1.0;
    let mut var_sum = 0.0;
    let mut median = None;
    let mut at_risk_last = 0i64;
    let mut distinct: Vec<f64> = times.to_vec();
    distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    if let Some(&last_t) = distinct.last() {
        at_risk_last = (0..n).filter(|&k| times[k] >= last_t - 1e-12).count() as i64;
    }
    for t in distinct {
        let at_risk = (0..n).filter(|&k| times[k] >= t - 1e-12).count() as f64;
        let d = (0..n).filter(|&k| (times[k] - t).abs() < 1e-10 && events[k] == 1).count() as f64;
        if d > 0.0 {
            var_sum += d / (at_risk * (at_risk - d));
            surv *= (at_risk - d) / at_risk;
            if median.is_none() && surv <= 0.5 {
                median = Some(t);
            }
        }
        let (lo, hi) = if surv > 0.0 && surv < 1.0 && var_sum > 0.0 {
            let theta = (-surv.ln()).ln();
            let se = var_sum.sqrt() / surv.ln().abs();
            let lo = (-(theta + 1.96 * se).exp()).exp();
            let hi = (-(theta - 1.96 * se).exp()).exp();
            (lo, hi)
        } else if surv >= 1.0 {
            (1.0, 1.0)
        } else {
            (0.0, 0.0)
        };
        out.push((t, surv, lo, hi));
    }
    KmOut { surv: out, median, at_risk_last }
}

fn surv_map(km: &KmOut) -> (Value, Value, Value) {
    let mut surv = serde_json::Map::new();
    let mut lo = serde_json::Map::new();
    let mut hi = serde_json::Map::new();
    for (t, s, l, h) in &km.surv {
        let key = if t.fract() == 0.0 { format!("{:.1}", t) } else { format!("{}", t) };
        surv.insert(key.clone(), r6(*s));
        lo.insert(key.clone(), r6(*l));
        hi.insert(key.clone(), r6(*h));
    }
    (Value::Object(surv), Value::Object(lo), Value::Object(hi))
}

pub fn dispatch_med(action: &str, tab: &Table, params: &serde_json::Map<String, Value>) -> Result<Value, String> {
    match action {
        "km" => km_action(tab, params),
        "cox" => cox_action(tab, params),
        "logrank" => logrank_action(tab, params),
        "forest" => forest_action(tab, params),
        "ttest" | "anova" | "prop" => power_action(action, params),
        _ => Err(format!("unknown action '{}'. valid: km, cox, logrank, forest, ttest, anova, prop", action)),
    }
}
fn ps(params: &serde_json::Map<String, Value>, key: &str) -> String {
    params.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string()
}
fn pf(params: &serde_json::Map<String, Value>, key: &str, def: f64) -> f64 {
    params.get(key).and_then(|v| v.as_f64()).unwrap_or(def)
}
fn pi(params: &serde_json::Map<String, Value>, key: &str, def: i64) -> i64 {
    params.get(key).and_then(|v| v.as_i64()).unwrap_or(def)
}

fn km_action(tab: &Table, params: &serde_json::Map<String, Value>) -> Result<Value, String> {
    let tcol = ps(params, "t");
    let ecol = ps(params, "e");
    let gcol = ps(params, "g");
    if tcol.is_empty() || ecol.is_empty() {
        return Err("km requires 't' (duration) and 'e' (event indicator)".into());
    }
    if tab.col_idx(&tcol).is_none() || tab.col_idx(&ecol).is_none() {
        return Err(format!("km requires columns '{}' and '{}' in data", tcol, ecol));
    }
    let times = tab.numeric(&tcol)?;
    let ev_raw = tab.numeric(&ecol)?;
    let n = times.len().min(ev_raw.len());
    let times = &times[..n];
    let events: Vec<i64> = ev_raw[..n].iter().map(|e| if *e >= 0.5 { 1 } else { 0 }).collect();
    if !gcol.is_empty() && tab.col_idx(&gcol).is_some() {
        let grp = tab.col(&gcol).unwrap();
        let mut labels: Vec<String> = Vec::new();
        for c in grp.iter().take(n) {
            let k = c.key();
            if !labels.contains(&k) {
                labels.push(k);
            }
        }
        let mut gmap = serde_json::Map::new();
        for lab in &labels {
            let idx: Vec<usize> = (0..n).filter(|&i| grp[i].key() == *lab).collect();
            let t2: Vec<f64> = idx.iter().map(|&i| times[i]).collect();
            let e2: Vec<i64> = idx.iter().map(|&i| events[i]).collect();
            let km = km_fit(&t2, &e2);
            let (s, lo, hi) = surv_map(&km);
            gmap.insert(lab.clone(), json!({"n": idx.len(), "median": km.median.map(|m| json!(m)).unwrap_or(Value::Null), "at_risk": km.at_risk_last, "surv": s, "lo": lo, "hi": hi}));
        }
        let mut out = json!({"g": Value::Object(gmap)});
        if labels.len() == 2 {
            let (z, pv) = logrank_two(tab, params, &labels)?;
            out.as_object_mut().unwrap().insert("logrank".into(), json!({"z": r6(z), "p": r6(pv), "g1": labels[0], "g2": labels[1]}));
        } else if labels.len() > 2 {
            let (z, pv) = logrank_multi(tab, params, &labels)?;
            out.as_object_mut().unwrap().insert("logrank".into(), json!({"z": r6(z), "p": r6(pv), "k": labels.len()}));
        }
        return Ok(out);
    }
    let km = km_fit(times, &events);
    let (s, lo, hi) = surv_map(&km);
    Ok(json!({"n": n, "median": km.median.map(|m| json!(m)).unwrap_or(Value::Null), "surv": s, "lo": lo, "hi": hi, "at_risk": km.at_risk_last}))
}

// Mantel-Haenszel log-rank, lifelines convention: test statistic is a
// z-like value? lifelines logrank_test.test_statistic is (O-E)/sqrt(V)
// for two groups (a z), and multivariate returns a chi2-like statistic.
// We verify against Python in the deep test and document the result.
fn logrank_two(tab: &Table, params: &serde_json::Map<String, Value>, labels: &[String]) -> Result<(f64, f64), String> {
    let tcol = ps(params, "t");
    let ecol = ps(params, "e");
    let gcol = ps(params, "g");
    let times = tab.numeric(&tcol)?;
    let ev_raw = tab.numeric(&ecol)?;
    let grp = tab.col(&gcol).unwrap();
    let n = times.len().min(ev_raw.len()).min(grp.len());
    let mut ev_times: Vec<f64> = (0..n).filter(|&i| ev_raw[i] >= 0.5).map(|i| times[i]).collect();
    ev_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ev_times.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    let (mut o1, mut e1, mut v) = (0.0, 0.0, 0.0);
    for t in ev_times {
        let risk: Vec<usize> = (0..n).filter(|&i| times[i] >= t - 1e-10).collect();
        let n_at = risk.len() as f64;
        let n1_at = risk.iter().filter(|&&i| grp[i].key() == labels[1]).count() as f64;
        let n0_at = n_at - n1_at;
        let d_at = risk.iter().filter(|&&i| (times[i] - t).abs() < 1e-10 && ev_raw[i] >= 0.5).count() as f64;
        let d1_at = risk.iter().filter(|&&i| (times[i] - t).abs() < 1e-10 && ev_raw[i] >= 0.5 && grp[i].key() == labels[1]).count() as f64;
        if n_at > 1.0 && d_at > 0.0 && n1_at > 0.0 && n0_at > 0.0 {
            e1 += d_at * n1_at / n_at;
            o1 += d1_at;
            v += (n1_at * n0_at * d_at * (n_at - d_at)) / (n_at * n_at * (n_at - 1.0));
        }
    }
    // lifelines logrank_test.test_statistic is the chi-square statistic
    // (O-E)^2/V, with p from chi2(1) — not the signed z.
    let chi2 = if v > 0.0 { (o1 - e1).powi(2) / v } else { 0.0 };
    Ok((chi2, chi2_sf(chi2, 1.0)))
}
fn logrank_multi(tab: &Table, params: &serde_json::Map<String, Value>, labels: &[String]) -> Result<(f64, f64), String> {
    let tcol = ps(params, "t");
    let ecol = ps(params, "e");
    let gcol = ps(params, "g");
    let times = tab.numeric(&tcol)?;
    let ev_raw = tab.numeric(&ecol)?;
    let grp = tab.col(&gcol).unwrap();
    let n = times.len().min(ev_raw.len()).min(grp.len());
    let mut ev_times: Vec<f64> = (0..n).filter(|&i| ev_raw[i] >= 0.5).map(|i| times[i]).collect();
    ev_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ev_times.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    let k = labels.len();
    let (mut os, mut es, mut vs) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
    for t in ev_times {
        let risk: Vec<usize> = (0..n).filter(|&i| times[i] >= t - 1e-10).collect();
        let n_at = risk.len() as f64;
        let d_at = risk.iter().filter(|&&i| (times[i] - t).abs() < 1e-10 && ev_raw[i] >= 0.5).count() as f64;
        if n_at > 1.0 && d_at > 0.0 {
            for (gi, gl) in labels.iter().enumerate() {
                let n_at_g = risk.iter().filter(|&&i| grp[i].key() == *gl).count() as f64;
                let d_at_g = risk.iter().filter(|&&i| (times[i] - t).abs() < 1e-10 && ev_raw[i] >= 0.5 && grp[i].key() == *gl).count() as f64;
                if n_at_g > 0.0 {
                    es[gi] += d_at * n_at_g / n_at;
                    os[gi] += d_at_g;
                    vs[gi] += (n_at_g * (n_at - n_at_g) * d_at * (n_at - d_at)) / (n_at * n_at * (n_at - 1.0));
                }
            }
        }
    }
    let chi2: f64 = (0..k).filter(|&gi| vs[gi] > 0.0).map(|gi| (os[gi] - es[gi]).powi(2) / vs[gi]).sum();
    Ok((chi2, chi2_sf(chi2, (k - 1) as f64)))
}

fn logrank_action(tab: &Table, params: &serde_json::Map<String, Value>) -> Result<Value, String> {
    let tcol = ps(params, "t");
    let ecol = ps(params, "e");
    let gcol = ps(params, "g");
    if tcol.is_empty() || ecol.is_empty() || gcol.is_empty() {
        return Err("logrank requires 't', 'e' and 'g'".into());
    }
    let grp = tab.col(&gcol).ok_or("group column not found")?;
    let mut labels: Vec<String> = Vec::new();
    for c in &grp {
        let k = c.key();
        if !labels.contains(&k) {
            labels.push(k);
        }
    }
    if labels.len() == 2 {
        let (z, pv) = logrank_two(tab, params, &labels)?;
        let grp = tab.col(&gcol).unwrap();
        let n1 = grp.iter().filter(|c| c.key() == labels[0]).count();
        let n2 = grp.iter().filter(|c| c.key() == labels[1]).count();
        Ok(json!({"z": r6(z), "p": r6(pv), "g1": labels[0], "g2": labels[1], "n1": n1, "n2": n2}))
    } else {
        let (z, pv) = logrank_multi(tab, params, &labels)?;
        Ok(json!({"z": r6(z), "p": r6(pv), "k": labels.len()}))
    }
}

// Cox PH with Efron ties (lifelines default), Newton-Raphson. Returns per-covariate
// c, s, z, p, hr, l, u and concordance.
fn cox_action(tab: &Table, params: &serde_json::Map<String, Value>) -> Result<Value, String> {
    let tcol = ps(params, "t");
    let ecol = ps(params, "e");
    let preds: Vec<String> = params.get("p").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
    if tcol.is_empty() || ecol.is_empty() || preds.is_empty() {
        return Err("cox requires 't', 'e' and 'p' (predictor list)".into());
    }
    let times = tab.numeric(&tcol)?;
    let ev_raw = tab.numeric(&ecol)?;
    let xcols: Vec<Vec<f64>> = preds.iter().map(|c| tab.numeric(c).unwrap_or_default()).collect();
    let n = times.len().min(ev_raw.len()).min(xcols.iter().map(|c| c.len()).min().unwrap_or(0));
    let times = &times[..n];
    let events: Vec<i64> = ev_raw[..n].iter().map(|e| if *e >= 0.5 { 1 } else { 0 }).collect();
    let k = preds.len();
    let mut x = DMatrix::zeros(n, k);
    for (j, c) in xcols.iter().enumerate() {
        for i in 0..n {
            x[(i, j)] = c[i];
        }
    }
    // Newton on the Efron partial log-likelihood — lifelines 0.30 (the
    // Python backend) uses Efron ties, not Breslow. For d = 1 the two
    // coincide, so untied data are unaffected. At an event time with d
    // tied events, Efron replaces the risk-set sums by
    //   S_k = S - (k/d) * S_events,  k = 0..d-1,
    // and the log-likelihood contribution is sum_events eta_i
    // minus sum_k log(S0_k). Gradient/Hessian follow by differentiating
    // that expression.
    let mut beta = nalgebra::DVector::zeros(k);
    for _ in 0..100 {
        let eta: Vec<f64> = (0..n).map(|i| (0..k).map(|j| x[(i, j)] * beta[j]).sum()).collect();
        let mut grad = nalgebra::DVector::zeros(k);
        let mut hess = DMatrix::zeros(k, k);
        let mut ev_times: Vec<f64> = (0..n).filter(|&i| events[i] == 1).map(|i| times[i]).collect();
        ev_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ev_times.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        for t in ev_times {
            let risk: Vec<usize> = (0..n).filter(|&i| times[i] >= t - 1e-10).collect();
            let evts: Vec<usize> = risk.iter().cloned().filter(|&i| (times[i] - t).abs() < 1e-10 && events[i] == 1).collect();
            let d = evts.len();
            let exps: Vec<f64> = risk.iter().map(|&i| eta[i].exp()).collect();
            let s0: f64 = exps.iter().sum();
            let mut s1 = nalgebra::DVector::zeros(k);
            let mut s2: DMatrix<f64> = DMatrix::zeros(k, k);
            for (ri, &i) in risk.iter().enumerate() {
                for a in 0..k {
                    s1[a] += exps[ri] * x[(i, a)];
                    for b in 0..k {
                        s2[(a, b)] += exps[ri] * x[(i, a)] * x[(i, b)];
                    }
                }
            }
            let mut s0e = 0.0;
            let mut s1e = nalgebra::DVector::zeros(k);
            let mut s2e: DMatrix<f64> = DMatrix::zeros(k, k);
            for &i in &evts {
                let e = eta[i].exp();
                s0e += e;
                for a in 0..k {
                    s1e[a] += e * x[(i, a)];
                    for b in 0..k {
                        s2e[(a, b)] += e * x[(i, a)] * x[(i, b)];
                    }
                }
            }
            for &i in &evts {
                for a in 0..k {
                    grad[a] += x[(i, a)];
                }
            }
            for kk in 0..d {
                let w = kk as f64 / d as f64;
                let s0k = s0 - w * s0e;
                if s0k <= 0.0 {
                    continue;
                }
                let s1k = &s1 - w * &s1e;
                let s2k = &s2 - w * &s2e;
                let mean_k = &s1k / s0k;
                for a in 0..k {
                    grad[a] -= mean_k[a];
                    for b in 0..k {
                        hess[(a, b)] -= s2k[(a, b)] / s0k - mean_k[a] * mean_k[b];
                    }
                }
            }
        }
        let step = match mat_inverse(&hess) {
            Some(inv) => &inv * &grad,
            None => break,
        };
        let new_beta = &beta - &step;
        let diff = (&new_beta - &beta).iter().map(|d| d.abs()).fold(0.0, f64::max);
        beta = new_beta;
        if diff < 1e-8 {
            break;
        }
    }
    // Final Hessian for SEs
    let eta: Vec<f64> = (0..n).map(|i| (0..k).map(|j| x[(i, j)] * beta[j]).sum()).collect();
    let mut hess = DMatrix::zeros(k, k);
    let mut ev_times: Vec<f64> = (0..n).filter(|&i| events[i] == 1).map(|i| times[i]).collect();
    ev_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ev_times.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    for t in ev_times {
        let risk: Vec<usize> = (0..n).filter(|&i| times[i] >= t - 1e-10).collect();
        let evts: Vec<usize> = risk.iter().cloned().filter(|&i| (times[i] - t).abs() < 1e-10 && events[i] == 1).collect();
        let d = evts.len();
        let exps: Vec<f64> = risk.iter().map(|&i| eta[i].exp()).collect();
        let s0: f64 = exps.iter().sum();
        let mut s1 = nalgebra::DVector::zeros(k);
        let mut s2: DMatrix<f64> = DMatrix::zeros(k, k);
        for (ri, &i) in risk.iter().enumerate() {
            for a in 0..k {
                s1[a] += exps[ri] * x[(i, a)];
                for b in 0..k {
                    s2[(a, b)] += exps[ri] * x[(i, a)] * x[(i, b)];
                }
            }
        }
        let mut s0e = 0.0;
        let mut s1e = nalgebra::DVector::zeros(k);
        let mut s2e: DMatrix<f64> = DMatrix::zeros(k, k);
        for &i in &evts {
            let e = eta[i].exp();
            s0e += e;
            for a in 0..k {
                s1e[a] += e * x[(i, a)];
                for b in 0..k {
                    s2e[(a, b)] += e * x[(i, a)] * x[(i, b)];
                }
            }
        }
        for kk in 0..d {
            let w = kk as f64 / d as f64;
            let s0k = s0 - w * s0e;
            if s0k <= 0.0 {
                continue;
            }
            let s1k = &s1 - w * &s1e;
            let s2k = &s2 - w * &s2e;
            let mean_k = &s1k / s0k;
            for a in 0..k {
                for b in 0..k {
                    hess[(a, b)] -= s2k[(a, b)] / s0k - mean_k[a] * mean_k[b];
                }
            }
        }
    }
    let cov = mat_inverse(&hess).map(|m| -m).unwrap_or_else(|| DMatrix::zeros(k, k));
    let mut fmap = serde_json::Map::new();
    for (j, name) in preds.iter().enumerate() {
        let se = cov[(j, j)].max(0.0).sqrt();
        let z = if se > 0.0 { beta[j] / se } else { 0.0 };
        fmap.insert(name.clone(), json!({"c": r6(beta[j]), "hr": r6(beta[j].exp()), "s": r6(se), "z": r6(z), "p": r6(2.0 * norm_sf(z.abs())), "l": r6((beta[j] - 1.96 * se).exp()), "u": r6((beta[j] + 1.96 * se).exp())}));
    }
    // Harrell concordance as lifelines computes it: a pair (i event, j)
    // is comparable when times[j] > times[i], OR times[j] == times[i] and
    // j is censored (a censored subject at t is still at risk at t);
    // event-event pairs at the same time are not comparable. Tied
    // predicted risks count 0.5.
    let (mut conc, mut tot) = (0.0, 0.0);
    for i in 0..n {
        if events[i] != 1 {
            continue;
        }
        for j in 0..n {
            if times[j] > times[i] || ((times[j] - times[i]).abs() < 1e-10 && events[j] == 0) {
                tot += 1.0;
                if eta[j] < eta[i] {
                    conc += 1.0;
                } else if (eta[j] - eta[i]).abs() < 1e-12 {
                    conc += 0.5;
                }
            }
        }
    }
    let ci = if tot > 0.0 { conc / tot } else { 0.0 };
    // Partial log-likelihood (Efron): sum_events eta_i
    // minus sum_k log(S0 - (k/d) * S_events) per event time.
    let mut ll = 0.0;
    let mut ev_times2: Vec<f64> = (0..n).filter(|&i| events[i] == 1).map(|i| times[i]).collect();
    ev_times2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ev_times2.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    for t in ev_times2 {
        let risk: Vec<usize> = (0..n).filter(|&i| times[i] >= t - 1e-10).collect();
        let s0: f64 = risk.iter().map(|&i| eta[i].exp()).sum();
        let evts: Vec<usize> = risk.iter().cloned().filter(|&i| (times[i] - t).abs() < 1e-10 && events[i] == 1).collect();
        let d = evts.len();
        let s0e: f64 = evts.iter().map(|&i| eta[i].exp()).sum();
        for &i in &evts {
            ll += eta[i];
        }
        for kk in 0..d {
            let s0k = s0 - (kk as f64 / d as f64) * s0e;
            if s0k > 0.0 {
                ll -= s0k.ln();
            }
        }
    }
    let n_events: i64 = events.iter().sum();
    Ok(json!({"n": n, "events": n_events, "cindex": r6(ci), "ll": r6(ll), "f": Value::Object(fmap)}))
}

fn forest_action(tab: &Table, params: &serde_json::Map<String, Value>) -> Result<Value, String> {
    // Python keys: k, fe{est,se,l,u,z,p,method}, re{est,se,l,u,tau2,z,p,method},
    // hetero{Q,Qdf,Qp,I2,tau2}, forest[{label,yi,vi,se,w_fe,w_re}].
    // yi column candidates, vi candidates, label candidates (as in Python).
    let yi_col = ["yi", "effect", "estimate", "logrr", "logor"].iter().find(|c| tab.col_idx(c).is_some()).map(|c| c.to_string());
    let mut vi_col = ["vi", "var", "se2"].iter().find(|c| tab.col_idx(c).is_some()).map(|c| c.to_string());
    let lab_col = ["label", "study", "name", "id", "author"].iter().find(|c| tab.col_idx(c).is_some()).map(|c| c.to_string());
    let yi_col = yi_col.ok_or("forest requires effect column (yi) and variance column (vi or se)")?;
    let yi = tab.numeric(&yi_col)?;
    let vi: Vec<f64> = if let Some(vc) = vi_col.take() {
        tab.numeric(&vc)?
    } else if tab.col_idx("se").is_some() {
        tab.numeric("se")?.iter().map(|x| x * x).collect()
    } else {
        return Err("forest requires effect column (yi) and variance column (vi or se)".into());
    };
    let labels: Vec<String> = lab_col.map(|c| tab.col(&c).unwrap().iter().map(|x| x.key()).collect()).unwrap_or_else(|| (0..yi.len()).map(|i| format!("st{}", i + 1)).collect());
    let n = yi.len().min(vi.len()).min(labels.len());
    if n < 2 {
        return Err("forest needs >= 2 studies".into());
    }
    if vi[..n].iter().any(|v| *v <= 0.0) {
        return Err("forest variance (vi) must be > 0 for all studies".into());
    }
    let yi = &yi[..n];
    let vi = &vi[..n];
    let labels = &labels[..n];
    let w: Vec<f64> = vi.iter().map(|v| 1.0 / v).collect();
    let sw: f64 = w.iter().sum();
    let fe: f64 = w.iter().zip(yi.iter()).map(|(a, b)| a * b).sum::<f64>() / sw;
    let var_fe = 1.0 / sw;
    let q: f64 = w.iter().zip(yi.iter()).map(|(a, b)| a * (b - fe).powi(2)).sum();
    let dfree = (n - 1) as f64;
    let qp = chi2_sf(q, dfree);
    let denom = sw - w.iter().map(|a| a * a).sum::<f64>() / sw;
    let tau2 = if denom > 0.0 { (q - dfree) / denom } else { 0.0 };
    let wre: Vec<f64> = if tau2 > 0.0 { vi.iter().map(|v| 1.0 / (v + tau2)).collect() } else { w.clone() };
    let swr: f64 = wre.iter().sum();
    let re: f64 = wre.iter().zip(yi.iter()).map(|(a, b)| a * b).sum::<f64>() / swr;
    let var_re = 1.0 / swr;
    let i2 = if q > 0.0 { ((q - dfree) / q).max(0.0) } else { 0.0 };
    let z_fe = fe / var_fe.sqrt();
    let z_re = re / var_re.sqrt();
    let forest: Vec<Value> = (0..n).map(|i| {
        let wgt_fe = w[i] / sw;
        let wgt_re = wre[i] / swr;
        json!({"label": labels[i], "yi": r6(yi[i]), "vi": r6(vi[i]), "se": r6(vi[i].sqrt()), "w_fe": r6(wgt_fe), "w_re": r6(wgt_re)})
    }).collect();
    Ok(json!({"k": n,
        "fe": {"est": r6(fe), "se": r6(var_fe.sqrt()), "l": r6(fe - 1.96 * var_fe.sqrt()), "u": r6(fe + 1.96 * var_fe.sqrt()), "z": r6(z_fe), "p": r6(2.0 * norm_sf(z_fe.abs())), "method": "inverse_variance_fixed"},
        "re": {"est": r6(re), "se": r6(var_re.sqrt()), "l": r6(re - 1.96 * var_re.sqrt()), "u": r6(re + 1.96 * var_re.sqrt()), "tau2": r6(tau2), "z": r6(z_re), "p": r6(2.0 * norm_sf(z_re.abs())), "method": "der_simonian_laird"},
        "hetero": {"Q": r6(q), "Qdf": n - 1, "Qp": r6(qp), "I2": r6(i2), "tau2": r6(tau2)},
        "forest": forest}))
}

fn power_action(action: &str, params: &serde_json::Map<String, Value>) -> Result<Value, String> {
    // Output keys match the Python statsmodels wrappers exactly.
    // A value of 0 (or missing) means "solve for this one".
    let alpha = pf(params, "a", 0.05);
    let ratio = pf(params, "ratio", 1.0);
    let alt = ps(params, "alt");
    let alt = if alt.is_empty() { "two-sided".to_string() } else { alt };
    let za2 = norm_ppf(1.0 - alpha / 2.0);
    let za1 = norm_ppf(1.0 - alpha);
    match action {
        "ttest" => {
            let n = pf(params, "n", 0.0);
            let d = pf(params, "d_es", 0.0);
            let pw = pf(params, "pw", 0.0);
            let power_at = |n1v: f64, dv: f64| -> f64 {
                let n2 = n1v * ratio;
                let ncp = dv * (n1v * n2 / (n1v + n2)).sqrt();
                let df = n1v + n2 - 2.0;
                match alt.as_str() {
                    "larger" => 1.0 - nct_cdf(t_ppf(1.0 - alpha, df), df, ncp),
                    "smaller" => nct_cdf(t_ppf(alpha, df), df, ncp),
                    _ => {
                        let tc = t_ppf(1.0 - alpha / 2.0, df);
                        1.0 - nct_cdf(tc, df, ncp) + nct_cdf(-tc, df, ncp)
                    }
                }
            };
            if n == 0.0 {
                if d == 0.0 || pw == 0.0 {
                    return Err("ttest power: pass 0 for exactly one of n, d_es, pw".into());
                }
                let mut lo = 2.0f64;
                let mut hi = 10000.0f64;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    if power_at(mid, d) < pw { lo = mid; } else { hi = mid; }
                }
                return Ok(json!({"nobs1": r6(hi.ceil()), "effect_size": r6(d), "power": r6(pw), "alpha": r6(alpha), "ratio": r6(ratio), "alt": alt}));
            }
            if d == 0.0 {
                if pw == 0.0 {
                    return Err("ttest power: pass 0 for exactly one of n, d_es, pw".into());
                }
                let mut lo = 0.0f64;
                let mut hi = 5.0f64;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    if power_at(n, mid) < pw { lo = mid; } else { hi = mid; }
                }
                return Ok(json!({"nobs1": r6(n), "effect_size": r6((lo + hi) / 2.0), "power": r6(pw), "alpha": r6(alpha), "ratio": r6(ratio), "alt": alt}));
            }
            if alt != "two-sided" && alt != "larger" && alt != "smaller" {
                return Err("alt must be two-sided|larger|smaller".into());
            }
            Ok(json!({"nobs1": r6(n), "effect_size": r6(d), "power": r6(power_at(n, d)), "alpha": r6(alpha), "ratio": r6(ratio), "alt": alt}))
        }
        "anova" => {
            let n = pf(params, "n", 0.0);
            let f = pf(params, "f", 0.0);
            let pw = pf(params, "pw", 0.0);
            let k = pi(params, "k", 3);
            if k < 2 {
                return Err("anova k (groups) must be >= 2".into());
            }
            let kf = k as f64;
            let power_at = |ntot: f64, fv: f64| -> f64 {
                let df1 = kf - 1.0;
                let df2 = ntot - kf;
                if df2 <= 0.0 { return 0.0; }
                1.0 - ncf_cdf(f_ppf(1.0 - alpha, df1, df2), df1, df2, ntot * fv * fv)
            };
            if n == 0.0 {
                if f == 0.0 || pw == 0.0 {
                    return Err("anova power: pass 0 for exactly one of n, f, pw".into());
                }
                let mut lo = kf * 2.0;
                let mut hi = 100000.0f64;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    if power_at(mid, f) < pw { lo = mid; } else { hi = mid; }
                }
                return Ok(json!({"nobs": r6(hi), "effect_size": r6(f), "power": r6(pw), "alpha": r6(alpha), "k_groups": k}));
            }
            if f == 0.0 {
                if pw == 0.0 {
                    return Err("anova power: pass 0 for exactly one of n, f, pw".into());
                }
                let mut lo = 0.0f64;
                let mut hi = 5.0f64;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    if power_at(n, mid) < pw { lo = mid; } else { hi = mid; }
                }
                return Ok(json!({"nobs": r6(n), "effect_size": r6((lo + hi) / 2.0), "power": r6(pw), "alpha": r6(alpha), "k_groups": k}));
            }
            Ok(json!({"nobs": r6(n), "effect_size": r6(f), "power": r6(power_at(n, f)), "alpha": r6(alpha), "k_groups": k}))
        }
        "prop" => {
            let n = pf(params, "n", 0.0);
            let p1 = pf(params, "p1", 0.0);
            let p2 = pf(params, "p2", 0.0);
            let pw = pf(params, "pw", 0.0);
            if !(0.0 < p1 && p1 < 1.0) || !(0.0 < p2 && p2 < 1.0) {
                return Err("prop proportions must be in (0,1)".into());
            }
            // statsmodels proportion_effectsize(p2, p1)
            let es = 2.0 * p2.sqrt().asin() - 2.0 * p1.sqrt().asin();
            let power_at = |n1v: f64| -> f64 {
                let neff = n1v * ratio / (1.0 + ratio);
                norm_cdf(es.abs() * neff.sqrt() - za2) + norm_cdf(-es.abs() * neff.sqrt() - za2)
            };
            if n == 0.0 {
                if pw == 0.0 {
                    return Err("prop power: pass 0 for exactly one of n, pw (p1, p2 required)".into());
                }
                let mut lo = 2.0f64;
                let mut hi = 1000000.0f64;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    if power_at(mid) < pw { lo = mid; } else { hi = mid; }
                }
                return Ok(json!({"effect_size": r6(es), "nobs1": r6(hi.ceil()), "power": r6(pw), "alpha": r6(alpha), "p1": r6(p1), "p2": r6(p2), "ratio": r6(ratio)}));
            }
            let _ = za1;
            Ok(json!({"effect_size": r6(es), "nobs1": r6(n), "power": r6(power_at(n)), "alpha": r6(alpha), "p1": r6(p1), "p2": r6(p2), "ratio": r6(ratio)}))
        }
        _ => Err("unknown power action".into()),
    }
}
pub fn irt_score(tab: &Table, params: &serde_json::Map<String, Value>) -> Result<Value, String> {
    let diff_s = params.get("diff").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if diff_s.is_empty() {
        return Err("irt_score requires 'diff' (item difficulties, comma-separated)".into());
    }
    let method = params.get("method").and_then(|v| v.as_str()).unwrap_or("mle").to_string();
    if method != "mle" {
        return Err(format!("irt_score method '{}' is Not yet in the Rust port (mle only); eap/map require girth quadrature", method));
    }
    let model = params.get("model").and_then(|v| v.as_str()).unwrap_or("2pl").to_string();
    let parse_list = |s: &str| -> Result<Vec<f64>, String> {
        s.split(',').map(|x| x.trim().parse::<f64>().map_err(|_| format!("must be comma-separated numbers, got '{}'", s))).collect()
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
    // Binary matrix, listwise: rows with any non-0/1 or missing are dropped
    // (girth _prepare_for_girth drops NaN rows; _validate_binary maps 1/2->0/1).
    let headers = tab.headers.clone();
    if headers.len() != ndiff {
        return Err(format!("diff has {} values but data has {} items", ndiff, headers.len()));
    }
    let mut data: Vec<Vec<f64>> = Vec::new();
    for row in &tab.rows {
        let mut r = Vec::new();
        let mut ok = true;
        for c in row.iter().take(ndiff) {
            match c {
                Cell::Num(x) if x.is_finite() => r.push(*x),
                _ => { ok = false; break; }
            }
        }
        if ok {
            // 1/2 coding -> 0/1
            let uniq: Vec<f64> = {
                let mut u: Vec<f64> = data.iter().flatten().cloned().collect();
                u.sort_by(|a, b| a.partial_cmp(b).unwrap());
                u.dedup();
                u
            };
            let _ = uniq;
            data.push(r);
        }
    }
    // Apply 1/2 -> 0/1 conversion globally if all values are in {1,2}
    let all_vals: Vec<f64> = data.iter().flatten().cloned().collect();
    let mut uniq: Vec<f64> = all_vals.clone();
    uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
    uniq.dedup();
    if uniq.len() == 2 && uniq[0] == 1.0 && uniq[1] == 2.0 {
        for r in data.iter_mut() {
            for x in r.iter_mut() { *x -= 1.0; }
        }
    }
    let use_3pl = model == "3pl" && guess.iter().cloned().fold(0.0, f64::max) > 0.0;
    let mut abilities = Vec::new();
    let mut ses = Vec::new();
    for row in &data {
        if row.iter().all(|&y| y == 1.0) || row.iter().all(|&y| y == 0.0) {
            // girth ability_mle returns NaN for extreme scores (2PL path);
            // Python's own 3PL path clips to +-6 instead.
            if use_3pl {
                abilities.push(if row.iter().all(|&y| y == 1.0) { 6.0 } else { -6.0 });
            } else {
                abilities.push(f64::NAN);
            }
            ses.push(f64::NAN);
            continue;
        }
        let nll = |th: f64| -> f64 {
            let mut tot = 0.0;
            for j in 0..ndiff {
                let z = (disc[j] * (th - diff[j])).clamp(-30.0, 30.0);
                let mut p = if use_3pl {
                    let g = guess[j].clamp(0.0, 0.5);
                    g + (1.0 - g) / (1.0 + (-z).exp())
                } else {
                    1.0 / (1.0 + (-z).exp())
                };
                p = p.clamp(1e-10, 1.0 - 1e-10);
                tot -= row[j] * p.ln() + (1.0 - row[j]) * (1.0 - p).ln();
            }
            tot
        };
        // Golden-section minimisation on [-6, 6] (bounded, like scipy
        // minimize_scalar bounded used by the Python 3PL path).
        let (mut a, mut b) = (-6.0f64, 6.0f64);
        let gr = (5.0f64.sqrt() - 1.0) / 2.0;
        let (mut c1, mut c2) = (b - gr * (b - a), a + gr * (b - a));
        for _ in 0..200 {
            if (b - a).abs() < 1e-5 { break; }
            if nll(c1) < nll(c2) { b = c2; } else { a = c1; }
            c1 = b - gr * (b - a);
            c2 = a + gr * (b - a);
        }
        let th = (a + b) / 2.0;
        let h = 1e-4;
        let info = (nll(th + h) - 2.0 * nll(th) + nll(th - h)) / (h * h);
        let se = if info > 0.0 { 1.0 / info.sqrt() } else { f64::NAN };
        abilities.push(th);
        ses.push(se);
    }
    let mut out = json!({"method": if use_3pl { "mle-3pl" } else { "mle" }, "model": model, "n": abilities.len(), "a": abilities.iter().map(|v| r6(*v)).collect::<Vec<_>>()});
    if use_3pl {
        out.as_object_mut().unwrap().insert("se".into(), json!(ses.iter().map(|v| r6(*v)).collect::<Vec<_>>()));
    }
    Ok(out)
}
pub fn irt_ctt(tab: &Table) -> Result<Value, String> {
    // Python keys: n_items, n_people, items[{p, r}], alpha, mean, sd.
    // p = item mean (difficulty), r = point-biserial vs corrected total.
    let headers = tab.headers.clone();
    if headers.is_empty() {
        return Err("no columns".into());
    }
    // Binary matrix with listwise deletion of non-finite rows (girth prep),
    // and 1/2 -> 0/1 conversion.
    let mut data: Vec<Vec<f64>> = Vec::new();
    for row in &tab.rows {
        let mut r = Vec::new();
        let mut ok = true;
        for c in row.iter().take(headers.len()) {
            match c {
                Cell::Num(x) if x.is_finite() => r.push(*x),
                _ => { ok = false; break; }
            }
        }
        if ok { data.push(r); }
    }
    let mut uniq: Vec<f64> = data.iter().flatten().cloned().collect();
    uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
    uniq.dedup();
    if uniq.len() == 2 && uniq[0] == 1.0 && uniq[1] == 2.0 {
        for r in data.iter_mut() { for x in r.iter_mut() { *x -= 1.0; } }
    }
    let n_people = data.len();
    let n_items = headers.len();
    if n_people < 2 {
        return Err("Need ≥2 cases".into());
    }
    let totals: Vec<f64> = data.iter().map(|r| r.iter().sum()).collect();
    let mut items = Vec::new();
    for j in 0..n_items {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        let pval = mean(&col);
        let corrected: Vec<f64> = (0..n_people).map(|i| totals[i] - col[i]).collect();
        let r_pb = if sd1(&corrected) > 0.0 && sd1(&col) > 0.0 {
            pearson(&col, &corrected).0
        } else { 0.0 };
        items.push(json!({"p": r6(pval), "r": r6(r_pb)}));
    }
    let item_vars: f64 = (0..n_items).map(|j| var_ddof1(&data.iter().map(|r| r[j]).collect::<Vec<_>>())).sum();
    let total_var = var_ddof1(&totals);
    let mut out = json!({"n_items": n_items, "n_people": n_people, "items": items, "mean": r6(mean(&totals)), "sd": r6(sd1(&totals))});
    if total_var > 0.0 && n_items > 1 {
        let alpha = (n_items as f64 / (n_items as f64 - 1.0)) * (1.0 - item_vars / total_var);
        out.as_object_mut().unwrap().insert("alpha".into(), r6(alpha));
    }
    Ok(out)
}
