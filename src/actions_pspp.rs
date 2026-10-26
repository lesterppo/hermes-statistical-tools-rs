// PSPP actions. Params: a small struct mirroring pspp_run kwargs.
use crate::*;
use nalgebra::DMatrix;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Default, Clone)]
pub struct P {
    pub a: String,
    pub v: Vec<String>,
    pub v1: String,
    pub g: Vec<Cell>,
    pub o: String,
    pub w: String,
    pub p: Vec<String>,
    pub c: Vec<String>,
    pub mu: Option<f64>,
    pub paired: bool,
    pub posthoc: String,
    pub rotate: String,
    pub n: i64,
    pub pos: f64,
    pub pos_ref: f64,
    pub effect_col: String,
    pub se_col: String,
    pub power_test: String,
    pub alpha: f64,
    pub es: f64,
    pub ratio: f64,
    pub k: i64,
    pub p0: f64,
    pub p1: f64,
    pub calc: String,
    pub or_val: Option<f64>,
    pub rr_val: Option<f64>,
    pub tvar: String,
    pub pw: f64, // power target (pspp default 0.80)
}
impl P {
    pub fn vs(&self) -> String {
        self.v.first().cloned().unwrap_or_default()
    }
}

fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let pooled = (((a.len() - 1) as f64 * var_ddof1(a) + (b.len() - 1) as f64 * var_ddof1(b))
        / (a.len() + b.len() - 2) as f64)
        .sqrt();
    if pooled > 0.0 {
        (mean(b) - mean(a)) / pooled
    } else {
        0.0
    }
}

pub fn dispatch(t: &str, tab: &Table, p: &P) -> Result<Value, String> {
    match t {
        "desc" => desc(tab, p),
        "freq" => freq(tab, p),
        "examine" => examine(tab, p),
        "means" => means(tab, p),
        "crosstab" => crosstab(tab, p),
        "ttest" => ttest(tab, p),
        "pttest" => pttest(tab, p),
        "ttest1" => ttest1(tab, p),
        "anova" => anova(tab, p),
        "mw" => mw(tab, p),
        "kw" => kw(tab, p),
        "wilcoxon" => wilcoxon(tab, p),
        "friedman" => friedman(tab, p),
        "sign" => sign(tab, p),
        "ks" => ks(tab, p),
        "runs" => runs(tab, p),
        "corr" => corr(tab, p),
        "spearman" => spearman(tab, p),
        "partial" => partial(tab, p),
        "reg" => reg(tab, p),
        "logistic" => logistic(tab, p),
        "poisson" => poisson(tab, p),
        "factor" => factor(tab, p),
        "reliability" => reliability(tab, p),
        "roc" => roc(tab, p),
        "eval" => eval_diag(tab, p),
        "surv" => surv_pspp(tab, p),
        "meta" => meta(tab, p),
        "blandaltman" => blandaltman(tab, p),
        "kappa" => kappa(tab, p),
        "evalue" => evalue(tab, p),
        "power" => power_pspp(p),
        "rank" => rank(tab, p),
        "hedgesg" => hedgesg(tab, p),
        "omega" => omega(tab, p),
        _ => Err(format!("Unknown test '{}'", t)),
    }
}

fn desc(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v.is_empty() {
        return Err("v (variable list) required".into());
    }
    let mut out = Vec::new();
    for var in &p.v {
        if let Ok(vals) = tab.numeric(var) {
            if vals.is_empty() {
                out.push(json!({"n": var, "N": 0}));
            } else {
                out.push(json!({"n": var, "N": vals.len(), "m": r4(mean(&vals)), "sd": r4(sd1(&vals)), "min": r4(vals.iter().fold(f64::INFINITY, |a, &b| a.min(b))), "max": r4(vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))}));
            }
        }
    }
    Ok(json!({"vars": out}))
}

fn freq(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || tab.col_idx(&p.a).is_none() {
        return Err("a (variable) required".into());
    }
    let col = tab.col(&p.a).unwrap();
    let mut counts: BTreeMap<String, (Cell, i64)> = BTreeMap::new();
    let mut total = 0i64;
    for c in &col {
        if c.is_valid() {
            let key = match c {
                Cell::Num(x) => {
                    if x.fract() == 0.0 {
                        format!("{}", *x as i64)
                    } else {
                        format!("{}", x)
                    }
                }
                Cell::Txt(s) => s.clone(),
                Cell::Miss => continue,
            };
            let e = counts.entry(key).or_insert((c.clone(), 0));
            e.1 += 1;
            total += 1;
        }
    }
    // Python sorts by (isinstance(str), value): numbers first, then strings.
    let mut entries: Vec<(Cell, i64)> = counts.into_values().collect();
    entries.sort_by(|(a, _), (b, _)| match (a, b) {
        (Cell::Num(x), Cell::Num(y)) => x.partial_cmp(y).unwrap(),
        (Cell::Num(_), Cell::Txt(_)) => std::cmp::Ordering::Less,
        (Cell::Txt(_), Cell::Num(_)) => std::cmp::Ordering::Greater,
        (Cell::Txt(x), Cell::Txt(y)) => x.cmp(y),
        _ => std::cmp::Ordering::Equal,
    });
    let mut freq = Vec::new();
    for (cell, n) in entries {
        let pct = if total > 0 { format!("{:.1}%", (n as f64 / total as f64 * 1000.0).round() / 10.0) } else { "0%".to_string() };
        let v = match cell {
            Cell::Num(x) => json!(x),
            Cell::Txt(s) => json!(s),
            Cell::Miss => json!(null),
        };
        freq.push(json!({"v": v, "n": n, "pct": pct}));
    }
    Ok(json!({"freq": freq}))
}

fn examine(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v.is_empty() {
        return Err("v (variable list) required".into());
    }
    let mut out = Vec::new();
    for var in &p.v {
        if let Ok(vals) = tab.numeric(var) {
            let n = vals.len();
            if n < 3 {
                out.push(json!({"n": var, "N": n, "e": "need ≥3 values"}));
                continue;
            }
            let mut e = json!({
                "n": var, "N": n,
                "m": r4(mean(&vals)), "sd": r4(sd1(&vals)),
                "min": r4(vals.iter().cloned().fold(f64::INFINITY, f64::min)),
                "max": r4(vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
                "sk": r4(skew(&vals)), "kt": r4(kurtosis_fisher(&vals)),
                "sw": Value::Null, "swp": Value::Null,
            });
            for q in [5, 10, 25, 50, 75, 90, 95] {
                e.as_object_mut().unwrap().insert(format!("p{}", q), r4(percentile(&vals, q as f64)));
            }
            out.push(e);
        }
    }
    Ok(json!({"vars": out}))
}

fn means(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (group var) and v (value var) required".into());
    }
    let groups = tab.unique_groups(&p.a)?;
    let mut out = Vec::new();
    for g in &groups {
        let vals = tab.groups_numeric(&p.v1, &p.a, std::slice::from_ref(g))?[0].clone();
        if vals.is_empty() {
            continue;
        }
        let gv = match g {
            Cell::Num(x) => json!(x),
            Cell::Txt(s) => json!(s),
            Cell::Miss => json!(null),
        };
        out.push(json!({"g": gv, "N": vals.len(), "m": r4(mean(&vals)), "sd": r4(sd1(&vals)), "min": r4(vals.iter().cloned().fold(f64::INFINITY, f64::min)), "max": r4(vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max))}));
    }
    Ok(json!({"groups": out}))
}

fn crosstab(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() {
        return Err("a (row var) and v (col var) required".into());
    }
    if tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err(format!("Variables not found: '{}', '{}'", p.a, p.v1));
    }
    let raws_a = tab.col(&p.a).unwrap();
    let raws_v = tab.col(&p.v1).unwrap();
    let wcol = if !p.w.is_empty() { tab.col(&p.w) } else { None };
    let mut counts: BTreeMap<(String, String), f64> = BTreeMap::new();
    let n = raws_a.len().min(raws_v.len());
    for i in 0..n {
        if !raws_a[i].is_valid() || !raws_v[i].is_valid() {
            continue;
        }
        let wt = match &wcol {
            Some(w) => match w.get(i) {
                Some(Cell::Num(x)) if x.is_finite() => *x,
                _ => continue,
            },
            None => 1.0,
        };
        *counts.entry((raws_a[i].key(), raws_v[i].key())).or_insert(0.0) += wt;
    }
    let mut row_labels: Vec<String> = counts.keys().map(|k| k.0.clone()).collect();
    let mut col_labels: Vec<String> = counts.keys().map(|k| k.1.clone()).collect();
    row_labels.sort();
    row_labels.dedup();
    col_labels.sort();
    col_labels.dedup();
    let (nr, nc) = (row_labels.len(), col_labels.len());
    if nr < 2 || nc < 2 {
        return Err(format!("Need ≥2×2 table, got {}×{}", nr, nc));
    }
    let mut obs = vec![vec![0.0; nc]; nr];
    let mut total = 0.0;
    for (r, c) in row_labels.iter().enumerate() {
        for (cc, v) in col_labels.iter().enumerate() {
            let x = counts.get(&(c.clone(), v.clone())).cloned().unwrap_or(0.0);
            obs[r][cc] = x;
            total += x;
        }
    }
    let row_sums: Vec<f64> = obs.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<f64> = (0..nc).map(|j| obs.iter().map(|r| r[j]).sum()).collect();
    let mut chi2 = 0.0;
    for i in 0..nr {
        for j in 0..nc {
            let e = row_sums[i] * col_sums[j] / total;
            if e > 0.0 {
                chi2 += (obs[i][j] - e).powi(2) / e;
            }
        }
    }
    let dof = (nr - 1) * (nc - 1);
    let mut res = json!({"n": total as i64, "rows": row_labels, "cols": col_labels, "χ²": r4(chi2), "df": dof, "p": r4(chi2_sf(chi2, dof as f64)), "phi": r4((chi2 / total).sqrt()), "V": r4((chi2 / (total * (nr.min(nc) - 1) as f64)).sqrt())});
    if nr == 2 && nc == 2 {
        // Fisher exact, two-sided (sum of hypergeometric probs <= observed prob)
        let (a, b, c, d) = (obs[0][0], obs[0][1], obs[1][0], obs[1][1]);
        let (a, b, c, d) = (a as i64, b as i64, c as i64, d as i64);
        let n1 = a + b;
        let m1 = a + c;
        let ntot = a + b + c + d;
        let p_obs = (ln_comb(n1 as usize, a as usize) + ln_comb((ntot - n1) as usize, (m1 - a) as usize) - ln_comb(ntot as usize, m1 as usize)).exp();
        let mut fp = 0.0;
        for aa in 0.max(m1 - (ntot - n1))..=(n1.min(m1)) {
            let pp = (ln_comb(n1 as usize, aa as usize) + ln_comb((ntot - n1) as usize, (m1 - aa) as usize) - ln_comb(ntot as usize, m1 as usize)).exp();
            if pp <= p_obs + 1e-12 {
                fp += pp;
            }
        }
        res.as_object_mut().unwrap().insert("fp".into(), r4(fp.min(1.0)));
        let (mut av, mut bv, mut cv, mut dv) = (a as f64, b as f64, c as f64, d as f64);
        if bv == 0.0 || cv == 0.0 {
            av += 0.5;
            bv += 0.5;
            cv += 0.5;
            dv += 0.5;
        }
        if bv > 0.0 && cv > 0.0 {
            let orv = (av * dv) / (bv * cv);
            let se = (1.0 / av + 1.0 / bv + 1.0 / cv + 1.0 / dv).sqrt();
            res.as_object_mut().unwrap().insert("OR".into(), r4(orv));
            res.as_object_mut().unwrap().insert("OR_ci".into(), json!([r4((orv.ln() - 1.96 * se).exp()), r4((orv.ln() + 1.96 * se).exp())]));
        }
        if p.paired {
            let (bv2, cv2) = (obs[0][1], obs[1][0]);
            if bv2 + cv2 > 0.0 {
                let mcn = (0.0f64.max((bv2 - cv2).abs() - 1.0)).powi(2) / (bv2 + cv2);
                res.as_object_mut().unwrap().insert("mchi2".into(), r4(mcn));
                res.as_object_mut().unwrap().insert("mp".into(), r4(chi2_sf(mcn, 1.0)));
            }
        }
    }
    Ok(res)
}

fn ttest(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || p.g.len() != 2 {
        return Err("a (group var), v (value var), g=[g1,g2] required".into());
    }
    if tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err(format!("Variables not found: '{}', '{}'", p.a, p.v1));
    }
    let grps = tab.groups_numeric(&p.v1, &p.a, &p.g)?;
    if grps[0].len() < 2 || grps[1].len() < 2 {
        return Err("Need ≥2 values per group".into());
    }
    let (x, y) = (&grps[0], &grps[1]);
    let (v1, v2) = (var_ddof1(x), var_ddof1(y));
    let (n1, n2) = (x.len(), y.len());
    if v1 == 0.0 && v2 == 0.0 {
        return Ok(json!({"n1": n1, "n2": n2, "m1": r4(mean(x)), "m2": r4(mean(y)), "t": 0, "df": 0, "p": 1.0, "d": 0, "md": 0, "ci_l": Value::Null, "ci_u": Value::Null, "warn": "zero variance in both groups"}));
    }
    if v1 == 0.0 || v2 == 0.0 {
        return Ok(json!({"n1": n1, "n2": n2, "m1": r4(mean(x)), "m2": r4(mean(y)), "t": Value::Null, "df": Value::Null, "p": Value::Null, "d": Value::Null, "md": Value::Null, "ci_l": Value::Null, "ci_u": Value::Null, "warn": "zero variance in one group"}));
    }
    let se = (v1 / n1 as f64 + v2 / n2 as f64).sqrt();
    let t = (mean(x) - mean(y)) / se;
    let df = (v1 / n1 as f64 + v2 / n2 as f64).powi(2)
        / ((v1 / n1 as f64).powi(2) / (n1 - 1) as f64 + (v2 / n2 as f64).powi(2) / (n2 - 1) as f64);
    let md = mean(y) - mean(x);
    let tc = t_ppf(0.975, df);
    Ok(json!({"n1": n1, "n2": n2, "m1": r4(mean(x)), "m2": r4(mean(y)), "sd1": r4(sd1(x)), "sd2": r4(sd1(y)), "t": r4(t), "df": r4(df), "p": r4(2.0 * t_sf(t.abs(), df)), "d": r4(cohens_d(x, y)), "md": r4(md), "ci_l": r4(md - tc * se), "ci_u": r4(md + tc * se)}))
}

fn pttest(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (var1), v (var2) required".into());
    }
    let x1 = tab.numeric(&p.a)?;
    let x2 = tab.numeric(&p.v1)?;
    let n = x1.len().min(x2.len());
    if n < 3 {
        return Err("Need ≥3 pairs".into());
    }
    let diff: Vec<f64> = (0..n).map(|i| x1[i] - x2[i]).collect();
    let sd = sd1(&diff);
    let t = if sd > 0.0 { mean(&diff) / (sd / (n as f64).sqrt()) } else { 0.0 };
    let d = if sd > 0.0 { mean(&diff) / sd } else { 0.0 };
    Ok(json!({"n": n, "m1": r4(mean(&x1[..n])), "m2": r4(mean(&x2[..n])), "t": r4(t), "df": n - 1, "p": r4(2.0 * t_sf(t.abs(), (n - 1) as f64)), "d": r4(d), "md": r4(mean(&diff))}))
}

fn ttest1(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v1.is_empty() || tab.col_idx(&p.v1).is_none() {
        return Err("v (variable) required".into());
    }
    let vals = tab.numeric(&p.v1)?;
    let n = vals.len();
    if n < 2 {
        return Err("Need ≥2 values".into());
    }
    let mu = p.mu.unwrap_or(0.0);
    let t = (mean(&vals) - mu) / (sd1(&vals) / (n as f64).sqrt());
    Ok(json!({"n": n, "m": r4(mean(&vals)), "sd": r4(sd1(&vals)), "t": r4(t), "df": n - 1, "p": r4(2.0 * t_sf(t.abs(), (n - 1) as f64)), "mu": mu, "md": r4(mean(&vals) - mu)}))
}

fn posthoc(groups: &[Vec<f64>], labels: &[Cell], method: &str) -> Vec<Value> {
    let k = groups.len();
    let total_n: usize = groups.iter().map(|g| g.len()).sum();
    let mse = groups.iter().map(|g| g.iter().map(|x| (x - mean(g)).powi(2)).sum::<f64>()).sum::<f64>() / (total_n - k) as f64;
    let mut raw: Vec<f64> = Vec::new();
    let mut pairs: Vec<Value> = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            let md = mean(&groups[j]) - mean(&groups[i]);
            let pv = if method == "gh" {
                let (n1, n2) = (groups[i].len(), groups[j].len());
                let (v1, v2) = (var_ddof1(&groups[i]), var_ddof1(&groups[j]));
                let s = v1 / n1 as f64 + v2 / n2 as f64;
                let tgh = if s > 0.0 { md.abs() / s.sqrt() } else { 0.0 };
                let df = if s > 0.0 { s.powi(2) / ((v1 / n1 as f64).powi(2) / (n1 - 1) as f64 + (v2 / n2 as f64).powi(2) / (n2 - 1) as f64) } else { 1.0 };
                studentized_range_sf(tgh * 2.0f64.sqrt(), k, df)
            } else {
                let se = (mse / 2.0 * (1.0 / groups[i].len() as f64 + 1.0 / groups[j].len() as f64)).sqrt();
                let q = if se > 0.0 { md.abs() / se } else { 0.0 };
                studentized_range_sf(q, k, (total_n - k) as f64)
            };
            raw.push(pv);
            let (li, lj) = (match &labels[i] { Cell::Num(x) => json!(x), Cell::Txt(s) => json!(s), Cell::Miss => json!(null) }, match &labels[j] { Cell::Num(x) => json!(x), Cell::Txt(s) => json!(s), Cell::Miss => json!(null) });
            pairs.push(json!({"i": li, "j": lj, "d": r4(md), "p": 0}));
        }
    }
    let adj: Vec<f64> = if method == "bonferroni" {
        raw.iter().map(|x| (x * raw.len() as f64).min(1.0)).collect()
    } else {
        raw.clone()
    };
    for (idx, pv) in adj.iter().enumerate() {
        pairs[idx].as_object_mut().unwrap().insert("p".into(), r4(*pv));
    }
    pairs
}

fn anova(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (group var), v (value var) required".into());
    }
    let uq = tab.unique_groups(&p.a)?;
    if uq.len() < 2 {
        return Err(format!("Need ≥2 groups, got {:?}", uq.iter().map(|c| c.key()).collect::<Vec<_>>()));
    }
    let groups = tab.groups_numeric(&p.v1, &p.a, &uq)?;
    if groups.iter().any(|g| g.len() < 2) {
        return Err("Need ≥2 values per group".into());
    }
    let all: Vec<f64> = groups.iter().flatten().cloned().collect();
    let gm = mean(&all);
    let ss_b: f64 = groups.iter().map(|g| g.len() as f64 * (mean(g) - gm).powi(2)).sum();
    let ss_w: f64 = groups.iter().map(|g| g.iter().map(|x| (x - mean(g)).powi(2)).sum::<f64>()).sum();
    let (df1, df2) = (groups.len() - 1, all.len() - groups.len());
    let f = (ss_b / df1 as f64) / (ss_w / df2 as f64);
    let eta2 = if ss_b + ss_w > 0.0 { ss_b / (ss_b + ss_w) } else { 0.0 };
    let method = if p.posthoc.is_empty() { "tukey" } else { p.posthoc.as_str() };
    let mut res = json!({"F": r4(f), "df1": df1, "df2": df2, "p": r4(f_sf(f, df1 as f64, df2 as f64)), "eta2": r4(eta2)});
    if ["tukey", "bonferroni", "gh", ""].contains(&method) {
        res.as_object_mut().unwrap().insert("posthoc".into(), json!(posthoc(&groups, &uq, method)));
    }
    Ok(res)
}

// Exact Mann-Whitney U distribution (no ties) via subset-sum DP over ranks.
fn mw_exact_p(u_obs: f64, n1: usize, n2: usize) -> f64 {
    let n = n1 + n2;
    // dp[i][s] = number of subsets of size i with rank-sum s
    let max_sum = n * (n + 1) / 2;
    let mut dp = vec![vec![0f64; max_sum + 1]; n1 + 1];
    dp[0][0] = 1.0;
    for r in 1..=n {
        for i in (1..=n1).rev() {
            for s in (r..=max_sum).rev() {
                dp[i][s] += dp[i - 1][s - r];
            }
        }
    }
    let total = dp[n1].iter().sum::<f64>();
    let s_obs = u_obs as usize + n1 * (n1 + 1) / 2;
    let p_le: f64 = dp[n1][..=s_obs.min(max_sum)].iter().sum::<f64>() / total;
    let p_ge: f64 = dp[n1][s_obs.min(max_sum)..].iter().sum::<f64>() / total;
    (2.0 * p_le.min(p_ge)).min(1.0)
}

fn mw(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || p.g.len() != 2 {
        return Err("a (group var), v (value var), g=[g1,g2] required".into());
    }
    if tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err(format!("Variables not found: '{}', '{}'", p.a, p.v1));
    }
    let grps = tab.groups_numeric(&p.v1, &p.a, &p.g)?;
    if grps[0].is_empty() || grps[1].is_empty() {
        return Err("Need ≥1 value per group".into());
    }
    let (x, y) = (&grps[0], &grps[1]);
    let mut all = x.clone();
    all.extend(y.iter().cloned());
    let ranks = rankdata(&all);
    let w: f64 = ranks[..x.len()].iter().sum();
    let n1 = x.len();
    let n2 = y.len();
    let u = w - n1 as f64 * (n1 as f64 + 1.0) / 2.0;
    let has_ties = {
        let mut s = all.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s.windows(2).any(|w| w[0] == w[1])
    };
    let pv = if !has_ties && n1 + n2 <= 60 {
        mw_exact_p(u, n1, n2)
    } else {
        // Asymptotic with continuity correction + tie correction (scipy path)
        let mut tie_corr = 0.0;
        {
            let mut s = all.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut i = 0;
            while i < s.len() {
                let mut j = i;
                while j + 1 < s.len() && s[j + 1] == s[i] {
                    j += 1;
                }
                let t = (j - i + 1) as f64;
                tie_corr += t * (t * t - 1.0);
                i = j + 1;
            }
        }
        let n = (n1 + n2) as f64;
        let sigma = ((n1 as f64 * n2 as f64 * (n + 1.0) - tie_corr * n1 as f64 * n2 as f64 / (n * (n - 1.0))) / 12.0).sqrt();
        let z = if sigma > 0.0 { (u - n1 as f64 * n2 as f64 / 2.0 - 0.5 * (u - n1 as f64 * n2 as f64 / 2.0).signum()) / sigma } else { 0.0 };
        2.0 * norm_sf(z.abs())
    };
    let z_mw = (u - n1 as f64 * n2 as f64 / 2.0) / ((n1 as f64 * n2 as f64 * (n1 + n2 + 1) as f64) / 12.0).sqrt();
    Ok(json!({"U": r4(u), "W": r4(w), "Z": r4(z_mw), "p": r4(pv)}))
}

fn kw(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (group var), v (value var) required".into());
    }
    let uq = tab.unique_groups(&p.a)?;
    if uq.len() < 2 {
        return Err(format!("Need ≥2 groups, got {:?}", uq.iter().map(|c| c.key()).collect::<Vec<_>>()));
    }
    let groups = tab.groups_numeric(&p.v1, &p.a, &uq)?;
    let all: Vec<f64> = groups.iter().flatten().cloned().collect();
    let ranks = rankdata(&all);
    let n = all.len() as f64;
    let mut off = 0usize;
    let mut h = 0.0;
    for g in &groups {
        let rs: f64 = ranks[off..off + g.len()].iter().sum();
        h += rs * rs / g.len() as f64;
        off += g.len();
    }
    h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0);
    // Tie correction (scipy applies it)
    let mut tie_corr = 0.0;
    {
        let mut s = all.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut i = 0;
        while i < s.len() {
            let mut j = i;
            while j + 1 < s.len() && s[j + 1] == s[i] {
                j += 1;
            }
            let t = (j - i + 1) as f64;
            tie_corr += t * t * t - t;
            i = j + 1;
        }
    }
    if tie_corr > 0.0 {
        h /= 1.0 - tie_corr / (n * n * n - n);
    }
    Ok(json!({"H": r4(h), "df": groups.len() - 1, "p": r4(chi2_sf(h, (groups.len() - 1) as f64))}))
}

// Wilcoxon: exact DP over rank sums when no ties/zeros, else normal approx.
fn wilcoxon_stat_p(diffs: &[f64]) -> (f64, f64) {
    let nz: Vec<f64> = diffs.iter().cloned().filter(|d| *d != 0.0).collect();
    let n = nz.len();
    let ranks = rankdata(&nz.iter().map(|d| d.abs()).collect::<Vec<_>>());
    let w_pos: f64 = nz.iter().zip(ranks.iter()).filter(|(d, _)| **d > 0.0).map(|(_, r)| *r).sum();
    let has_ties = {
        let mut s: Vec<f64> = nz.iter().map(|d| d.abs()).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s.windows(2).any(|w| w[0] == w[1])
    };
    if !has_ties && n <= 50 {
        let total = n * (n + 1) / 2;
        let mut dp = vec![0f64; total + 1];
        dp[0] = 1.0;
        for r in 1..=n {
            for s in (r..=total).rev() {
                dp[s] += dp[s - r];
            }
        }
        let tot: f64 = dp.iter().sum();
        let w = w_pos as usize;
        // scipy exact two-sided: sum of probabilities of sums at least as
        // extreme as min(w, total-w) on both tails
        let m = w.min(total - w);
        let mut pv = 0.0;
        for s in 0..=total {
            if s <= m || s >= total - m {
                pv += dp[s];
            }
        }
        (w_pos, (pv / tot).min(1.0))
    } else {
        let mu = n as f64 * (n as f64 + 1.0) / 4.0;
        let sigma = (n as f64 * (n as f64 + 1.0) * (2.0 * n as f64 + 1.0) / 24.0).sqrt();
        let z = if sigma > 0.0 { (w_pos - mu - 0.5 * (w_pos - mu).signum()) / sigma } else { 0.0 };
        (w_pos, 2.0 * norm_sf(z.abs()))
    }
}

fn wilcoxon(tab: &Table, p: &P) -> Result<Value, String> {
    if !p.a.is_empty() && !p.v1.is_empty() && tab.col_idx(&p.a).is_some() && tab.col_idx(&p.v1).is_some() {
        let x1 = tab.numeric(&p.a)?;
        let x2 = tab.numeric(&p.v1)?;
        let n = x1.len().min(x2.len());
        let diffs: Vec<f64> = (0..n).map(|i| x1[i] - x2[i]).collect();
        let (w, pv) = wilcoxon_stat_p(&diffs);
        return Ok(json!({"n": n, "W": r4(w), "p": r4(pv), "type": "paired"}));
    }
    if !p.v1.is_empty() && tab.col_idx(&p.v1).is_some() {
        let vals = tab.numeric(&p.v1)?;
        let mu = p.mu.unwrap_or(0.0);
        let diffs: Vec<f64> = vals.iter().map(|x| x - mu).collect();
        let (w, pv) = wilcoxon_stat_p(&diffs);
        return Ok(json!({"n": vals.len(), "W": r4(w), "p": r4(pv), "type": "one-sample", "mu": mu}));
    }
    Err("a (var1) + v (var2) for paired, or v (var) + mu for one-sample".into())
}

fn friedman(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v.len() < 2 {
        return Err("v (2+ repeated measure variables) required".into());
    }
    for var in &p.v {
        if tab.col_idx(var).is_none() {
            return Err(format!("Variable '{}' not found", var));
        }
    }
    let arrays: Vec<Vec<f64>> = p.v.iter().map(|v| tab.numeric(v).unwrap_or_default()).collect();
    let n = arrays.iter().map(|a| a.len()).min().unwrap_or(0);
    if n < 3 {
        return Err("Need ≥3 subjects".into());
    }
    let k = arrays.len() as f64;
    let mut rank_sums = vec![0.0; arrays.len()];
    for i in 0..n {
        let row: Vec<f64> = arrays.iter().map(|a| a[i]).collect();
        let r = rankdata(&row);
        for j in 0..arrays.len() {
            rank_sums[j] += r[j];
        }
    }
    let chi2 = 12.0 / (n as f64 * k * (k + 1.0)) * rank_sums.iter().map(|s| s * s).sum::<f64>() - 3.0 * n as f64 * (k + 1.0);
    Ok(json!({"χ²": r4(chi2), "df": arrays.len() - 1, "p": r4(chi2_sf(chi2, (arrays.len() - 1) as f64)), "n": n}))
}

fn sign(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (var1), v (var2) required".into());
    }
    let x1 = tab.numeric(&p.a)?;
    let x2 = tab.numeric(&p.v1)?;
    let n = x1.len().min(x2.len());
    let (mut pos, mut neg, mut ties) = (0i64, 0i64, 0i64);
    for i in 0..n {
        let d = x1[i] - x2[i];
        if d > 0.0 {
            pos += 1;
        } else if d < 0.0 {
            neg += 1;
        } else {
            ties += 1;
        }
    }
    let total = (pos + neg) as usize;
    let pv = if total > 0 { (2.0 * binom_cdf((pos.min(neg)) as usize, total, 0.5)).min(1.0) } else { 1.0 };
    Ok(json!({"pos": pos, "neg": neg, "ties": ties, "n": total, "p": r4(pv)}))
}

fn ks_stat(a: &[f64], b: &[f64]) -> f64 {
    let mut xs = a.to_vec();
    let mut ys = b.to_vec();
    xs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ys.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut d = 0.0;
    let mut i = 0usize;
    let mut j = 0usize;
    while i < xs.len() || j < ys.len() {
        let v = if j >= ys.len() || (i < xs.len() && xs[i] <= ys[j]) { xs[i] } else { ys[j] };
        while i < xs.len() && xs[i] <= v {
            i += 1;
        }
        while j < ys.len() && ys[j] <= v {
            j += 1;
        }
        let diff = (i as f64 / xs.len() as f64 - j as f64 / ys.len() as f64).abs();
        if diff > d {
            d = diff;
        }
    }
    d
}

fn ks(tab: &Table, p: &P) -> Result<Value, String> {
    if !p.a.is_empty() && !p.v1.is_empty() && p.g.len() >= 2 {
        if tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
            return Err(format!("Variables not found: '{}', '{}'", p.a, p.v1));
        }
        let grps = tab.groups_numeric(&p.v1, &p.a, &p.g[..2])?;
        let d = ks_stat(&grps[0], &grps[1]);
        let ne = grps[0].len() as f64 * grps[1].len() as f64 / (grps[0].len() + grps[1].len()) as f64;
        let lam = (ne.sqrt() + 0.12 + 0.11 / ne.sqrt()) * d;
        return Ok(json!({"D": r4(d), "p": r4(ks_sf(lam)), "type": "two-sample"}));
    }
    if !p.v1.is_empty() && tab.col_idx(&p.v1).is_some() {
        let vals = tab.numeric(&p.v1)?;
        let m = mean(&vals);
        let s = sd1(&vals);
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len() as f64;
        let mut d = 0.0f64;
        for (i, x) in sorted.iter().enumerate() {
            let f = norm_cdf((x - m) / s);
            d = d.max(((i + 1) as f64 / n - f).abs()).max((f - i as f64 / n).abs());
        }
        let lam = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * d;
        return Ok(json!({"D": r4(d), "p": r4(ks_sf(lam)), "type": "one-sample (vs normal)"}));
    }
    Err("a+v+g for two-sample, or v for one-sample vs normal".into())
}

fn runs(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v1.is_empty() || tab.col_idx(&p.v1).is_none() {
        return Err("v (variable) required".into());
    }
    let vals = tab.numeric(&p.v1)?;
    let n = vals.len();
    if n < 2 {
        return Err("Need ≥2 values".into());
    }
    let med = median_sorted(vals.clone());
    let above: Vec<bool> = vals.iter().map(|x| *x > med).collect();
    let mut r = 1i64;
    for i in 1..n {
        if above[i] != above[i - 1] {
            r += 1;
        }
    }
    let n1 = above.iter().filter(|b| **b).count() as f64;
    let n2 = n as f64 - n1;
    if n1 < 1.0 || n2 < 1.0 {
        return Err("Need both above and below median".into());
    }
    let mu = 2.0 * n1 * n2 / (n1 + n2) + 1.0;
    let sigma = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2) / ((n1 + n2).powi(2) * (n1 + n2 - 1.0))).sqrt();
    let z = if sigma > 0.0 { (r as f64 - mu) / sigma } else { 0.0 };
    Ok(json!({"runs": r, "median": r4(med), "Z": r4(z), "p": r4(2.0 * norm_sf(z.abs()))}))
}

fn corr(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v.len() < 2 {
        return Err("v (2+ variables) required".into());
    }
    for var in &p.v {
        if tab.col_idx(var).is_none() {
            return Err(format!("Variable '{}' not found", var));
        }
    }
    let arrays: Vec<Vec<f64>> = p.v.iter().map(|v| tab.numeric(v).unwrap_or_default()).collect();
    let n = arrays.iter().map(|a| a.len()).min().unwrap_or(0);
    let mut pairs = Vec::new();
    for i in 0..p.v.len() {
        for j in (i + 1)..p.v.len() {
            let (r, pv) = pearson(&arrays[i][..n], &arrays[j][..n]);
            pairs.push(json!({"v": format!("{}-{}", p.v[i], p.v[j]), "r": r4(r), "p": r4(pv)}));
        }
    }
    Ok(json!({"pairs": pairs}))
}

fn spearman(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v.len() < 2 {
        return Err("v (2+ variables) required".into());
    }
    for var in &p.v {
        if tab.col_idx(var).is_none() {
            return Err(format!("Variable '{}' not found", var));
        }
    }
    let arrays: Vec<Vec<f64>> = p.v.iter().map(|v| tab.numeric(v).unwrap_or_default()).collect();
    let n = arrays.iter().map(|a| a.len()).min().unwrap_or(0);
    let ranks: Vec<Vec<f64>> = arrays.iter().map(|a| rankdata(&a[..n])).collect();
    let mut pairs = Vec::new();
    for i in 0..p.v.len() {
        for j in (i + 1)..p.v.len() {
            let (rho, pv) = pearson(&ranks[i], &ranks[j]);
            pairs.push(json!({"v": format!("{}-{}", p.v[i], p.v[j]), "rho": r4(rho), "p": r4(pv)}));
        }
    }
    Ok(json!({"pairs": pairs}))
}

fn partial(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v.is_empty() || p.c.is_empty() {
        return Err("v (variables) and c (control variables) required".into());
    }
    let all_vars: Vec<String> = p.v.iter().cloned().chain(p.c.iter().cloned()).collect();
    for var in &all_vars {
        if tab.col_idx(var).is_none() {
            return Err(format!("Variable '{}' not found", var));
        }
    }
    let arrays: Vec<Vec<f64>> = all_vars.iter().map(|v| tab.numeric(v).unwrap_or_default()).collect();
    let n = arrays.iter().map(|a| a.len()).min().unwrap_or(0);
    let m = all_vars.len();
    let mut cov = DMatrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mi = mean(&arrays[i][..n]);
            let mj = mean(&arrays[j][..n]);
            cov[(i, j)] = (0..n).map(|t| (arrays[i][t] - mi) * (arrays[j][t] - mj)).sum::<f64>() / (n - 1) as f64;
        }
    }
    let prec = mat_inverse(&cov).ok_or("Precision matrix singular — variables may be collinear")?;
    let df = n as i64 - all_vars.len() as i64;
    let mut pairs = Vec::new();
    for i in 0..p.v.len() {
        for j in (i + 1)..p.v.len() {
            let rp = -prec[(i, j)] / (prec[(i, i)] * prec[(j, j)]).sqrt();
            let pv = if df > 0 && rp.abs() < 1.0 {
                let tv = rp * ((df as f64) / (1.0 - rp * rp)).sqrt();
                Some(2.0 * t_sf(tv.abs(), df as f64))
            } else {
                None
            };
            pairs.push(json!({"v": format!("{}-{}", p.v[i], p.v[j]), "r": r4(rp), "p": r4o(pv), "control": p.c}));
        }
    }
    Ok(json!({"pairs": pairs, "df": df}))
}

fn reg(tab: &Table, p: &P) -> Result<Value, String> {
    if p.o.is_empty() || p.p.is_empty() {
        return Err("o (outcome) and p (predictors) required".into());
    }
    if tab.col_idx(&p.o).is_none() {
        return Err(format!("Outcome '{}' not found", p.o));
    }
    for pv in &p.p {
        if tab.col_idx(pv).is_none() {
            return Err(format!("Predictor '{}' not found", pv));
        }
    }
    let y = tab.numeric(&p.o)?;
    let xcols: Vec<Vec<f64>> = p.p.iter().map(|pv| tab.numeric(pv).unwrap_or_default()).collect();
    let n = y.len().min(xcols.iter().map(|c| c.len()).min().unwrap_or(0));
    let y = &y[..n];
    let xcols: Vec<Vec<f64>> = xcols.iter().map(|c| c[..n].to_vec()).collect();
    let k = xcols.len() + 1;
    if n <= k {
        return Err(format!("Need more observations ({}) than parameters ({})", n, k));
    }
    let o = ols(y, &xcols, true).ok_or("Matrix singular — predictors may be collinear")?;
    let ss_tot: f64 = y.iter().map(|e| (e - mean(y)).powi(2)).sum();
    let ss_reg = ss_tot - o.ss_res;
    let f = if k > 1 && o.ss_res > 0.0 { (ss_reg / (k - 1) as f64) / (o.ss_res / (n - k) as f64) } else { 0.0 };
    let fp = if k > 1 && o.ss_res > 0.0 { f_sf(f, (k - 1) as f64, (n - k) as f64) } else { 0.0 };
    let mut coefs = Vec::new();
    let names: Vec<String> = std::iter::once("(Intercept)".to_string()).chain(p.p.iter().cloned()).collect();
    for (i, name) in names.iter().enumerate() {
        coefs.push(json!({"n": name, "B": r4(o.beta[i]), "SE": r4(o.se[i]), "t": r4(o.t[i]), "p": r4(o.p[i])}));
    }
    let mut res = json!({"R": r4(o.r2.max(0.0).sqrt()), "R2": r4(o.r2), "aR2": r4(o.adj_r2), "F": r4(f), "Fp": r4(fp), "n": n, "coef": coefs});
    if p.p.len() > 1 {
        let mut vifs = Map::new();
        for (idx, pv) in p.p.iter().enumerate() {
            let others: Vec<Vec<f64>> = p.p.iter().enumerate().filter(|(j, _)| *j != idx).map(|(_, q)| tab.numeric(q).unwrap_or_default()[..n].to_vec()).collect();
            let ysub = tab.numeric(pv).unwrap_or_default()[..n].to_vec();
            let vif = match ols(&ysub, &others, true) {
                Some(o2) if o2.r2 < 1.0 => Some(1.0 / (1.0 - o2.r2)),
                _ => None,
            };
            vifs.insert(pv.clone(), r4o(vif));
        }
        res.as_object_mut().unwrap().insert("vif".into(), Value::Object(vifs));
    }
    Ok(res)
}

// Generic IRLS GLM: link log (poisson) or logit (logistic).
struct Glm {
    beta: Vec<f64>,
    se: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    converged: bool,
    mu: Vec<f64>,
    loglik: f64,
}
fn glm_fit(y: &[f64], x: &DMatrix<f64>, link: &str) -> Option<Glm> {
    let (n, k) = (x.nrows(), x.ncols());
    let mut beta = DVector::zeros(k);
    if link == "poisson" {
        beta[0] = mean(y).max(0.1).ln();
    }
    let mut converged = false;
    for _ in 0..50 {
        let eta = x * &beta;
        let eta_c: Vec<f64> = eta.iter().map(|e| e.clamp(-30.0, 30.0)).collect();
        let mu: Vec<f64> = eta_c.iter().map(|e| if link == "logit" { 1.0 / (1.0 + (-e).exp()) } else { e.exp() }).collect();
        let mu_c: Vec<f64> = mu.iter().map(|m| if link == "logit" { m.clamp(1e-10, 1.0 - 1e-10) } else { m.clamp(1e-10, 1e10) }).collect();
        // W and working response
        let mut xtwx = DMatrix::zeros(k, k);
        let mut xtwz = DVector::zeros(k);
        for i in 0..n {
            let w = if link == "logit" { mu_c[i] * (1.0 - mu_c[i]) } else { mu_c[i] };
            let z = eta_c[i] + (y[i] - mu_c[i]) / if link == "logit" { mu_c[i] * (1.0 - mu_c[i]) } else { mu_c[i] };
            for a in 0..k {
                xtwz[a] += x[(i, a)] * w * z;
                for b in 0..k {
                    xtwx[(a, b)] += x[(i, a)] * w * x[(i, b)];
                }
            }
        }
        let inv = mat_inverse(&xtwx)?;
        let beta_new = &inv * xtwz;
        let diff = (&beta_new - &beta).iter().map(|d| d.abs()).fold(0.0, f64::max);
        beta = beta_new;
        if diff < 1e-6 {
            converged = true;
            break;
        }
    }
    let eta = x * &beta;
    let eta_c: Vec<f64> = eta.iter().map(|e| e.clamp(-30.0, 30.0)).collect();
    let mu: Vec<f64> = eta_c.iter().map(|e| if link == "logit" { 1.0 / (1.0 + (-e).exp()) } else { e.exp() }).collect();
    let mu_c: Vec<f64> = mu.iter().map(|m| if link == "logit" { m.clamp(1e-10, 1.0 - 1e-10) } else { m.clamp(1e-10, 1e10) }).collect();
    let mut xtwx = DMatrix::zeros(k, k);
    for i in 0..n {
        let w = if link == "logit" { mu_c[i] * (1.0 - mu_c[i]) } else { mu_c[i] };
        for a in 0..k {
            for b in 0..k {
                xtwx[(a, b)] += x[(i, a)] * w * x[(i, b)];
            }
        }
    }
    let cov = mat_inverse(&xtwx)?;
    let mut se = vec![0.0; k];
    let mut z = vec![0.0; k];
    let mut pv = vec![0.0; k];
    for i in 0..k {
        se[i] = cov[(i, i)].max(0.0).sqrt();
        z[i] = if se[i] > 0.0 { beta[i] / se[i] } else { 0.0 };
        pv[i] = 2.0 * norm_sf(z[i].abs());
    }
    let loglik = if link == "logit" {
        y.iter().zip(mu_c.iter()).map(|(yi, m)| yi * m.ln() + (1.0 - yi) * (1.0 - m).ln()).sum()
    } else {
        y.iter().zip(mu_c.iter()).map(|(yi, m)| yi * m.ln() - m).sum()
    };
    Some(Glm { beta: beta.iter().cloned().collect(), se, z, p: pv, converged, mu: mu_c, loglik })
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

fn logistic(tab: &Table, p: &P) -> Result<Value, String> {
    if p.o.is_empty() || p.p.is_empty() {
        return Err("o (binary outcome) and p (predictors) required".into());
    }
    if tab.col_idx(&p.o).is_none() {
        return Err(format!("Outcome '{}' not found", p.o));
    }
    for pv in &p.p {
        if tab.col_idx(pv).is_none() {
            return Err(format!("Predictor '{}' not found", pv));
        }
    }
    let y_raw = tab.numeric(&p.o)?;
    let mut uniq: Vec<i64> = y_raw.iter().filter(|x| x.is_finite()).map(|x| *x as i64).collect();
    uniq.sort();
    uniq.dedup();
    if uniq.len() != 2 {
        return Err(format!("Outcome must be binary (2 unique values), got {:?}", uniq));
    }
    let (lo, hi) = (uniq[0] as f64, uniq[1] as f64);
    let y_full: Vec<f64> = y_raw.iter().map(|v| if *v == hi { 1.0 } else { 0.0 }).collect();
    let xcols_full: Vec<Vec<f64>> = p.p.iter().map(|pv| tab.numeric(pv).unwrap_or_default()).collect();
    let n = y_full.len().min(xcols_full.iter().map(|c| c.len()).min().unwrap_or(0));
    let y = &y_full[..n];
    let xcols: Vec<Vec<f64>> = xcols_full.iter().map(|c| c[..n].to_vec()).collect();
    let x = design(&xcols, n);
    let g = glm_fit(y, &x, "logit").ok_or("Matrix singular — predictors may be collinear or complete separation")?;
    let null_p = mean(y).clamp(1e-10, 1.0 - 1e-10);
    let null_ll: f64 = y.iter().map(|yi| yi * null_p.ln() + (1.0 - yi) * (1.0 - null_p).ln()).sum();
    let mcf = if null_ll != 0.0 { 1.0 - g.loglik / null_ll } else { 0.0 };
    let names: Vec<String> = std::iter::once("(Intercept)".to_string()).chain(p.p.iter().cloned()).collect();
    let mut coefs = Vec::new();
    for (i, name) in names.iter().enumerate() {
        coefs.push(json!({"n": name, "B": r4(g.beta[i]), "SE": r4(g.se[i]), "z": r4(g.z[i]), "p": r4(g.p[i]), "OR": r4(g.beta[i].exp())}));
    }
    let mut res = json!({"n": n, "R2_mcfadden": r4(mcf), "coef": coefs});
    let pred_pos: Vec<f64> = g.mu.iter().map(|m| if *m >= 0.5 { 1.0 } else { 0.0 }).collect();
    let perfect = pred_pos.iter().zip(y.iter()).all(|(a, b)| a == b);
    let max_se = g.se.iter().cloned().fold(0.0, f64::max);
    let max_b = g.beta.iter().map(|b| b.abs()).fold(0.0, f64::max);
    if !g.converged || perfect || max_se > 100.0 || max_b > 10.0 {
        res.as_object_mut().unwrap().insert("warn_separation".into(), json!(format!("possible complete/quasi-complete separation (converged={}, perfect_classification={}, max|B|={:.2}, maxSE={:.2}): coefficients/SEs may be inflated; consider Firth penalized logistic regression or collapsing sparse categories", g.converged, perfect, max_b, max_se)));
    }
    Ok(res)
}

fn poisson(tab: &Table, p: &P) -> Result<Value, String> {
    if p.o.is_empty() || p.p.is_empty() {
        return Err("o (count outcome), p (predictors) required".into());
    }
    if tab.col_idx(&p.o).is_none() {
        return Err(format!("Outcome '{}' not found", p.o));
    }
    for pv in &p.p {
        if tab.col_idx(pv).is_none() {
            return Err(format!("Predictor '{}' not found", pv));
        }
    }
    let y_full = tab.numeric(&p.o)?;
    let xcols_full: Vec<Vec<f64>> = p.p.iter().map(|pv| tab.numeric(pv).unwrap_or_default()).collect();
    let n = y_full.len().min(xcols_full.iter().map(|c| c.len()).min().unwrap_or(0));
    let k = xcols_full.len() + 1;
    if n <= k {
        return Err(format!("Need more observations ({}) than parameters ({})", n, k));
    }
    let y = &y_full[..n];
    let xcols: Vec<Vec<f64>> = xcols_full.iter().map(|c| c[..n].to_vec()).collect();
    let x = design(&xcols, n);
    let g = glm_fit(y, &x, "poisson").ok_or("Matrix singular")?;
    let dev: f64 = y.iter().zip(g.mu.iter()).map(|(yi, m)| 2.0 * (yi * (yi.max(1e-10) / m).ln() - (yi - m))).sum();
    let ymean = mean(y);
    let null_dev: f64 = y.iter().map(|yi| 2.0 * (yi * (yi.max(1e-10) / ymean).ln() - (yi - ymean))).sum();
    let names: Vec<String> = std::iter::once("(Intercept)".to_string()).chain(p.p.iter().cloned()).collect();
    let mut coefs = Vec::new();
    for (i, name) in names.iter().enumerate() {
        coefs.push(json!({"n": name, "B": r4(g.beta[i]), "SE": r4(g.se[i]), "z": r4(g.z[i]), "p": r4(g.p[i]), "IRR": r4(g.beta[i].exp())}));
    }
    Ok(json!({"n": n, "deviance": r4(dev), "null_deviance": r4(null_dev), "coef": coefs}))
}

fn factor(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v.len() < 2 {
        return Err("v (2+ variables) required".into());
    }
    for var in &p.v {
        if tab.col_idx(var).is_none() {
            return Err(format!("Variable '{}' not found", var));
        }
    }
    let arrays: Vec<Vec<f64>> = p.v.iter().map(|v| tab.numeric(v).unwrap_or_default()).collect();
    let n = arrays.iter().map(|a| a.len()).min().unwrap_or(0);
    let m = p.v.len();
    // Standardized data -> correlation matrix
    let mut corr = DMatrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let (a, b) = (&arrays[i][..n], &arrays[j][..n]);
            let (r, _) = pearson(a, b);
            corr[(i, j)] = if i == j { 1.0 } else { r };
        }
    }
    let eig = nalgebra::SymmetricEigen::new(corr.clone());
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());
    let n_comp = if p.n > 0 { (p.n as usize).min(m) } else { m };
    let mut loadings = DMatrix::zeros(m, n_comp);
    for (ci, &ei) in order.iter().take(n_comp).enumerate() {
        let ev = eig.eigenvalues[ei].max(0.0);
        for r in 0..m {
            loadings[(r, ci)] = eig.eigenvectors[(r, ei)] * ev.sqrt();
        }
    }
    // Varimax (same iteration as original)
    if p.rotate == "varimax" && n_comp > 1 {
        let mut l = loadings.clone();
        for _ in 0..50 {
            let mut l_new = l.clone();
            // L2, M, SVD of L^T (L*M)
            let l2 = l.map(|x| x * x);
            // Python: M = L2 - mean(L2, axis=0) / len(variables), i.e. the
            // column mean is divided by n_vars a second time.
            let col_mean: Vec<f64> = (0..n_comp).map(|j| (0..m).map(|i| l2[(i, j)]).sum::<f64>() / (m as f64 * m as f64)).collect();
            let mut lm = DMatrix::zeros(m, n_comp);
            for i in 0..m {
                for j in 0..n_comp {
                    lm[(i, j)] = l[(i, j)] * (l2[(i, j)] - col_mean[j]);
                }
            }
            let prod = l.transpose() * lm;
            let svd = prod.svd(true, true);
            if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
                let rot = &u * &vt;
                l_new = &l * rot;
            }
            let diff = (&l_new - &l).iter().map(|d| d.abs()).fold(0.0, f64::max);
            l = l_new;
            if diff < 1e-6 {
                break;
            }
        }
        loadings = l;
    }
    let eigenvalues: Vec<f64> = order.iter().take(n_comp).map(|&ei| eig.eigenvalues[ei]).collect();
    let total_var: f64 = eigenvalues.iter().take(m).sum::<f64>().max(eig.eigenvalues.iter().sum::<f64>());
    let total_var = if total_var == 0.0 { eig.eigenvalues.iter().sum() } else { total_var };
    let mut comps = Vec::new();
    for ci in 0..n_comp {
        let var_expl: f64 = (0..m).map(|r| loadings[(r, ci)].powi(2)).sum();
        let mut load_map = Map::new();
        for (j, var) in p.v.iter().enumerate() {
            load_map.insert(var.clone(), r4(loadings[(j, ci)]));
        }
        comps.push(json!({"component": ci + 1, "eigenvalue": r4(eigenvalues[ci]), "var_pct": r4(if total_var > 0.0 { var_expl / total_var * 100.0 } else { 0.0 }), "loadings": Value::Object(load_map)}));
    }
    Ok(json!({"components": comps, "total_var": r4(total_var)}))
}

fn reliability(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v.len() < 2 {
        return Err("v (2+ item variables) required".into());
    }
    for var in &p.v {
        if tab.col_idx(var).is_none() {
            return Err(format!("Variable '{}' not found", var));
        }
    }
    let arrays: Vec<Vec<f64>> = p.v.iter().map(|v| tab.numeric(v).unwrap_or_default()).collect();
    let n = arrays.iter().map(|a| a.len()).min().unwrap_or(0);
    let k = arrays.len() as f64;
    let item_vars: f64 = arrays.iter().map(|a| var_ddof1(&a[..n])).sum();
    let totals: Vec<f64> = (0..n).map(|i| arrays.iter().map(|a| a[i]).sum()).collect();
    let total_var = var_ddof1(&totals);
    let alpha = if total_var > 0.0 { (k / (k - 1.0)) * (1.0 - item_vars / total_var) } else { 0.0 };
    let mut item_total = Vec::new();
    for (i, var) in p.v.iter().enumerate() {
        let without: Vec<f64> = (0..n).map(|t| arrays.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, a)| a[t]).sum()).collect();
        let (r, pv) = pearson(&arrays[i][..n], &without);
        item_total.push(json!({"n": var, "r": r4(r), "p": r4(pv)}));
    }
    Ok(json!({"alpha": r4(alpha), "n_items": p.v.len(), "n_cases": n, "item_total": item_total}))
}

fn roc(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (binary outcome), v (predictor) required".into());
    }
    let y_raw = tab.numeric(&p.a)?;
    let s_raw = tab.numeric(&p.v1)?;
    let n = y_raw.len().min(s_raw.len());
    let mut pairs: Vec<(f64, f64)> = (0..n).map(|i| (s_raw[i], if (y_raw[i] - p.pos).abs() < 0.001 { 1.0 } else { 0.0 })).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n_pos = pairs.iter().filter(|(_, y)| *y == 1.0).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return Err("Need both positive and negative cases".into());
    }
    let (mut tpr, mut fpr, mut thresh) = (vec![0.0], vec![0.0], vec![f64::INFINITY]);
    let (mut tp, mut fp) = (0usize, 0usize);
    let mut i = 0;
    while i < n {
        let s = pairs[i].0;
        let mut j = i;
        while j < n && pairs[j].0 == s {
            if pairs[j].1 == 1.0 {
                tp += 1;
            } else {
                fp += 1;
            }
            j += 1;
        }
        tpr.push(tp as f64 / n_pos as f64);
        fpr.push(fp as f64 / n_neg as f64);
        thresh.push(s);
        i = j;
    }
    let mut auc = 0.0;
    for k in 0..fpr.len() - 1 {
        auc += (fpr[k + 1] - fpr[k]) * (tpr[k + 1] + tpr[k]) / 2.0;
    }
    let js: Vec<f64> = tpr.iter().zip(fpr.iter()).map(|(t, f)| t - f).collect();
    let best = js.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
    Ok(json!({"AUC": r4(auc), "n_pos": n_pos, "n_neg": n_neg, "youden": r4(js[best]), "best_sens": r4(tpr[best]), "best_spec": r4(1.0 - fpr[best]), "threshold": if best > 0 { r4(thresh[best]) } else { Value::Null }}))
}

fn eval_diag(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (reference standard), v (test result) required".into());
    }
    let refr = tab.numeric(&p.a)?;
    let test = tab.numeric(&p.v1)?;
    let n = refr.len().min(test.len());
    let (mut tp, mut fp, mut fn_, mut tn) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n {
        let tpos = (test[i] - p.pos).abs() < 0.001;
        let rpos = (refr[i] - p.pos_ref).abs() < 0.001;
        match (tpos, rpos) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let total = tp + fp + fn_ + tn;
    if total == 0 {
        return Err("No valid data".into());
    }
    let f = |num: i64, den: i64| -> Option<f64> { if den > 0 { Some(num as f64 / den as f64) } else { None } };
    let sens = f(tp, tp + fn_);
    let spec = f(tn, tn + fp);
    let ppv = f(tp, tp + fp);
    let npv = f(tn, tn + fn_);
    let prev = Some((tp + fn_) as f64 / total as f64);
    let lr_pos = match (sens, spec) { (Some(s), Some(sp)) if sp < 1.0 => Some(s / (1.0 - sp)), _ => None };
    let lr_neg = match (sens, spec) { (Some(s), Some(sp)) if sp > 0.0 => Some((1.0 - s) / sp), _ => None };
    let dor = if fp > 0 && fn_ > 0 { Some(tp as f64 * tn as f64 / (fp as f64 * fn_ as f64)) } else { None };
    let acc = Some((tp + tn) as f64 / total as f64);
    let youden = match (sens, spec) { (Some(s), Some(sp)) => Some(s + sp - 1.0), _ => None };
    let wilson = |ph: f64, nn: i64| -> [Value; 2] {
        let z = 1.96;
        let den = 1.0 + z * z / nn as f64;
        let lo = (ph + z * z / (2.0 * nn as f64) - z * (ph * (1.0 - ph) / nn as f64 + z * z / (4.0 * nn as f64 * nn as f64)).sqrt()) / den;
        let hi = (ph + z * z / (2.0 * nn as f64) + z * (ph * (1.0 - ph) / nn as f64 + z * z / (4.0 * nn as f64 * nn as f64)).sqrt()) / den;
        [r4(lo.max(0.0)), r4(hi.min(1.0))]
    };
    let sens_ci = sens.map(|s| json!(wilson(s, tp + fn_)));
    let spec_ci = spec.map(|s| json!(wilson(s, tn + fp)));
    Ok(json!({"TP": tp, "FP": fp, "FN": fn_, "TN": tn, "n": total, "sens": r4o(sens), "sens_ci": sens_ci.unwrap_or(Value::Null), "spec": r4o(spec), "spec_ci": spec_ci.unwrap_or(Value::Null), "ppv": r4o(ppv), "npv": r4o(npv), "lr_pos": r4o(lr_pos), "lr_neg": r4o(lr_neg), "dor": r4o(dor), "acc": r4o(acc), "youden": r4o(youden), "prev": r4o(prev)}))
}

// PSPP surv: KM list + median + Mantel-Haenszel log-rank (no Cox here).
pub fn km_curve(times: &[f64], events: &[i64]) -> (Vec<(f64, f64)>, Option<f64>) {
    let n = times.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());
    let mut out = Vec::new();
    let mut surv = 1.0;
    let mut at_risk = n as f64;
    let mut i = 0;
    let mut median = None;
    while i < n {
        let t = times[order[i]];
        let mut d = 0i64;
        let mut j = i;
        while j < n && (times[order[j]] - t).abs() < 1e-10 {
            if events[order[j]] == 1 {
                d += 1;
            }
            j += 1;
        }
        if d > 0 {
            surv *= (at_risk - d as f64) / at_risk;
        }
        out.push((t, surv));
        if median.is_none() && surv <= 0.5 {
            median = Some(t);
        }
        at_risk -= d as f64;
        // Note: Python's pspp surv decrements at_risk only by d (events), not
        // by censored at this time — but censored subjects at time t are
        // removed from the risk set for later times only via the loop's
        // counting? Actually Python sets n_at_risk -= d only, and appends a
        // point for EVERY distinct time (even censor-only times). The
        // at-risk for the next event time should exclude censored subjects
        // whose time < next t. We emulate by tracking: at each step, the
        // subjects counted are those with time >= current t. Recompute
        // below is handled by the caller for log-rank; for the KM curve,
        // Python's surv at censor-only times uses the stale at_risk, which
        // equals (# with time >= t) only if no censoring before t... we
        // keep Python's exact behaviour: at_risk decreases by d only.
        i = j;
    }
    (out, median)
}

fn surv_pspp(tab: &Table, p: &P) -> Result<Value, String> {
    if p.o.is_empty() || p.v1.is_empty() {
        return Err("o (event indicator: 1=event, 0=censored) and v (time) required".into());
    }
    if tab.col_idx(&p.o).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err(format!("Variables not found: '{}', '{}'", p.o, p.v1));
    }
    let times = tab.numeric(&p.v1)?;
    let ev_raw = tab.numeric(&p.o)?;
    let n = times.len().min(ev_raw.len());
    let times = &times[..n];
    let events: Vec<i64> = ev_raw[..n].iter().map(|e| if *e >= 0.5 { 1 } else { 0 }).collect();
    if events.iter().cloned().collect::<std::collections::HashSet<_>>().len() < 2 {
        return Err("Need both events (1) and censorings (0)".into());
    }
    let (curve, median) = km_curve(times, &events);
    let mut res = json!({"km": curve.iter().map(|(t, s)| json!({"t": t, "s": r4(*s)})).collect::<Vec<_>>()});
    if let Some(m) = median {
        res.as_object_mut().unwrap().insert("median_surv".into(), json!(m));
    }
    if !p.a.is_empty() && tab.col_idx(&p.a).is_some() {
        let grp_col = tab.col(&p.a).unwrap();
        let groups_raw = &grp_col[..n.min(grp_col.len())];
        let mut labels: Vec<String> = groups_raw.iter().filter(|c| c.is_valid()).map(|c| c.key()).collect();
        labels.sort();
        labels.dedup();
        if labels.len() >= 2 {
            // Reconstruct sorted order like Python
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());
            let times_f: Vec<f64> = order.iter().map(|&i| times[i]).collect();
            let ev_f: Vec<i64> = order.iter().map(|&i| events[i]).collect();
            let pos_group: Vec<String> = order.iter().map(|&i| groups_raw[i].key()).collect();
            if labels.len() == 2 {
                let (mut o1, mut e1, mut v) = (0.0, 0.0, 0.0);
                let mut ev_times: Vec<f64> = times_f.iter().zip(ev_f.iter()).filter(|(_, &e)| e == 1).map(|(t, _)| *t).collect();
                ev_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ev_times.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
                for t in ev_times {
                    let risk: Vec<usize> = (0..n).filter(|&i| times_f[i] >= t - 1e-10).collect();
                    let n_at = risk.len() as f64;
                    let n1_at = risk.iter().filter(|&&i| pos_group[i] == labels[1]).count() as f64;
                    let n0_at = n_at - n1_at;
                    let d_at = risk.iter().filter(|&&i| (times_f[i] - t).abs() < 1e-10 && ev_f[i] == 1).count() as f64;
                    let d1_at = risk.iter().filter(|&&i| (times_f[i] - t).abs() < 1e-10 && ev_f[i] == 1 && pos_group[i] == labels[1]).count() as f64;
                    if n_at > 1.0 && d_at > 0.0 && n1_at > 0.0 && n0_at > 0.0 {
                        e1 += d_at * n1_at / n_at;
                        o1 += d1_at;
                        v += (n1_at * n0_at * d_at * (n_at - d_at)) / (n_at * n_at * (n_at - 1.0));
                    }
                }
                let chi2 = if v > 0.0 { (o1 - e1).powi(2) / v } else { 0.0 };
                res.as_object_mut().unwrap().insert("logrank".into(), json!({"χ²": r4(chi2), "df": 1, "p": r4(chi2_sf(chi2, 1.0)), "groups": labels[..2].to_vec(), "O1": r4(o1), "E1": r4(e1)}));
            } else {
                let k = labels.len();
                let (mut os, mut es, mut vs) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
                let mut ev_times: Vec<f64> = times_f.iter().zip(ev_f.iter()).filter(|(_, &e)| e == 1).map(|(t, _)| *t).collect();
                ev_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ev_times.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
                for t in ev_times {
                    let risk: Vec<usize> = (0..n).filter(|&i| times_f[i] >= t - 1e-10).collect();
                    let n_at = risk.len() as f64;
                    let d_at = risk.iter().filter(|&&i| (times_f[i] - t).abs() < 1e-10 && ev_f[i] == 1).count() as f64;
                    if n_at > 1.0 && d_at > 0.0 {
                        for (gi, gl) in labels.iter().enumerate() {
                            let n_at_g = risk.iter().filter(|&&i| pos_group[i] == *gl).count() as f64;
                            let d_at_g = risk.iter().filter(|&&i| (times_f[i] - t).abs() < 1e-10 && ev_f[i] == 1 && pos_group[i] == *gl).count() as f64;
                            if n_at_g > 0.0 {
                                es[gi] += d_at * n_at_g / n_at;
                                os[gi] += d_at_g;
                                vs[gi] += (n_at_g * (n_at - n_at_g) * d_at * (n_at - d_at)) / (n_at * n_at * (n_at - 1.0));
                            }
                        }
                    }
                }
                let chi2: f64 = (0..k).filter(|&gi| vs[gi] > 0.0).map(|gi| (os[gi] - es[gi]).powi(2) / vs[gi]).sum();
                res.as_object_mut().unwrap().insert("logrank".into(), json!({"χ²": r4(chi2), "df": k - 1, "p": r4(chi2_sf(chi2, (k - 1) as f64)), "groups": labels}));
            }
        }
    }
    Ok(res)
}

pub fn meta_from(effects: &[f64], ses: &[f64], labels: &[String]) -> Value {
    meta_from_nd(effects, ses, labels, 4)
}
pub fn meta_from_nd(effects: &[f64], ses: &[f64], labels: &[String], nd: i32) -> Value {
    let rv = |x: f64| -> Value {
        match rnd(x, nd) {
            Some(v) => json!(v),
            None => Value::Null,
        }
    };
    let k = effects.len();
    let w_fixed: Vec<f64> = ses.iter().map(|s| 1.0 / (s * s)).collect();
    let sw: f64 = w_fixed.iter().sum();
    let fe_est: f64 = w_fixed.iter().zip(effects.iter()).map(|(w, e)| w * e).sum::<f64>() / sw;
    let fe_se = (1.0 / sw).sqrt();
    let fe_z = fe_est / fe_se;
    let q: f64 = w_fixed.iter().zip(effects.iter()).map(|(w, e)| w * (e - fe_est).powi(2)).sum();
    let df = (k - 1) as f64;
    let p_het = chi2_sf(q, df);
    let i2 = if q > 0.0 { ((q - df) / q * 100.0).max(0.0) } else { 0.0 };
    let c = sw - w_fixed.iter().map(|w| w * w).sum::<f64>() / sw;
    let tau2 = if c > 0.0 { ((q - df) / c).max(0.0) } else { 0.0 };
    let w_random: Vec<f64> = ses.iter().map(|s| 1.0 / (s * s + tau2)).collect();
    let swr: f64 = w_random.iter().sum();
    let re_est: f64 = w_random.iter().zip(effects.iter()).map(|(w, e)| w * e).sum::<f64>() / swr;
    let re_se = (1.0 / swr).sqrt();
    let re_z = re_est / re_se;
    let studies: Vec<Value> = (0..k).map(|i| json!({"study": labels[i], "es": rv(effects[i]), "se": rv(ses[i]), "ci_l": rv(effects[i] - 1.96 * ses[i]), "ci_u": rv(effects[i] + 1.96 * ses[i]), "w_fixed_pct": rv(w_fixed[i] / sw * 100.0), "w_random_pct": rv(w_random[i] / swr * 100.0)})).collect();
    json!({"k": k, "fixed": {"es": rv(fe_est), "se": rv(fe_se), "z": rv(fe_z), "p": rv(2.0 * norm_sf(fe_z.abs())), "ci_l": rv(fe_est - 1.96 * fe_se), "ci_u": rv(fe_est + 1.96 * fe_se)}, "random": {"es": rv(re_est), "se": rv(re_se), "z": rv(re_z), "p": rv(2.0 * norm_sf(re_z.abs())), "ci_l": rv(re_est - 1.96 * re_se), "ci_u": rv(re_est + 1.96 * re_se)}, "het": {"Q": rv(q), "df": k - 1, "p": rv(p_het), "I2_pct": rv(i2), "tau2": rv(tau2)}, "studies": studies})
}

fn meta(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v1.is_empty() {
        return Err("v (study label column) required".into());
    }
    let mut e = if !p.effect_col.is_empty() { p.effect_col.clone() } else { "effect".to_string() };
    if tab.col_idx(&e).is_none() {
        let mut found = None;
        for cand in ["effect", "es", "or", "rr", "hr", "md", "smd", "d"] {
            if tab.col_idx(cand).is_some() {
                found = Some(cand.to_string());
                break;
            }
        }
        e = found.ok_or_else(|| format!("Need effect size column. Found: {:?}", tab.headers))?;
    }
    let mut se_col = if !p.se_col.is_empty() { p.se_col.clone() } else { "se".to_string() };
    if tab.col_idx(&se_col).is_none() {
        let mut found = None;
        for cand in ["se", "var", "ci_l", "ci_u"] {
            if tab.col_idx(cand).is_some() {
                found = Some(cand.to_string());
                break;
            }
        }
        se_col = found.ok_or_else(|| format!("Need SE or CI columns. Found: {:?}", tab.headers))?;
    }
    let labels: Vec<String> = tab.col(&p.v1).unwrap_or_default().iter().take(100).map(|c| c.key()).collect();
    let effects = tab.numeric(&e)?;
    let ses = if se_col == "ci_l" || se_col == "ci_u" {
        let lo = tab.numeric("ci_l").or_else(|_| tab.numeric("ci_low"))?;
        let hi = tab.numeric("ci_u").or_else(|_| tab.numeric("ci_high"))?;
        lo.iter().zip(hi.iter()).map(|(l, h)| (h - l) / (2.0 * 1.96)).collect::<Vec<_>>()
    } else if se_col.to_lowercase().contains("var") {
        tab.numeric(&se_col)?.iter().map(|v| v.max(0.0).sqrt()).collect()
    } else {
        tab.numeric(&se_col)?
    };
    let n = effects.len().min(ses.len()).min(labels.len());
    let mut eff = Vec::new();
    let mut ss = Vec::new();
    let mut lab = Vec::new();
    for i in 0..n {
        if effects[i].is_finite() && ses[i].is_finite() && ses[i] > 0.0 {
            eff.push(effects[i]);
            ss.push(ses[i]);
            lab.push(labels[i].clone());
        }
    }
    if eff.len() < 2 {
        return Err(format!("Need ≥2 valid studies, got {}", eff.len()));
    }
    Ok(meta_from(&eff, &ss, &lab))
}

fn blandaltman(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (method1), v (method2) required".into());
    }
    let m1 = tab.numeric(&p.a)?;
    let m2 = tab.numeric(&p.v1)?;
    let n = m1.len().min(m2.len());
    let diff: Vec<f64> = (0..n).map(|i| m1[i] - m2[i]).collect();
    let avg: Vec<f64> = (0..n).map(|i| (m1[i] + m2[i]) / 2.0).collect();
    let md = mean(&diff);
    let sd = sd1(&diff);
    let lo = md - 1.96 * sd;
    let hi = md + 1.96 * sd;
    let se_loa = sd * (3.0 / n as f64).sqrt();
    let (r, pv) = pearson(&avg, &diff);
    Ok(json!({"n": n, "mean_diff": r4(md), "sd_diff": r4(sd), "loa_lower": r4(lo), "loa_upper": r4(hi), "loa_lower_ci": [r4(lo - 1.96 * se_loa), r4(lo + 1.96 * se_loa)], "loa_upper_ci": [r4(hi - 1.96 * se_loa), r4(hi + 1.96 * se_loa)], "prop_bias_r": r4(r), "prop_bias_p": r4(pv)}))
}

fn kappa(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (rater1), v (rater2) required".into());
    }
    let r1 = tab.col(&p.a).unwrap();
    let r2 = tab.col(&p.v1).unwrap();
    let n = r1.len().min(r2.len());
    let mut pairs: Vec<(Cell, Cell)> = Vec::new();
    for i in 0..n {
        if r1[i].is_valid() && r2[i].is_valid() {
            pairs.push((r1[i].clone(), r2[i].clone()));
        }
    }
    if pairs.is_empty() {
        return Err("No complete pairs".into());
    }
    let mut cats: Vec<Cell> = Vec::new();
    for (x, y) in &pairs {
        for c in [x, y] {
            if !cats.contains(c) {
                cats.push(c.clone());
            }
        }
    }
    cats.sort_by(|a, b| match (a, b) {
        (Cell::Num(x), Cell::Num(y)) => x.partial_cmp(y).unwrap(),
        (Cell::Txt(x), Cell::Txt(y)) => x.cmp(y),
        (Cell::Num(_), Cell::Txt(_)) => std::cmp::Ordering::Less,
        _ => std::cmp::Ordering::Equal,
    });
    if cats.len() < 2 {
        return Err(format!("Need ≥2 categories, got {:?}", cats.iter().map(|c| c.key()).collect::<Vec<_>>()));
    }
    let k = cats.len();
    let mut obs = vec![vec![0.0; k]; k];
    for (x, y) in &pairs {
        obs[cats.iter().position(|c| c == x).unwrap()][cats.iter().position(|c| c == y).unwrap()] += 1.0;
    }
    let total: f64 = obs.iter().flatten().sum();
    let p_o: f64 = (0..k).map(|i| obs[i][i]).sum::<f64>() / total;
    let row_sums: Vec<f64> = obs.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<f64> = (0..k).map(|j| obs.iter().map(|r| r[j]).sum()).collect();
    let p_e: f64 = row_sums.iter().zip(col_sums.iter()).map(|(a, b)| a * b).sum::<f64>() / (total * total);
    let kv = if p_e < 1.0 { (p_o - p_e) / (1.0 - p_e) } else { 0.0 };
    let se = if total > 0.0 && p_e < 1.0 { (p_o * (1.0 - p_o) / (total * (1.0 - p_e).powi(2))).sqrt() } else { 0.0 };
    let z = if se > 0.0 { kv / se } else { 0.0 };
    // Python passes w verbatim ("" when unset) and computes a weighted
    // kappa whenever w != "unweighted" and k > 2 (quadratic unless linear).
    let w = p.w.clone();
    let mut res = json!({"kappa": r4(kv), "se": r4(se), "z": r4(z), "p": r4(2.0 * norm_sf(z.abs())), "p_o": r4(p_o), "p_e": r4(p_e), "n": total as i64, "weight": w.clone(), "agree_pct": r4(p_o * 100.0)});
    if w != "unweighted" && k > 2 {
        let mut w_obs = 0.0;
        let mut w_exp = 0.0;
        for i in 0..k {
            for j in 0..k {
                let wt = if w == "linear" { 1.0 - (i as f64 - j as f64).abs() / (k - 1) as f64 } else { 1.0 - ((i as f64 - j as f64) / (k - 1) as f64).powi(2) };
                w_obs += wt * obs[i][j] / total;
                w_exp += wt * row_sums[i] * col_sums[j] / (total * total);
            }
        }
        let wk = if w_exp < 1.0 { (w_obs - w_exp) / (1.0 - w_exp) } else { 0.0 };
        res.as_object_mut().unwrap().insert("weighted_kappa".into(), r4(wk));
    }
    Ok(res)
}

fn evalue(tab: &Table, p: &P) -> Result<Value, String> {
    let ev = |v: f64, key: &str| -> Value {
        if v <= 1.0 {
            let inv = 1.0 / v;
            let e = if inv > 1.0 { inv + (inv * (inv - 1.0)).sqrt() } else { 1.0 };
            json!({key: v, "E_value": r4(e), "E_lower": r4(0.0), "note": format!("{}<=1: E-value computed on reciprocal (protective effect)", key)})
        } else {
            json!({key: v, "E_value": r4(v + (v * (v - 1.0)).sqrt()), "E_lower": r4(1.0 + (1.0 - 1.0 / v).sqrt())})
        }
    };
    if let Some(ov) = p.or_val {
        if ov <= 0.0 {
            return Err("or must be > 0".into());
        }
        return Ok(ev(ov, "OR"));
    }
    if let Some(rv) = p.rr_val {
        if rv <= 0.0 {
            return Err("rr must be > 0".into());
        }
        if rv <= 1.0 {
            let inv = 1.0 / rv;
            let e = if inv > 1.0 { inv + (inv * (inv - 1.0)).sqrt() } else { 1.0 };
            return Ok(json!({"RR": rv, "E_value": r4(e), "note": "RR<=1: E-value computed on reciprocal (protective effect)"}));
        }
        return Ok(json!({"RR": rv, "E_value": r4(rv + (rv * (rv - 1.0)).sqrt())}));
    }
    if p.a.is_empty() || p.v1.is_empty() || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (exposure), v (outcome) for 2x2 data, or pass or=/rr= directly".into());
    }
    let exp = tab.numeric(&p.a)?;
    let out = tab.numeric(&p.v1)?;
    let n = exp.len().min(out.len());
    let (mut a, mut b, mut c, mut d) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n {
        match (exp[i] == 1.0, out[i] == 1.0) {
            (true, true) => a += 1,
            (true, false) => b += 1,
            (false, true) => c += 1,
            (false, false) => d += 1,
        }
    }
    if b == 0 || c == 0 {
        return Err("Zero cell — cannot compute OR".into());
    }
    let orv = a as f64 * d as f64 / (b as f64 * c as f64);
    let e = if orv > 1.0 { orv + (orv * (orv - 1.0)).sqrt() } else { 0.0 };
    Ok(json!({"a": a, "b": b, "c": c, "d": d, "OR": r4(orv), "E_value": r4(e), "E_lower": r4(if orv > 1.0 { 1.0 + (1.0 - 1.0 / orv).sqrt() } else { 0.0 })}))
}

fn power_pspp(p: &P) -> Result<Value, String> {
    let calc = if p.calc.is_empty() { "n".to_string() } else { p.calc.clone() };
    let test = if p.power_test.is_empty() { "ttest".to_string() } else { p.power_test.clone() };
    let alpha = p.alpha;
    let pw = if p.pw > 0.0 { p.pw } else { 0.80 };
    match test.as_str() {
        "ttest" => {
            if calc == "n" {
                let za = norm_ppf(1.0 - alpha / 2.0);
                let zb = norm_ppf(pw);
                let n1 = (1.0 + 1.0 / p.ratio) * ((za + zb) / p.es).powi(2);
                let n2 = n1 * p.ratio;
                Ok(json!({"n1": n1.ceil() as i64, "n2": n2.ceil() as i64, "n_total": (n1 + n2).ceil() as i64, "d": p.es, "alpha": alpha, "power": pw}))
            } else {
                let n1 = p.n.max(2);
                let n2 = n1 as f64 * p.ratio;
                let ncp = p.es * (n1 as f64 * n2 / (n1 as f64 + n2)).sqrt();
                let df = n1 as f64 + n2 - 2.0;
                let tc = t_ppf(1.0 - alpha / 2.0, df);
                let power = 1.0 - nct_cdf(tc, df, ncp) + nct_cdf(-tc, df, ncp);
                Ok(json!({"power": r4(power), "n1": n1, "n2": n2.ceil() as i64, "d": p.es, "alpha": alpha}))
            }
        }
        "prop" => {
            if calc == "n" {
                let za = norm_ppf(1.0 - alpha / 2.0);
                let zb = norm_ppf(pw);
                let pb = (p.p0 + p.p1) / 2.0;
                let n = (za * (2.0 * pb * (1.0 - pb)).sqrt() + zb * (p.p0 * (1.0 - p.p0) + p.p1 * (1.0 - p.p1)).sqrt()).powi(2) / (p.p1 - p.p0).powi(2);
                Ok(json!({"n_per_group": n.ceil() as i64, "n_total": (2.0 * n).ceil() as i64, "p0": p.p0, "p1": p.p1, "alpha": alpha, "power": pw}))
            } else {
                let n = p.n.max(2);
                let za = norm_ppf(1.0 - alpha / 2.0);
                let se = (p.p0 * (1.0 - p.p0) / n as f64 + p.p1 * (1.0 - p.p1) / n as f64).sqrt();
                let zp = ((p.p1 - p.p0).abs() - za * (2.0 * p.p0 * (1.0 - p.p0) / n as f64).sqrt()) / se;
                Ok(json!({"power": r4(norm_cdf(zp)), "n_per_group": n, "p0": p.p0, "p1": p.p1, "alpha": alpha}))
            }
        }
        "anova" => {
            if calc == "n" {
                let za = norm_ppf(1.0 - alpha / 2.0);
                let zb = norm_ppf(pw);
                let g = p.k as f64;
                let npg = if p.es > 0.0 { 2.0 * (za + zb).powi(2) * g / (p.es.powi(2) * (g - 1.0)) } else { 0.0 };
                Ok(json!({"n_per_group": npg.ceil() as i64, "n_total": (npg.ceil() * g) as i64, "groups": p.k, "f": p.es, "alpha": alpha, "power": pw}))
            } else {
                let npg = p.n.max(2);
                let g = p.k;
                let lam = g as f64 * npg as f64 * p.es.powi(2);
                let df1 = (g - 1) as f64;
                let df2 = (g * (npg - 1)) as f64;
                let fc = f_ppf(1.0 - alpha, df1, df2);
                let power = 1.0 - ncf_cdf(fc, df1, df2, lam);
                Ok(json!({"power": r4(power), "n_per_group": npg, "n_total": npg * g, "groups": g, "f": p.es, "alpha": alpha}))
            }
        }
        _ => Err(format!("Unknown test '{}' or calc '{}'", test, calc)),
    }
}

fn rank(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v1.is_empty() || tab.col_idx(&p.v1).is_none() {
        return Err("v (variable) required".into());
    }
    let vals = tab.numeric(&p.v1)?;
    let ranks = rankdata(&vals);
    Ok(json!({"n": vals.len(), "ranks": ranks.iter().map(|r| r4(*r)).collect::<Vec<_>>()}))
}

fn hedgesg(tab: &Table, p: &P) -> Result<Value, String> {
    if p.a.is_empty() || p.v1.is_empty() || p.g.len() < 2 || tab.col_idx(&p.a).is_none() || tab.col_idx(&p.v1).is_none() {
        return Err("a (group var), v (value var), g=[g1,g2] required".into());
    }
    let grps = tab.groups_numeric(&p.v1, &p.a, &p.g[..2])?;
    let (n1, n2) = (grps[0].len(), grps[1].len());
    if n1 < 2 || n2 < 2 {
        return Err("Need ≥2 per group".into());
    }
    let pooled = (((n1 - 1) as f64 * var_ddof1(&grps[0]) + (n2 - 1) as f64 * var_ddof1(&grps[1])) / (n1 + n2 - 2) as f64).sqrt();
    let d = if pooled > 0.0 { (mean(&grps[1]) - mean(&grps[0])) / pooled } else { 0.0 };
    let df = (n1 + n2 - 2) as f64;
    let j = if df > 2.0 { 1.0 - 3.0 / (4.0 * df - 1.0) } else { 1.0 };
    let g = d * j;
    let se_g = ((n1 + n2) as f64 / (n1 as f64 * n2 as f64) + g * g / (2.0 * (n1 + n2) as f64)).sqrt();
    let fmt_grp = |c: &Cell| -> String {
        match c {
            Cell::Num(x) if x.fract() == 0.0 => format!("{}", *x as i64),
            other => other.key(),
        }
    };
    Ok(json!({"d": r4(d), "g": r4(g), "se": r4(se_g), "n1": n1, "n2": n2, "direction": format!("{} minus {} (positive = {} higher)", fmt_grp(&p.g[1]), fmt_grp(&p.g[0]), fmt_grp(&p.g[1]))}))
}

fn omega(tab: &Table, p: &P) -> Result<Value, String> {
    if p.v.len() < 2 {
        return Err("v (2+ item variables) required".into());
    }
    for var in &p.v {
        if tab.col_idx(var).is_none() {
            return Err(format!("Variable '{}' not found", var));
        }
    }
    let arrays: Vec<Vec<f64>> = p.v.iter().map(|v| tab.numeric(v).unwrap_or_default()).collect();
    let n = arrays.iter().map(|a| a.len()).min().unwrap_or(0);
    let m = p.v.len();
    let mut corr = DMatrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let (r, _) = pearson(&arrays[i][..n], &arrays[j][..n]);
            corr[(i, j)] = if i == j { 1.0 } else { r };
        }
    }
    let eig = nalgebra::SymmetricEigen::new(corr);
    let mut idx = 0usize;
    for i in 1..m {
        if eig.eigenvalues[i] > eig.eigenvalues[idx] {
            idx = i;
        }
    }
    let loadings: Vec<f64> = (0..m).map(|r| (eig.eigenvectors[(r, idx)] * eig.eigenvalues[idx].max(0.0).sqrt()).abs()).collect();
    let sum_load: f64 = loadings.iter().sum();
    let unique: f64 = loadings.iter().map(|l| (1.0 - l * l).max(0.0)).sum();
    let om = sum_load * sum_load / (sum_load * sum_load + unique);
    let item_vars: f64 = arrays.iter().map(|a| var_ddof1(&a[..n])).sum();
    let totals: Vec<f64> = (0..n).map(|i| arrays.iter().map(|a| a[i]).sum()).collect();
    let total_var = var_ddof1(&totals);
    let alpha = if total_var > 0.0 { (m as f64 / (m as f64 - 1.0)) * (1.0 - item_vars / total_var) } else { 0.0 };
    let mut load_map = Map::new();
    for (i, var) in p.v.iter().enumerate() {
        load_map.insert(var.clone(), r4(loadings[i]));
    }
    Ok(json!({"omega": r4(om), "alpha": r4(alpha), "loadings": Value::Object(load_map), "note": "single-factor approximation; assumes a unidimensional scale (verify with 'factor')."}))
}
