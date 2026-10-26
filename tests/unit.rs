// Unit tests with KNOWN answers (textbook / hand-computed), not values
// copied from the implementation under test.
use hermes_statistical_tools_rs::actions_pspp::{dispatch, P};
use hermes_statistical_tools_rs::{
    binom_cdf, chi2_sf, f_sf, mean, ncf_cdf, nct_cdf, norm_sf, ols, parse_csv, pearson,
    studentized_range_sf, t_sf, Cell,
};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn mean_sd() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    assert!(approx(mean(&v), 2.5, 1e-12));
    // sample sd of 1..4 = sqrt(5/3) = 1.290994
    assert!(approx(hermes_statistical_tools_rs::sd1(&v), 1.290994, 1e-5));
}

#[test]
fn pearson_perfect_and_known() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let (r, p) = pearson(&x, &y);
    assert!(approx(r, 1.0, 1e-12));
    assert!(approx(p, 0.0, 1e-12));
    // t(0.975, df=4) = 2.7764 (standard table)
    assert!(approx(2.0 * t_sf(2.7764, 4.0), 0.05, 1e-3));
    // chi2(3.8415, df=1) sf = 0.05
    assert!(approx(chi2_sf(3.8415, 1.0), 0.05, 1e-3));
    // F(4.0, df1=2, df2=12) sf ≈ 0.0466 (standard table / scipy)
    assert!(approx(f_sf(4.0, 2.0, 12.0), 0.0466, 1e-3));
    // normal two-sided: z=1.96 -> p=0.05
    assert!(approx(2.0 * norm_sf(1.96), 0.05, 1e-3));
}

#[test]
fn ols_exact_line() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 8.0, 11.0, 14.0, 17.0]; // y = 2 + 3x
    let o = ols(&y, &[x], true).unwrap();
    assert!(approx(o.beta[0], 2.0, 1e-9));
    assert!(approx(o.beta[1], 3.0, 1e-9));
    assert!(approx(o.r2, 1.0, 1e-9));
}

#[test]
fn studentized_range_critical_value() {
    // Tukey table: k=3, df=20, alpha=0.05 critical q = 3.58
    let sf = studentized_range_sf(3.58, 3, 20.0);
    assert!(approx(sf, 0.05, 0.01), "sf={}", sf);
    // k=2 studentized range = sqrt(2)*|t|; sf at q=2.776*sqrt(2)?? use
    // q=4.0, k=2, df=10: sf should equal 2*t_sf(4/sqrt2, 10) = 2*t_sf(2.828,10)
    let sf2 = studentized_range_sf(4.0, 2, 10.0);
    let expected = 2.0 * t_sf(4.0 / 2.0f64.sqrt(), 10.0);
    assert!(approx(sf2, expected, 0.01), "sf2={} expected={}", sf2, expected);
}

#[test]
fn ttest_hand_computed() {
    let tab = parse_csv("g,v\n1,10\n1,12\n1,11\n2,20\n2,22\n2,21").unwrap();
    let mut p = P::default();
    p.a = "g".into();
    p.v1 = "v".into();
    p.g = vec![Cell::Num(1.0), Cell::Num(2.0)];
    let out = dispatch("ttest", &tab, &p).unwrap();
    // diff = -10, se = sqrt(1/3+1/3) = 0.8165, t = -12.2474, df = 4
    assert!(approx(out["t"].as_f64().unwrap(), -12.2474, 1e-3));
    assert!(approx(out["df"].as_f64().unwrap(), 4.0, 1e-3));
    assert!(approx(out["p"].as_f64().unwrap(), 0.0003, 1e-3));
    assert!(approx(out["d"].as_f64().unwrap(), 10.0, 1e-3));
}

#[test]
fn crosstab_chi2_hand_computed() {
    // [[10,5],[5,10]]: expected all 7.5, chi2 = 4*6.25/7.5 = 3.3333, p(df=1)=0.0679
    let tab = parse_csv("a,v\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,1\n0,1\n0,1\n0,1\n0,1\n1,0\n1,0\n1,0\n1,0\n1,0\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1").unwrap();
    let mut p = P::default();
    p.a = "a".into();
    p.v1 = "v".into();
    let out = dispatch("crosstab", &tab, &p).unwrap();
    assert!(approx(out["χ²"].as_f64().unwrap(), 3.3333, 1e-3));
    assert!(approx(out["p"].as_f64().unwrap(), 0.0679, 1e-3));
}

#[test]
fn mw_exact_small() {
    // [1,2,3] vs [4,5,6]: U=0, exact two-sided p = 2/20 = 0.1
    let tab = parse_csv("g,v\n1,1\n1,2\n1,3\n2,4\n2,5\n2,6").unwrap();
    let mut p = P::default();
    p.a = "g".into();
    p.v1 = "v".into();
    p.g = vec![Cell::Num(1.0), Cell::Num(2.0)];
    let out = dispatch("mw", &tab, &p).unwrap();
    assert!(approx(out["U"].as_f64().unwrap(), 0.0, 1e-9));
    assert!(approx(out["p"].as_f64().unwrap(), 0.1, 1e-9));
}

#[test]
fn wilcoxon_all_positive() {
    // diffs 1..5 all positive: W = 15, exact two-sided p = 2/32 = 0.0625
    let tab = parse_csv("a,v\n2,1\n4,2\n6,3\n8,4\n10,5").unwrap();
    let mut p = P::default();
    p.a = "a".into();
    p.v1 = "v".into();
    let out = dispatch("wilcoxon", &tab, &p).unwrap();
    assert!(approx(out["W"].as_f64().unwrap(), 15.0, 1e-9));
    assert!(approx(out["p"].as_f64().unwrap(), 0.0625, 1e-9));
}

#[test]
fn sign_test_hand() {
    // 8 pos, 2 neg: p = 2 * BinomCDF(2; 10, 0.5) = 2*0.0547 = 0.1094
    assert!(approx(2.0 * binom_cdf(2, 10, 0.5), 0.1094, 1e-3));
    let tab = parse_csv("a,v\n2,1\n2,1\n2,1\n2,1\n2,1\n2,1\n2,1\n2,1\n1,2\n1,2").unwrap();
    let mut p = P::default();
    p.a = "a".into();
    p.v1 = "v".into();
    let out = dispatch("sign", &tab, &p).unwrap();
    assert_eq!(out["pos"].as_i64().unwrap(), 8);
    assert_eq!(out["neg"].as_i64().unwrap(), 2);
    assert!(approx(out["p"].as_f64().unwrap(), 0.1094, 1e-3));
}

#[test]
fn power_ttest_n_hand() {
    // PSPP normal formula: n1 = 2*((1.96+0.8416)/0.5)^2 = 62.78 -> 63
    let tab = parse_csv("x\n1").unwrap();
    let mut p = P::default();
    p.power_test = "ttest".into();
    p.calc = "n".into();
    p.alpha = 0.05;
    p.es = 0.5;
    p.ratio = 1.0;
    p.pw = 0.80;
    let out = dispatch("power", &tab, &p).unwrap();
    assert_eq!(out["n1"].as_i64().unwrap(), 63);
    assert_eq!(out["n2"].as_i64().unwrap(), 63);
}

#[test]
fn nct_ncf_sanity() {
    // nct with ncp=0 is central t: CDF(0)=0.5
    assert!(approx(nct_cdf(0.0, 10.0, 0.0), 0.5, 1e-3));
    // ncf with lam=0 is central F: CDF(2; 2,10) = 1 - f_sf(2,2,10)
    let central = 1.0 - f_sf(2.0, 2.0, 10.0);
    assert!(approx(ncf_cdf(2.0, 2.0, 10.0, 0.0), central, 1e-3));
}

#[test]
fn errors_are_json_not_panics() {
    let tab = parse_csv("x\n1\n2").unwrap();
    let p = P::default();
    assert!(dispatch("ttest", &tab, &p).is_err());
    assert!(dispatch("nosuchtest", &tab, &p).is_err());
    assert!(parse_csv("").is_err());
}
