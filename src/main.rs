// hst — Rust port of hermes-statistical-tools.
// Usage:
//   hst pspp ttest --csv-file data.csv --a g --v v --g 1,2
//   hst pspp desc --csv "x,y\n1,2\n..." --v x,y
//   hst medical_ext km --csv-file surv.csv --params '{"t":"time","e":"event","g":"grp"}'
//   hst irt score --csv-file items.csv
//   hst pspp --list   (lists implemented actions; unlisted = Not yet)
// Output is one compact JSON object on stdout. Errors: {"e": "..."}.
use clap::Parser;
use hermes_statistical_tools_rs::actions_irt_new;
use hermes_statistical_tools_rs::actions_med;
use hermes_statistical_tools_rs::actions_pspp::{self, P};
use hermes_statistical_tools_rs::{err, parse_cell_list, parse_csv, Cell, Table};
use serde_json::{json, Value};
use std::io::Read;

#[derive(Parser, Debug)]
#[command(name = "hst", version, about = "Agent-native statistical tools (Rust port of hermes-statistical-tools)")]
struct Cli {
    /// Tool: pspp | medical_ext | irt | statsmodels | sem
    tool: Option<String>,
    /// Action / test type (pspp: desc, ttest, ...; medical_ext: km, cox, ...)
    action: Option<String>,
    #[arg(long)]
    list: bool,
    /// CSV data as a string (use --csv-file for a file, or - for stdin)
    #[arg(long)]
    csv: Option<String>,
    #[arg(long)]
    csv_file: Option<String>,
    /// Grouping / row variable
    #[arg(long)]
    a: Option<String>,
    /// Value variable(s), comma-separated
    #[arg(long)]
    v: Option<String>,
    /// Group labels, comma-separated (numbers parsed as numbers)
    #[arg(long)]
    g: Option<String>,
    /// Outcome / dependent variable
    #[arg(long)]
    o: Option<String>,
    /// Weight variable / kappa weight type
    #[arg(long)]
    w: Option<String>,
    /// Predictor variables, comma-separated
    #[arg(long)]
    p: Option<String>,
    /// Control variables, comma-separated
    #[arg(long)]
    c: Option<String>,
    /// Population mean (one-sample tests)
    #[arg(long)]
    mu: Option<f64>,
    /// Paired flag
    #[arg(long)]
    paired: bool,
    /// Post-hoc method: tukey | bonferroni | gh
    #[arg(long)]
    posthoc: Option<String>,
    /// Rotation: varimax
    #[arg(long)]
    rotate: Option<String>,
    /// Components (factor) / per-group n (power)
    #[arg(long)]
    n: Option<i64>,
    /// Positive class label
    #[arg(long)]
    pos: Option<f64>,
    /// Positive reference class
    #[arg(long)]
    pos_ref: Option<f64>,
    /// Effect-size column (meta)
    #[arg(long)]
    effect_col: Option<String>,
    /// SE column (meta)
    #[arg(long)]
    se_col: Option<String>,
    /// Power test: ttest | prop | anova
    #[arg(long)]
    power_test: Option<String>,
    /// Alpha
    #[arg(long)]
    alpha: Option<f64>,
    /// Effect size
    #[arg(long)]
    es: Option<f64>,
    /// n2/n1 ratio
    #[arg(long)]
    ratio: Option<f64>,
    /// Number of groups
    #[arg(long)]
    k: Option<i64>,
    /// Baseline proportion
    #[arg(long)]
    p0: Option<f64>,
    /// Alternative proportion
    #[arg(long)]
    p1: Option<f64>,
    /// calc: n | power
    #[arg(long)]
    calc: Option<String>,
    /// Direct OR (evalue)
    #[arg(long)]
    or_val: Option<f64>,
    /// Direct RR (evalue)
    #[arg(long)]
    rr_val: Option<f64>,
    /// Time variable (pspp surv)
    #[arg(long)]
    tvar: Option<String>,
    /// Power target (default 0.80)
    #[arg(long)]
    pw: Option<f64>,
    /// Quantile level (quantreg, default 0.5)
    #[arg(long)]
    tau: Option<f64>,
    /// Penalty (lasso, default 0.1)
    #[arg(long)]
    lam: Option<f64>,
    /// Knots per term (gam, default 5)
    #[arg(long)]
    df: Option<i64>,
    /// Lags (adf, default 1)
    #[arg(long)]
    lags: Option<i64>,
    /// Extra parameters as JSON (medical_ext / irt style: t, e, g, p, d_es, f, n, pw, a, ratio, alt, k, p1, p2)
    #[arg(long)]
    params: Option<String>,
    /// GEE family (statsmodels): gaussian | binomial | poisson | gamma
    #[arg(long)]
    fam: Option<String>,
    /// GEE covariance structure: independence | exchangeable | ar1 | unstructured
    #[arg(long)]
    cov: Option<String>,
    /// MICE model: ols | logit | poisson
    #[arg(long)]
    model_type: Option<String>,
    /// SEM model description (lavaan-like syntax)
    #[arg(long)]
    desc: Option<String>,
    /// SEM inspect mode: raw | std_est
    #[arg(long)]
    mode: Option<String>,
}

const IMPLEMENTED_PSPP: &[&str] = &[
    "desc", "freq", "examine", "means", "crosstab", "ttest", "pttest", "ttest1", "anova", "mw",
    "kw", "wilcoxon", "friedman", "sign", "ks", "runs", "corr", "spearman", "partial", "reg",
    "logistic", "poisson", "factor", "reliability", "roc", "eval", "surv", "meta", "blandaltman",
    "kappa", "evalue", "power", "rank", "hedgesg", "omega",
    "dca", "quantreg", "negbin", "mediation", "pscore", "gam", "zip", "lasso", "adf", "beta",
];
const NOT_YET_PSPP: &[&str] = &[];

fn split(s: &Option<String>) -> Vec<String> {
    s.as_deref()
        .unwrap_or("")
        .split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

fn main() {
    let cli = Cli::parse();
    if cli.list {
        println!("{}", serde_json::to_string(&json!({
            "pspp_implemented": IMPLEMENTED_PSPP,
            "pspp_not_yet": NOT_YET_PSPP,
            "medical_ext_implemented": ["km", "cox", "logrank", "forest", "ttest", "anova", "prop"],
            "irt_implemented": ["rasch", "2pl", "3pl", "grm", "pcm", "score", "ctt"],
            "irt_not_yet": [],
            "statsmodels_implemented": ["mlm", "gee", "gee_ord", "gee_nom", "anova_rm", "mice"],
            "statsmodels_not_yet": [],
            "sem_implemented": ["sem_fit", "sem_inspect"],
            "sem_not_yet": [],
        })).unwrap());
        return;
    }
    let tool = match &cli.tool {
        Some(t) => t.clone(),
        None => {
            println!("{}", err("tool required: pspp | medical_ext | irt | statsmodels | sem"));
            std::process::exit(1);
        }
    };
    let action = match &cli.action {
        Some(a) => a.clone(),
        None => {
            println!("{}", err("action required"));
            std::process::exit(1);
        }
    };
    // CSV source
    let csv_text: Result<String, String> = (|| {
        if let Some(f) = &cli.csv_file {
            return std::fs::read_to_string(f).map_err(|e| format!("cannot read csv file: {}", e));
        }
        if let Some(s) = &cli.csv {
            if s == "-" {
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf).map_err(|e| e.to_string())?;
                return Ok(buf);
            }
            return Ok(s.clone());
        }
        Ok(String::new())
    })();
    let csv_text = match csv_text {
        Ok(s) => s,
        Err(e) => {
            println!("{}", err(&e));
            std::process::exit(1);
        }
    };
    let params: serde_json::Map<String, Value> = cli
        .params
        .as_deref()
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    let out: Value = match tool.as_str() {
        "pspp" => run_pspp(&action, &csv_text, &cli, &params),
        "medical_ext" => run_medical(&action, &csv_text, &params),
        "irt" => run_irt(&action, &csv_text, &params),
        "statsmodels" => run_statsmodels(&action, &csv_text, &cli, &params),
        "sem" => run_sem(&action, &csv_text, &cli, &params),
        _ => err(&format!("unknown tool '{}'", tool)),
    };
    let is_err = out.get("e").is_some();
    println!("{}", serde_json::to_string(&out).unwrap());
    if is_err {
        std::process::exit(1);
    }
}

fn run_pspp(action: &str, csv_text: &str, cli: &Cli, params: &serde_json::Map<String, Value>) -> Value {
    if csv_text.trim().is_empty() {
        return err("d (CSV data) required");
    }
    let tab = match parse_csv(csv_text) {
        Ok(t) => t,
        Err(e) => return err(&e),
    };
    let vlist = split(&cli.v);
    let plist = split(&cli.p);
    let clist = split(&cli.c);
    let mut p = P {
        a: cli.a.clone().unwrap_or_default(),
        v: vlist.clone(),
        v1: vlist.first().cloned().unwrap_or_default(),
        g: cli.g.as_deref().map(parse_cell_list).unwrap_or_default(),
        o: cli.o.clone().unwrap_or_default(),
        w: cli.w.clone().unwrap_or_default(),
        p: plist,
        c: clist,
        mu: cli.mu,
        paired: cli.paired,
        posthoc: cli.posthoc.clone().unwrap_or_default(),
        rotate: cli.rotate.clone().unwrap_or_default(),
        n: cli.n.unwrap_or(0),
        pos: cli.pos.unwrap_or(1.0),
        pos_ref: cli.pos_ref.unwrap_or(1.0),
        effect_col: cli.effect_col.clone().unwrap_or_default(),
        se_col: cli.se_col.clone().unwrap_or_default(),
        power_test: cli.power_test.clone().unwrap_or_else(|| "ttest".to_string()),
        alpha: cli.alpha.unwrap_or(0.05),
        es: cli.es.unwrap_or(0.5),
        ratio: cli.ratio.unwrap_or(1.0),
        k: cli.k.unwrap_or(2),
        p0: cli.p0.unwrap_or(0.5),
        p1: cli.p1.unwrap_or(0.7),
        calc: cli.calc.clone().unwrap_or_else(|| "n".to_string()),
        or_val: cli.or_val,
        rr_val: cli.rr_val,
        tvar: cli.tvar.clone().unwrap_or_default(),
        pw: cli.pw.unwrap_or(0.80),
        tau: cli.tau.unwrap_or(0.5),
        lam: cli.lam.unwrap_or(0.1),
        df_knots: cli.df.unwrap_or(5),
        lags: cli.lags.unwrap_or(1),
    };
    // --params JSON overrides (e.g. {"or": 2.5}, {"mu": 0})
    if let Some(v) = params.get("or").and_then(|x| x.as_f64()) {
        p.or_val = Some(v);
    }
    if let Some(v) = params.get("rr").and_then(|x| x.as_f64()) {
        p.rr_val = Some(v);
    }
    if let Some(v) = params.get("mu").and_then(|x| x.as_f64()) {
        p.mu = Some(v);
    }
    if let Some(v) = params.get("power").and_then(|x| x.as_f64()) {
        p.pw = v;
    }
    if let Some(v) = params.get("tau").and_then(|x| x.as_f64()) {
        p.tau = v;
    }
    if let Some(v) = params.get("lam").and_then(|x| x.as_f64()) {
        p.lam = v;
    }
    if let Some(v) = params.get("df").and_then(|x| x.as_i64()) {
        p.df_knots = v;
    }
    if let Some(v) = params.get("lags").and_then(|x| x.as_i64()) {
        p.lags = v;
    }
    match actions_pspp::dispatch(action, &tab, &p) {
        Ok(inner) => json!({"k": inner}),
        Err(e) => err(&e),
    }
}

fn run_medical(action: &str, csv_text: &str, params: &serde_json::Map<String, Value>) -> Value {
    // Power actions take structured params, no CSV needed.
    if matches!(action, "ttest" | "anova" | "prop") {
        return match actions_med::dispatch_med(action, &dummy_table(), params) {
            Ok(v) => v,
            Err(e) => err(&e),
        };
    }
    if csv_text.trim().is_empty() {
        return err("d (CSV data) required");
    }
    let tab = match parse_csv(csv_text) {
        Ok(t) => t,
        Err(e) => return err(&e),
    };
    match actions_med::dispatch_med(action, &tab, params) {
        Ok(v) => v,
        Err(e) => err(&e),
    }
}
fn dummy_table() -> Table {
    Table { headers: vec![], rows: vec![] }
}

fn run_irt(action: &str, csv_text: &str, params: &serde_json::Map<String, Value>) -> Value {
    if csv_text.trim().is_empty() {
        return err("d (CSV data) required");
    }
    let tab = match parse_csv(csv_text) {
        Ok(t) => t,
        Err(e) => return err(&e),
    };
    match action {
        "score" | "irt_score" => match actions_irt_new::irt_score_new(&tab, params) {
            Ok(v) => v,
            Err(e) => err(&e),
        },
        "ctt" | "irt_ctt" => match actions_med::irt_ctt(&tab) {
            Ok(v) => v,
            Err(e) => err(&e),
        },
        "rasch" | "irt_rasch" => match actions_irt_new::irt_rasch(&tab, params) {
            Ok(v) => v,
            Err(e) => err(&e),
        },
        "2pl" | "irt_2pl" => match actions_irt_new::irt_2pl(&tab) {
            Ok(v) => v,
            Err(e) => err(&e),
        },
        "3pl" | "irt_3pl" => match actions_irt_new::irt_3pl(&tab) {
            Ok(v) => v,
            Err(e) => err(&e),
        },
        "grm" | "irt_grm" => match actions_irt_new::irt_grm(&tab) {
            Ok(v) => v,
            Err(e) => err(&e),
        },
        "pcm" | "irt_pcm" => match actions_irt_new::irt_pcm(&tab) {
            Ok(v) => v,
            Err(e) => err(&e),
        },
        _ => err(&format!("unknown irt action '{}'", action)),
    }
}

fn params_str(params: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    params.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
}

fn params_str_list(params: &serde_json::Map<String, Value>, key: &str) -> Vec<String> {
    params
        .get(key)
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

fn run_statsmodels(
    action: &str,
    csv_text: &str,
    cli: &Cli,
    params: &serde_json::Map<String, Value>,
) -> Value {
    if csv_text.trim().is_empty() {
        return err("d (CSV data) required");
    }
    let tab = match parse_csv(csv_text) {
        Ok(t) => t,
        Err(e) => return err(&e),
    };
    // Within factors: params {"w": [...]} wins, else comma-split --w.
    let wlist = {
        let pl = params_str_list(params, "w");
        if !pl.is_empty() {
            pl
        } else {
            split(&cli.w)
        }
    };
    let a = hermes_statistical_tools_rs::actions_sm_new::SmArgs {
        o: cli.o.clone().unwrap_or_default(),
        p: split(&cli.p),
        g: cli.a.clone().unwrap_or_default(),
        fam: cli.fam.clone().or_else(|| params_str(params, "fam")).unwrap_or_else(|| "gaussian".to_string()),
        cov: cli.cov.clone().or_else(|| params_str(params, "cov")).unwrap_or_else(|| "exchangeable".to_string()),
        w: wlist,
        n_imp: cli.n.unwrap_or_else(|| params.get("n").and_then(|v| v.as_i64()).unwrap_or(5)),
        model: cli.model_type.clone().or_else(|| params_str(params, "mod")).unwrap_or_else(|| "ols".to_string()),
    };
    match hermes_statistical_tools_rs::actions_sm_new::dispatch_sm(action, &tab, &a) {
        Ok(v) => v,
        Err(e) => err(&e),
    }
}

fn run_sem(
    action: &str,
    csv_text: &str,
    cli: &Cli,
    params: &serde_json::Map<String, Value>,
) -> Value {
    let desc = cli.desc.clone().or_else(|| params_str(params, "desc")).unwrap_or_default();
    let mode = cli.mode.clone().or_else(|| params_str(params, "mode")).unwrap_or_else(|| "raw".to_string());
    // sem_inspect without data is a cache miss in the Python tool; the
    // Rust port reports the same error via an empty table.
    let tab = if csv_text.trim().is_empty() {
        Table { headers: vec![], rows: vec![] }
    } else {
        match parse_csv(csv_text) {
            Ok(t) => t,
            Err(e) => return err(&e),
        }
    };
    match hermes_statistical_tools_rs::actions_sem_new::dispatch_sem(action, &tab, &desc, &mode) {
        Ok(v) => v,
        Err(e) => err(&e),
    }
}
