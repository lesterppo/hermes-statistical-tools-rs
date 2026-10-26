// hst — Rust port of hermes-statistical-tools.
// Usage:
//   hst pspp ttest --csv-file data.csv --a g --v v --g 1,2
//   hst pspp desc --csv "x,y\n1,2\n..." --v x,y
//   hst medical_ext km --csv-file surv.csv --params '{"t":"time","e":"event","g":"grp"}'
//   hst irt score --csv-file items.csv
//   hst pspp --list   (lists implemented actions; unlisted = Not yet)
// Output is one compact JSON object on stdout. Errors: {"e": "..."}.
use clap::Parser;
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
    /// Extra parameters as JSON (medical_ext / irt style: t, e, g, p, d_es, f, n, pw, a, ratio, alt, k, p1, p2)
    #[arg(long)]
    params: Option<String>,
}

const IMPLEMENTED_PSPP: &[&str] = &[
    "desc", "freq", "examine", "means", "crosstab", "ttest", "pttest", "ttest1", "anova", "mw",
    "kw", "wilcoxon", "friedman", "sign", "ks", "runs", "corr", "spearman", "partial", "reg",
    "logistic", "poisson", "factor", "reliability", "roc", "eval", "surv", "meta", "blandaltman",
    "kappa", "evalue", "power", "rank", "hedgesg", "omega",
];
const NOT_YET_PSPP: &[&str] = &[
    "dca", "quantreg", "negbin", "mediation", "pscore", "gam", "zip", "lasso", "adf", "beta",
];

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
            "irt_implemented": ["score", "ctt"],
            "irt_not_yet": ["rasch", "2pl", "3pl", "grm", "pcm"],
            "statsmodels": "not_yet",
            "sem": "not_yet",
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
        "statsmodels" => err("statsmodels actions (mlm, gee, gee_ord, gee_nom, anova_rm, mice) are Not yet in the Rust port: they require statsmodels internals"),
        "sem" => err("sem actions (sem_fit, sem_inspect) are Not yet in the Rust port: they require semopy"),
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
        "score" | "irt_score" => match actions_med::irt_score(&tab, params) {
            Ok(v) => v,
            Err(e) => err(&e),
        },
        "ctt" | "irt_ctt" => match actions_med::irt_ctt(&tab) {
            Ok(v) => v,
            Err(e) => err(&e),
        },
        "rasch" | "irt_rasch" | "2pl" | "irt_2pl" | "3pl" | "irt_3pl" | "grm" | "irt_grm" | "pcm" | "irt_pcm" => {
            err("irt model actions (rasch, 2pl, 3pl, grm, pcm) are Not yet in the Rust port: they require girth marginal-maximum-likelihood internals")
        }
        _ => err(&format!("unknown irt action '{}'", action)),
    }
}
