#!/usr/bin/env python3
"""Live deep test: Rust hst vs Python original, same deterministic CSVs.
Run from the repo root with the Python scratch venv active, or set PYBIN.
Usage: python3 scripts/deep_test.py [--only SUBSTR]
Compares parsed JSON. Numeric pass: |a-b| <= 5e-4 OR rel <= 5e-3.
String keys/values must match exactly, except keys listed in SKIP.
Writes deep_test_report.json next to this script and prints a summary.
"""
import importlib.util
import json
import math
import random
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path.home() / "workspace" / "hst-py-orig"
PYBIN = str(SCRATCH / ".venv" / "bin" / "python")
RUST = str(REPO / "target" / "release" / "hst")

# Stub the Hermes registry so tool modules import standalone.
class _Reg:
    def register(self, **kw):
        pass

_tools = types.ModuleType("tools")
_tools.__path__ = []
_tools.registry = _Reg()
sys.modules["tools"] = _tools
sys.modules["tools.registry"] = types.SimpleNamespace(registry=_Reg())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOOLS = SCRATCH / "tools"
pspp = load("pspp_tool", TOOLS / "pspp_tool.py")
med = load("medical_ext_tool", TOOLS / "medical_ext_tool.py")
irt = load("irt_tool", TOOLS / "irt_tool.py")

rng = random.Random(20261026)


def csv(header, rows):
    def _row(r):
        if isinstance(r, (list, tuple)):
            return r
        return (r,)
    return header + "\n" + "\n".join(",".join(str(x) for x in _row(r)) for r in rows)


# ---------- deterministic datasets ----------
g2 = csv("g,v", [(1, 10 + rng.gauss(0, 1)) for _ in range(12)] + [(2, 20 + rng.gauss(0, 1)) for _ in range(12)])
# Rebuild with fixed values for reproducibility across runs (rng above is
# deterministic, but keep the exact strings by regenerating identically):
rng = random.Random(42)
g2_rows = [(1, round(10 + rng.gauss(0, 1.2), 2)) for _ in range(12)] + [
    (2, round(20 + rng.gauss(0, 1.2), 2)) for _ in range(12)
]
g2 = csv("g,v", g2_rows)
paired = csv("pre,post", [(round(50 + rng.gauss(0, 5), 2), round(55 + rng.gauss(0, 5), 2)) for _ in range(20)])
one = csv("v", [round(100 + rng.gauss(0, 10), 2) for _ in range(30)])
anova_d = csv("g,v", [(g, round(10 + 10 * (g - 1) + rng.gauss(0, 2), 2)) for g in (1, 2, 3) for _ in range(10)])
ct = csv("a,v", [(0, 0)] * 10 + [(0, 1)] * 5 + [(1, 0)] * 5 + [(1, 1)] * 10)
ct3 = csv("a,v", [(0, 0)] * 8 + [(0, 1)] * 4 + [(0, 2)] * 2 + [(1, 0)] * 3 + [(1, 1)] * 9 + [(1, 2)] * 5 + [(2, 0)] * 6 + [(2, 1)] * 2 + [(2, 2)] * 7)
corr_d = csv("x,y,z", [(i, round(2 * i + rng.gauss(0, 3), 2), round(50 - i + rng.gauss(0, 4), 2)) for i in range(1, 26)])
reg_d = csv("x1,x2,y,bin,cnt", [(round(rng.gauss(0, 1), 2), round(rng.gauss(0, 1), 2), 0, 0, 0) for _ in range(30)])
reg_rows = []
for _ in range(30):
    x1 = round(rng.gauss(0, 1), 2)
    x2 = round(rng.gauss(0, 1), 2)
    y = round(3 + 2 * x1 - 1.5 * x2 + rng.gauss(0, 1), 2)
    b = 1 if (x1 + rng.gauss(0, 0.5)) > 0 else 0
    cnt = max(0, int(round(5 + 2 * x1 + rng.gauss(0, 1))))
    reg_rows.append((x1, x2, y, b, cnt))
reg_d = csv("x1,x2,y,bin,cnt", reg_rows)
items = csv("i1,i2,i3,i4", [(round(rng.gauss(0, 1), 2), round(rng.gauss(0, 1), 2), round(rng.gauss(0, 1), 2), round(rng.gauss(0, 1), 2)) for _ in range(40)])
roc_d = csv("ref,score", [(1 if rng.random() < 0.5 else 0, round(rng.random(), 2)) for _ in range(40)])
kap = csv("r1,r2", [(rng.choice([1, 2, 3]), rng.choice([1, 2, 3])) for _ in range(30)])
meta_d = csv("study,effect,se", [("S1", 0.5, 0.2), ("S2", 0.3, 0.15), ("S3", 0.8, 0.3), ("S4", -0.1, 0.25), ("S5", 0.4, 0.18)])
forest_d = csv("label,yi,vi", [("S1", 0.5, 0.04), ("S2", 0.3, 0.0225), ("S3", 0.8, 0.09), ("S4", -0.1, 0.0625), ("S5", 0.4, 0.0324)])
surv_d = csv("time,event,grp", [(round(rng.expovariate(0.1), 1), 1 if rng.random() < 0.7 else 0, "A") for _ in range(15)] + [(round(rng.expovariate(0.07), 1), 1 if rng.random() < 0.7 else 0, "B") for _ in range(15)])
cox_d = csv("time,event,x1,x2", [(round(rng.expovariate(0.1), 1), 1 if rng.random() < 0.7 else 0, round(rng.gauss(0, 1), 2), round(rng.gauss(0, 1), 2)) for _ in range(40)])
irt_d = csv("i1,i2,i3,i4,i5", [[1 if rng.random() < 0.6 else 0 for _ in range(5)] for _ in range(25)])
fried_d = csv("m1,m2,m3", [(round(rng.gauss(0, 1), 2), round(rng.gauss(1, 1), 2), round(rng.gauss(2, 1), 2)) for _ in range(20)])

# ---------- case definitions ----------
# Each case: (name, py_json_str, rust_argv_list)
CASES = []


def add(name, py_fn, rust_args):
    CASES.append((name, py_fn, rust_args))


def P(**kw):
    return kw


# PSPP cases: py_fn returns the pspp_run string; rust args mirror it.
def pspp_case(name, t, d, rust_extra, **kw):
    add("pspp/" + name, lambda: pspp.pspp_run(t=t, d=d, **kw), ["pspp", t, "--csv", d] + rust_extra)


pspp_case("desc", "desc", one, ["--v", "v"], v=["v"])
pspp_case("freq", "freq", g2, ["--a", "g"], a="g")
pspp_case("examine", "examine", one, ["--v", "v"], v=["v"])
pspp_case("means", "means", g2, ["--a", "g", "--v", "v"], a="g", v="v")
pspp_case("crosstab2", "crosstab", ct, ["--a", "a", "--v", "v"], a="a", v="v")
pspp_case("crosstab3", "crosstab", ct3, ["--a", "a", "--v", "v"], a="a", v="v")
pspp_case("ttest", "ttest", g2, ["--a", "g", "--v", "v", "--g", "1,2"], a="g", v="v", g=[1, 2])
pspp_case("pttest", "pttest", paired, ["--a", "pre", "--v", "post"], a="pre", v="post")
pspp_case("ttest1", "ttest1", one, ["--v", "v", "--mu", "100"], v="v", mu=100)
pspp_case("anova", "anova", anova_d, ["--a", "g", "--v", "v"], a="g", v="v")
pspp_case("anova_bonf", "anova", anova_d, ["--a", "g", "--v", "v", "--posthoc", "bonferroni"], a="g", v="v", posthoc="bonferroni")
pspp_case("anova_gh", "anova", anova_d, ["--a", "g", "--v", "v", "--posthoc", "gh"], a="g", v="v", posthoc="gh")
pspp_case("mw", "mw", g2, ["--a", "g", "--v", "v", "--g", "1,2"], a="g", v="v", g=[1, 2])
pspp_case("kw", "kw", anova_d, ["--a", "g", "--v", "v"], a="g", v="v")
pspp_case("wilcoxon", "wilcoxon", paired, ["--a", "pre", "--v", "post"], a="pre", v="post")
pspp_case("friedman", "friedman", fried_d, ["--v", "m1,m2,m3"], v=["m1", "m2", "m3"])
pspp_case("sign", "sign", paired, ["--a", "pre", "--v", "post"], a="pre", v="post")
pspp_case("ks1", "ks", one, ["--v", "v"], v="v")
pspp_case("ks2", "ks", g2, ["--a", "g", "--v", "v", "--g", "1,2"], a="g", v="v", g=[1, 2])
pspp_case("runs", "runs", one, ["--v", "v"], v="v")
pspp_case("corr", "corr", corr_d, ["--v", "x,y,z"], v=["x", "y", "z"])
pspp_case("spearman", "spearman", corr_d, ["--v", "x,y,z"], v=["x", "y", "z"])
pspp_case("partial", "partial", corr_d, ["--v", "x,y", "--c", "z"], v=["x", "y"], c=["z"])
pspp_case("reg", "reg", reg_d, ["--o", "y", "--p", "x1,x2"], o="y", p=["x1", "x2"])
pspp_case("logistic", "logistic", reg_d, ["--o", "bin", "--p", "x1,x2"], o="bin", p=["x1", "x2"])
pspp_case("poisson", "poisson", reg_d, ["--o", "cnt", "--p", "x1,x2"], o="cnt", p=["x1", "x2"])
pspp_case("factor", "factor", items, ["--v", "i1,i2,i3,i4"], v=["i1", "i2", "i3", "i4"])
pspp_case("factor_varimax", "factor", items, ["--v", "i1,i2,i3,i4", "--rotate", "varimax"], v=["i1", "i2", "i3", "i4"], rotate="varimax")
pspp_case("reliability", "reliability", items, ["--v", "i1,i2,i3,i4"], v=["i1", "i2", "i3", "i4"])
pspp_case("roc", "roc", roc_d, ["--a", "ref", "--v", "score"], a="ref", v="score")
pspp_case("eval", "eval", roc_d, ["--a", "ref", "--v", "score"], a="ref", v="score")
pspp_case("surv", "surv", surv_d, ["--o", "event", "--v", "time", "--a", "grp"], o="event", v="time", a="grp")
pspp_case("meta", "meta", meta_d, ["--v", "study", "--effect-col", "effect", "--se-col", "se"], v="study", effect_col="effect", se_col="se")
pspp_case("blandaltman", "blandaltman", paired, ["--a", "pre", "--v", "post"], a="pre", v="post")
pspp_case("kappa", "kappa", kap, ["--a", "r1", "--v", "r2"], a="r1", v="r2")
pspp_case("evalue_or", "evalue", one, ["--or-val", "2.5"], or_val=2.5)
pspp_case("evalue_2x2", "evalue", ct, ["--a", "a", "--v", "v"], a="a", v="v")
pspp_case("power_ttest_n", "power", one, ["--power-test", "ttest", "--calc", "n", "--es", "0.5"], power_test="ttest", calc="n", es_val=0.5)
pspp_case("power_ttest_pw", "power", one, ["--power-test", "ttest", "--calc", "power", "--es", "0.5", "--n", "30"], power_test="ttest", calc="power", es_val=0.5, n=30)
pspp_case("power_prop_n", "power", one, ["--power-test", "prop", "--calc", "n", "--p0", "0.5", "--p1", "0.7"], power_test="prop", calc="n", p0=0.5, p1=0.7)
pspp_case("power_anova_pw", "power", one, ["--power-test", "anova", "--calc", "power", "--es", "0.25", "--n", "30", "--k", "3"], power_test="anova", calc="power", es_val=0.25, n=30, k=3)
pspp_case("rank", "rank", one, ["--v", "v"], v="v")
pspp_case("hedgesg", "hedgesg", g2, ["--a", "g", "--v", "v", "--g", "1,2"], a="g", v="v", g=[1, 2])
pspp_case("omega", "omega", items, ["--v", "i1,i2,i3,i4"], v=["i1", "i2", "i3", "i4"])

# Medical cases
add("med/km", lambda: med.medical_ext_run(action="km", d=surv_d, t="time", e="event"),
    ["medical_ext", "km", "--csv", surv_d, "--params", json.dumps({"t": "time", "e": "event"})])
add("med/km_grouped", lambda: med.medical_ext_run(action="km", d=surv_d, t="time", e="event", g="grp"),
    ["medical_ext", "km", "--csv", surv_d, "--params", json.dumps({"t": "time", "e": "event", "g": "grp"})])
add("med/logrank", lambda: med.medical_ext_run(action="logrank", d=surv_d, t="time", e="event", g="grp"),
    ["medical_ext", "logrank", "--csv", surv_d, "--params", json.dumps({"t": "time", "e": "event", "g": "grp"})])
add("med/cox", lambda: med.medical_ext_run(action="cox", d=cox_d, t="time", e="event", p=["x1", "x2"]),
    ["medical_ext", "cox", "--csv", cox_d, "--params", json.dumps({"t": "time", "e": "event", "p": ["x1", "x2"]})])
add("med/forest", lambda: med.medical_ext_run(action="forest", d=forest_d),
    ["medical_ext", "forest", "--csv", forest_d])
add("med/power_ttest_n", lambda: med.medical_ext_run(action="ttest", n=0, d_es=0.5, pw=0.8),
    ["medical_ext", "ttest", "--params", json.dumps({"n": 0, "d_es": 0.5, "pw": 0.8})])
add("med/power_ttest_pw", lambda: med.medical_ext_run(action="ttest", n=30, d_es=0.5, pw=0),
    ["medical_ext", "ttest", "--params", json.dumps({"n": 30, "d_es": 0.5, "pw": 0})])
add("med/power_ttest_d", lambda: med.medical_ext_run(action="ttest", n=30, d_es=0, pw=0.8),
    ["medical_ext", "ttest", "--params", json.dumps({"n": 30, "d_es": 0, "pw": 0.8})])
add("med/power_anova_n", lambda: med.medical_ext_run(action="anova", n=0, f=0.25, pw=0.8, k=3),
    ["medical_ext", "anova", "--params", json.dumps({"n": 0, "f": 0.25, "pw": 0.8, "k": 3})])
add("med/power_anova_pw", lambda: med.medical_ext_run(action="anova", n=159, f=0.25, pw=0, k=3),
    ["medical_ext", "anova", "--params", json.dumps({"n": 159, "f": 0.25, "pw": 0, "k": 3})])
add("med/power_prop_n", lambda: med.medical_ext_run(action="prop", n=0, p1=0.5, p2=0.7, pw=0.8),
    ["medical_ext", "prop", "--params", json.dumps({"n": 0, "p1": 0.5, "p2": 0.7, "pw": 0.8})])
add("med/power_prop_pw", lambda: med.medical_ext_run(action="prop", n=100, p1=0.5, p2=0.7, pw=0),
    ["medical_ext", "prop", "--params", json.dumps({"n": 100, "p1": 0.5, "p2": 0.7, "pw": 0})])

# IRT cases
add("irt/ctt", lambda: irt.irt_run(action="irt_ctt", d=irt_d),
    ["irt", "ctt", "--csv", irt_d])
add("irt/score_2pl", lambda: irt.irt_run(action="irt_score", d=irt_d, diff="-1,-0.5,0,0.5,1", method="mle", model="2pl"),
    ["irt", "score", "--csv", irt_d, "--params", json.dumps({"diff": "-1,-0.5,0,0.5,1", "method": "mle", "model": "2pl"})])
add("irt/score_3pl", lambda: irt.irt_run(action="irt_score", d=irt_d, diff="-1,-0.5,0,0.5,1", disc="1,1.2,0.8,1,1.1", guess="0.2,0.2,0.2,0.2,0.2", method="mle", model="3pl"),
    ["irt", "score", "--csv", irt_d, "--params", json.dumps({"diff": "-1,-0.5,0,0.5,1", "disc": "1,1.2,0.8,1,1.1", "guess": "0.2,0.2,0.2,0.2,0.2", "method": "mle", "model": "3pl"})])

# Error cases (both should return {"e": ...})
add("err/empty_csv", lambda: pspp.pspp_run(t="desc", d="", v=["v"]),
    ["pspp", "desc", "--csv", "", "--v", "v"])
add("err/missing_col", lambda: pspp.pspp_run(t="desc", d=one, v=["nope"]),
    ["pspp", "desc", "--csv", one, "--v", "nope"])
add("err/non_numeric", lambda: pspp.pspp_run(t="ttest1", d="v\n1\nabc\n3", v="v"),
    ["pspp", "ttest1", "--csv", "v\n1\nabc\n3", "--v", "v"])
add("err/small_n", lambda: pspp.pspp_run(t="ttest", d="g,v\n1,10\n2,20", a="g", v="v", g=[1, 2]),
    ["pspp", "ttest", "--csv", "g,v\n1,10\n2,20", "--a", "g", "--v", "v", "--g", "1,2"])
add("err/unknown_action", lambda: pspp.pspp_run(t="nosuch", d=one, v=["v"]),
    ["pspp", "nosuch", "--csv", one, "--v", "v"])

# ---------- comparison ----------
SKIP_KEYS = {"sw", "swp"}  # Shapiro-Wilk not implemented in Rust (null)


def num_close(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return False
        return abs(a - b) <= 5e-4 or (b != 0 and abs(a - b) / abs(b) <= 5e-3)
    return a == b


def compare(py, rs, path, diffs):
    if path.endswith("loadings") and isinstance(py, dict) and isinstance(rs, dict):
        # factor loadings: per-component sign is arbitrary (eigenvector sign)
        for k, v in py.items():
            if k not in rs:
                diffs.append(f"{path}.{k}: missing in Rust (py={v!r})")
            elif isinstance(v, (int, float)) and isinstance(rs[k], (int, float)):
                if not num_close(abs(v), abs(rs[k])):
                    diffs.append(f"{path}.{k}: |py|={abs(v)} |rs|={abs(rs[k])}")
            elif v != rs[k]:
                diffs.append(f"{path}.{k}: py={v!r} rs={rs[k]!r}")
        for k in rs:
            if k not in py:
                diffs.append(f"{path}.{k}: extra in Rust (rs={rs[k]!r})")
        return
    if isinstance(py, dict) and isinstance(rs, dict):
        for k, v in py.items():
            if k in SKIP_KEYS:
                continue
            if k not in rs:
                diffs.append(f"{path}.{k}: missing in Rust (py={v!r})")
            else:
                compare(v, rs[k], f"{path}.{k}", diffs)
        for k in rs:
            if k not in py and k not in SKIP_KEYS:
                diffs.append(f"{path}.{k}: extra in Rust (rs={rs[k]!r})")
    elif isinstance(py, list) and isinstance(rs, list):
        if len(py) != len(rs):
            diffs.append(f"{path}: length {len(py)} vs {len(rs)}")
        else:
            for i, (a, b) in enumerate(zip(py, rs)):
                compare(a, b, f"{path}[{i}]", diffs)
    elif isinstance(py, (int, float)) and isinstance(rs, (int, float)):
        if not num_close(py, rs):
            diffs.append(f"{path}: py={py} rs={rs}")
    else:
        if py != rs:
            # allow int/float equivalence
            if isinstance(py, (int, float)) and isinstance(rs, (int, float)) and num_close(py, rs):
                return
            diffs.append(f"{path}: py={py!r} rs={rs!r}")


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else ""
    results = []
    for name, py_fn, rust_args in CASES:
        if only and only not in name:
            continue
        try:
            py_out = json.loads(py_fn())
        except Exception as e:
            py_out = {"e": f"py harness error: {e}"}
        proc = subprocess.run([RUST] + rust_args, capture_output=True, text=True, timeout=300)
        try:
            rs_out = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {"e": "empty stdout"}
        except Exception:
            rs_out = {"e": f"unparsable stdout: {proc.stdout[:200]}"}
        # Unwrap PSPP k
        py_cmp = py_out.get("k", py_out) if isinstance(py_out, dict) and "k" in py_out else py_out
        rs_cmp = rs_out.get("k", rs_out) if isinstance(rs_out, dict) and "k" in rs_out else rs_out
        diffs = []
        if "e" in py_out or "e" in rs_out:
            if ("e" in py_out) != ("e" in rs_out):
                diffs.append(f"error mismatch: py={py_out.get('e')} rs={rs_out.get('e')}")
            # error messages need not match verbatim, only presence
        else:
            compare(py_cmp, rs_cmp, name, diffs)
        verdict = "PASS" if not diffs else "DIVERGENCE"
        results.append({"name": name, "verdict": verdict, "diffs": diffs[:20],
                        "py": py_out if verdict != "PASS" else None,
                        "rs": rs_out if verdict != "PASS" else None})
        print(f"{verdict:10s} {name}" + (f"  :: {diffs[0]}" if diffs else ""))
    npass = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"\n{npass}/{len(results)} PASS")
    (REPO / "scripts" / "deep_test_report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
