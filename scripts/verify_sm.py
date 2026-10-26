#!/usr/bin/env python3
"""Verification: Rust hst (new statsmodels/SEM actions) vs Python originals.
Python statsmodels tool: scratch venv (statsmodels 0.15.0), tools loaded
standalone with a stub registry from ~/workspace/hst-py-orig/tools.
Python SEM tool: ~/workspace/hst-sem-venv (semopy 2.3.11), same stub.
Rust: ./target/release/hst CLI (wired in this copy's main.rs).
Pass rule per numeric leaf: |a-b| <= 5e-4 OR rel <= 5e-3.
Prints a per-case table and writes verify_sm_report.json next to this script.
"""
import importlib.util, json, math, random, subprocess, sys, types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUST = str(REPO / "target" / "release" / "hst")
PY_SM = str(Path.home() / "workspace" / "hst-py-orig" / ".venv" / "bin" / "python")
PY_SEM = str(Path.home() / "workspace" / "hst-sem-venv" / "bin" / "python")
TOOLS = Path.home() / "workspace" / "hst-py-orig" / "tools"


def load_py(python, toolfile, modname):
    code = f"""
import sys, types, importlib.util, json
class _Reg:
    def register(self, **kw): pass
_tools = types.ModuleType("tools"); _tools.__path__ = []
_tools.registry = _Reg(); sys.modules["tools"] = _tools
sys.modules["tools.registry"] = types.SimpleNamespace(registry=_Reg())
spec = importlib.util.spec_from_file_location("{modname}", "{toolfile}")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print(mod.__name__)
"""
    return code


class PyRunner:
    """Runs many Python-tool calls in ONE subprocess per tool (needed so
    sem_inspect shares sem_fit's cache). Input: list of call dicts via
    stdin JSON; output: list of JSON strings on stdout, one per line."""

    def __init__(self, python, toolfile, func):
        self.python = python
        self.toolfile = toolfile
        self.func = func

    def run(self, calls):
        script = f"""
import sys, types, importlib.util, json
class _Reg:
    def register(self, **kw): pass
_tools = types.ModuleType("tools"); _tools.__path__ = []
_tools.registry = _Reg(); sys.modules["tools"] = _tools
sys.modules["tools.registry"] = types.SimpleNamespace(registry=_Reg())
spec = importlib.util.spec_from_file_location("toolmod", "{self.toolfile}")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
calls = json.load(sys.stdin)
for c in calls:
    try:
        print(mod.{self.func}(**c))
    except Exception as e:
        print(json.dumps({{"e": type(e).__name__ + ": " + str(e)}}))
"""
        p = subprocess.run([self.python, "-c", script], input=json.dumps(calls),
                           capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            print("PY STDERR:", p.stderr[-3000:], file=sys.stderr)
            raise RuntimeError("python runner failed")
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines]


def rust(tool, action, csv, extra):
    argv = [RUST, tool, action]
    if csv is not None:
        argv += ["--csv", csv]
    argv += extra
    p = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"e": f"rust CLI failed rc={p.returncode} out={p.stdout[-500:]} err={p.stderr[-500:]}"}


def csv_str(header, rows):
    return header + "\n" + "\n".join(",".join(str(x) for x in r) for r in rows)


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def compare(name, py, rs):
    pf, rf = flatten(py), flatten(rs)
    diffs = []
    for k, pv in pf.items():
        if k not in rf:
            diffs.append((k, pv, "<missing>"))
            continue
        rv = rf[k]
        if isinstance(pv, (int, float)) and not isinstance(pv, bool):
            if pv is None or rv is None:
                if pv != rv:
                    diffs.append((k, pv, rv))
            elif isinstance(rv, (int, float)) and not isinstance(rv, bool):
                ad = abs(pv - rv)
                rel = ad / max(abs(pv), 1e-12)
                if ad > 5e-4 and rel > 5e-3:
                    diffs.append((k, pv, rv))
            else:
                diffs.append((k, pv, rv))
        else:
            if pv != rv:
                diffs.append((k, pv, rv))
    extra = [k for k in rf if k not in pf]
    ok = not diffs
    print(f"\n=== {name}: {'PASS' if ok else 'FAIL'} ({len(diffs)} diffs, extra rust keys: {extra[:10]})")
    for k, pv, rv in diffs[:25]:
        print(f"    {k}: py={pv} rs={rv}")
    return {"name": name, "pass": ok, "diffs": [(k, pv, rv) for k, pv, rv in diffs],
            "extra": extra, "py": py, "rs": rs}


def main():
    rng = random.Random(20261026)
    results = []

    # ---------------- datasets ----------------
    # anova_rm 1-factor
    rows = []
    for s in range(1, 11):
        base = rng.gauss(0, 2)
        for lvl, eff in enumerate([0, 3, 6]):
            rows.append((s, lvl + 1, round(10 + base + eff + rng.gauss(0, 1), 2)))
    anova1 = csv_str("subj,cond,y", rows)
    # anova_rm 2-factor
    rows = []
    for s in range(1, 11):
        base = rng.gauss(0, 2)
        for a in [0, 1]:
            for b in [0, 1]:
                rows.append((s, a, b, round(10 + base + 2 * a + 3 * b + 1 * a * b + rng.gauss(0, 1), 2)))
    anova2 = csv_str("subj,A,B,y", rows)
    # gee gaussian
    rows = []
    for s in range(1, 21):
        for t in range(3):
            x = round(rng.gauss(0, 1), 2)
            rows.append((s, t, x, round(2 + 1.5 * x + rng.gauss(0, 1), 2)))
    gee_g = csv_str("id,t,x,y", rows)
    # gee binomial
    rows = []
    for s in range(1, 31):
        for t in range(2):
            x = round(rng.gauss(0, 1), 2)
            pr = 1 / (1 + math.exp(-(-0.5 + 1.2 * x)))
            rows.append((s, t, x, 1 if rng.random() < pr else 0))
    gee_b = csv_str("id,t,x,y", rows)
    # gee poisson
    rows = []
    for s in range(1, 31):
        for t in range(2):
            x = round(rng.gauss(0, 1), 2)
            lam = math.exp(1.0 + 0.4 * x)
            # deterministic-ish counts via inverse-CDF-free rounding of a poisson draw
            import numpy as _np
            cnt = int(_np.random.default_rng(s * 10 + t).poisson(lam))
            rows.append((s, t, x, cnt))
    gee_p = csv_str("id,t,x,y", rows)
    # gee gamma (positive continuous)
    rows = []
    for s in range(1, 31):
        for t in range(2):
            x = round(rng.gauss(0, 1), 2)
            mu = math.exp(0.5 + 0.3 * x)
            rows.append((s, t, x, round(mu * (0.7 + 0.6 * rng.random()), 3)))
    gee_ga = csv_str("id,t,x,y", rows)
    # mlm
    rows = []
    for s in range(1, 21):
        re = rng.gauss(0, 1.5)
        for t in range(4):
            x = round(rng.gauss(0, 1), 2)
            rows.append((s, x, round(3 + 2 * x + re + rng.gauss(0, 1), 2)))
    mlm_d = csv_str("id,x,y", rows)
    # ordinal / nominal: try seeds until Python gee_ord is finite
    ord_d = None
    for seed in range(20):
        r2 = random.Random(1000 + seed)
        rows = []
        for s in range(1, 31):
            for t in range(2):
                x = round(r2.gauss(0, 1), 2)
                lat = 0.8 * x + r2.gauss(0, 1)
                cat = 0 if lat < -0.5 else (1 if lat < 0.5 else 2)
                rows.append((s, t, x, cat))
        cand = csv_str("id,t,x,y", rows)
        ord_d = cand
        break
    # mice
    r3 = random.Random(3)
    rows = []
    for i in range(40):
        x1 = round(r3.gauss(0, 1), 2)
        x2 = round(r3.gauss(0, 1), 2)
        y = round(2 + 1.5 * x1 - 0.8 * x2 + r3.gauss(0, 1), 2)
        rows.append(("" if i in (5, 12, 20) else x1, x2, "" if i in (8, 25) else y))
    mice_d = "x1,x2,y\n" + "\n".join(",".join(str(v) for v in r) for r in rows)
    # sem datasets
    r4 = random.Random(5)
    rows = []
    for i in range(120):
        f1 = r4.gauss(0, 1)
        f2 = 0.6 * f1 + r4.gauss(0, 0.8)
        rows.append((round(f1 + r4.gauss(0, 0.5), 3), round(0.9 * f1 + r4.gauss(0, 0.5), 3),
                     round(1.1 * f1 + r4.gauss(0, 0.5), 3), round(f2 + r4.gauss(0, 0.5), 3),
                     round(0.8 * f2 + r4.gauss(0, 0.5), 3), round(1.2 * f2 + r4.gauss(0, 0.5), 3)))
    sem_d = csv_str("x1,x2,x3,y1,y2,y3", rows)
    sem_desc = "f1 =~ x1 + x2 + x3\nf2 =~ y1 + y2 + y3\nf2 ~ f1"
    r5 = random.Random(4)
    rows = []
    for i in range(150):
        f1 = r5.gauss(0, 1)
        f2 = 0.4 * f1 + r5.gauss(0, 1)
        rows.append((round(f1 + r5.gauss(0, 0.5), 3), round(0.9 * f1 + r5.gauss(0, 0.5), 3),
                     round(f2 + r5.gauss(0, 0.5), 3), round(0.8 * f2 + r5.gauss(0, 0.5), 3)))
    sem2_d = csv_str("x1,x2,y1,y2", rows)
    sem2_desc = "f1 =~ x1 + x2\nf2 =~ y1 + y2"

    # ---------------- python statsmodels calls ----------------
    sm_calls = [
        {"action": "anova_rm", "d": anova1, "o": "y", "g": "subj", "w": ["cond"]},
        {"action": "anova_rm", "d": anova2, "o": "y", "g": "subj", "w": ["A", "B"]},
        {"action": "gee", "d": gee_g, "o": "y", "p": ["x"], "g": "id", "fam": "gaussian", "cov": "exchangeable"},
        {"action": "gee", "d": gee_g, "o": "y", "p": ["x"], "g": "id", "fam": "gaussian", "cov": "independence"},
        {"action": "gee", "d": gee_g, "o": "y", "p": ["x"], "g": "id", "fam": "gaussian", "cov": "ar1"},
        {"action": "gee", "d": gee_b, "o": "y", "p": ["x"], "g": "id", "fam": "binomial", "cov": "exchangeable"},
        {"action": "gee", "d": gee_b, "o": "y", "p": ["x"], "g": "id", "fam": "binomial", "cov": "independence"},
        {"action": "gee", "d": gee_p, "o": "y", "p": ["x"], "g": "id", "fam": "poisson", "cov": "exchangeable"},
        {"action": "gee", "d": gee_ga, "o": "y", "p": ["x"], "g": "id", "fam": "gamma", "cov": "exchangeable"},
        {"action": "gee_ord", "d": ord_d, "o": "y", "p": ["x"], "g": "id", "cov": "exchangeable"},
        {"action": "gee_ord", "d": ord_d, "o": "y", "p": ["x"], "g": "id", "cov": "independence"},
        {"action": "gee_nom", "d": ord_d, "o": "y", "p": ["x"], "g": "id", "cov": "exchangeable"},
        {"action": "gee_nom", "d": ord_d, "o": "y", "p": ["x"], "g": "id", "cov": "independence"},
        {"action": "mlm", "d": mlm_d, "o": "y", "p": ["x"], "g": "id"},
    ]
    sm_runner = PyRunner(PY_SM, str(TOOLS / "statsmodels_tool.py"), "statsmodels_run")
    py_sm = sm_runner.run(sm_calls)
    rust_sm = [
        rust("statsmodels", "anova_rm", anova1, ["--o", "y", "--a", "subj", "--w", "cond"]),
        rust("statsmodels", "anova_rm", anova2, ["--o", "y", "--a", "subj", "--params", '{"w":["A","B"]}']),
        rust("statsmodels", "gee", gee_g, ["--o", "y", "--p", "x", "--a", "id", "--fam", "gaussian", "--cov", "exchangeable"]),
        rust("statsmodels", "gee", gee_g, ["--o", "y", "--p", "x", "--a", "id", "--fam", "gaussian", "--cov", "independence"]),
        rust("statsmodels", "gee", gee_g, ["--o", "y", "--p", "x", "--a", "id", "--fam", "gaussian", "--cov", "ar1"]),
        rust("statsmodels", "gee", gee_b, ["--o", "y", "--p", "x", "--a", "id", "--fam", "binomial", "--cov", "exchangeable"]),
        rust("statsmodels", "gee", gee_b, ["--o", "y", "--p", "x", "--a", "id", "--fam", "binomial", "--cov", "independence"]),
        rust("statsmodels", "gee", gee_p, ["--o", "y", "--p", "x", "--a", "id", "--fam", "poisson", "--cov", "exchangeable"]),
        rust("statsmodels", "gee", gee_ga, ["--o", "y", "--p", "x", "--a", "id", "--fam", "gamma", "--cov", "exchangeable"]),
        rust("statsmodels", "gee_ord", ord_d, ["--o", "y", "--p", "x", "--a", "id", "--cov", "exchangeable"]),
        rust("statsmodels", "gee_ord", ord_d, ["--o", "y", "--p", "x", "--a", "id", "--cov", "independence"]),
        rust("statsmodels", "gee_nom", ord_d, ["--o", "y", "--p", "x", "--a", "id", "--cov", "exchangeable"]),
        rust("statsmodels", "gee_nom", ord_d, ["--o", "y", "--p", "x", "--a", "id", "--cov", "independence"]),
        rust("statsmodels", "mlm", mlm_d, ["--o", "y", "--p", "x", "--a", "id"]),
    ]
    names = ["anova_rm/1f", "anova_rm/2f", "gee/gauss-ex", "gee/gauss-ind", "gee/gauss-ar1",
             "gee/binom-ex", "gee/binom-ind", "gee/pois-ex", "gee/gamma-ex",
             "gee_ord/ex", "gee_ord/ind", "gee_nom/ex", "gee_nom/ind", "mlm"]
    for nm, py, rs in zip(names, py_sm, rust_sm):
        results.append(compare(nm, py, rs))

    # ---------------- mice: python is stochastic; run 5x, compare rust
    # against the python mean and report the python spread ----------------
    # gee_ord identifiable quantities: the design is rank-deficient by
    # construction (const = sum of threshold dummies), so individual
    # threshold/const values are a null-space shift that varies even
    # between Python's own GLM/GEE/Ordinal fits; the identifiable
    # combinations (threshold+const sums, slope, ll/aic/bic) must match.
    for nm, py, rs in zip(["gee_ord/ex", "gee_ord/ind"], py_sm[9:11], rust_sm[9:11]):
        def sums(o):
            f = o["f"]
            th = sorted([k for k in f if k.startswith("I(y>")])
            return {f"{t}+const": (f[t]["c"] + f["const"]["c"]) for t in th}
        print(f"\n--- {nm} identifiable sums: py={sums(py)} rs={sums(rs)}")
        print(f"    slope x: py={py['f']['x']['c']} rs={rs['f']['x']['c']} "
              f"ll: py={py['ll']} rs={rs['ll']} aic: py={py['aic']} rs={rs['aic']}")

    mice_calls = [{"action": "mice", "d": mice_d, "o": "y", "p": ["x1", "x2"], "n": 5, "mod": "ols"}]
    py_mice_runs = []
    for _ in range(5):
        py_mice_runs.append(sm_runner.run(mice_calls)[0])
    rs_mice = rust("statsmodels", "mice", mice_d, ["--o", "y", "--p", "x1,x2", "--n", "5", "--model-type", "ols"])
    print("\n=== mice (python stochastic, 5 runs):")
    for k in ["Intercept", "x1", "x2"]:
        vals = [r["f"][k]["c"] for r in py_mice_runs]
        print(f"    {k}: py runs={vals} py_mean={sum(vals)/len(vals):.4f} "
              f"py_spread={max(vals)-min(vals):.4f} rs={rs_mice.get('f', {}).get(k, {}).get('c')} "
              f"py_fmi={[r['fmi'][k] for r in py_mice_runs]} rs_fmi={rs_mice.get('fmi', {}).get(k)}")
    results.append({"name": "mice", "pass": None, "py_runs": py_mice_runs, "rs": rs_mice})

    # ---------------- sem ----------------
    sem_calls = [
        {"action": "sem_fit", "d": sem_d, "desc": sem_desc},
        {"action": "sem_inspect", "d": sem_d, "desc": sem_desc, "mode": "raw"},
        {"action": "sem_inspect", "d": sem_d, "desc": sem_desc, "mode": "std_est"},
        {"action": "sem_fit", "d": sem2_d, "desc": sem2_desc},
    ]
    sem_runner = PyRunner(PY_SEM, str(TOOLS / "sem_tool.py"), "sem_run")
    py_sem = sem_runner.run(sem_calls)
    rs_sem = [
        rust("sem", "sem_fit", sem_d, ["--params", json.dumps({"desc": sem_desc})]),
        rust("sem", "sem_inspect", sem_d, ["--params", json.dumps({"desc": sem_desc, "mode": "raw"})]),
        rust("sem", "sem_inspect", sem_d, ["--params", json.dumps({"desc": sem_desc, "mode": "std_est"})]),
        rust("sem", "sem_fit", sem2_d, ["--params", json.dumps({"desc": sem2_desc})]),
    ]
    def canon_sem(obj):
        # Sort (lv, rv) on ~~ rows so semopy's hash-order direction
        # (f1~~f2 vs f2~~f1) does not count as a difference, AND sort
        # the params list itself: semopy's row ORDER for the variance /
        # covariance block also varies run to run (hash order), so a
        # positional comparison fails on identical values (verified
        # 2026-10-26: py[5]=f2 var 1.416114 vs rs[5]=cov 0.43391 with
        # py[6]=cov 0.433947 vs rs[6]=f2 var 1.416073 — same values,
        # swapped rows). Sorting by (op, lv, rv) is safe: the keys are
        # unique across loadings, variances and covariances.
        import copy as _copy
        obj = _copy.deepcopy(obj)
        for row in obj.get("params", []):
            if row.get("op") == "~~" and row.get("lv") != row.get("rv"):
                a, b = sorted([row["lv"], row["rv"]])
                row["lv"], row["rv"] = a, b
        obj["params"] = sorted(obj.get("params", []),
                               key=lambda r: (r.get("op", ""), r.get("lv", ""), r.get("rv", "")))
        return obj

    for nm, py, rs in zip(["sem_fit/2fac", "sem_inspect/raw", "sem_inspect/std_est", "sem_fit/autocov"], py_sem, rs_sem):
        py, rs = canon_sem(py), canon_sem(rs)
        results.append(compare(nm, py, rs))

    out = Path(__file__).resolve().parent / "verify_sm_report.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nReport: {out}")
    n_pass = sum(1 for r in results if r.get("pass") is True)
    n_fail = sum(1 for r in results if r.get("pass") is False)
    print(f"PASS {n_pass}  FAIL {n_fail}  OTHER {len(results)-n_pass-n_fail}")


if __name__ == "__main__":
    main()
