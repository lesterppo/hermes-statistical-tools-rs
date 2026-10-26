#!/usr/bin/env python3
"""Verify the 10 new PSPP actions: Python original vs Rust binary.
Tolerance: |a-b| <= 5e-4 OR rel <= 5e-3 (deep-test rule). Strings exact.
Run: ~/workspace/hst-py-orig/.venv/bin/python scripts/verify_new10.py
"""
import importlib.util, json, math, random, subprocess, sys, types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path.home() / "workspace" / "hst-py-orig"
PYBIN = str(SCRATCH / ".venv" / "bin" / "python")
RUST = str(REPO / "target" / "release" / "hst")

class _Reg:
    def register(self, **kw): pass
_tools = types.ModuleType("tools"); _tools.__path__ = []
_tools.registry = _Reg(); sys.modules["tools"] = _tools
sys.modules["tools.registry"] = types.SimpleNamespace(registry=_Reg())
spec = importlib.util.spec_from_file_location("pspp_tool", SCRATCH / "tools" / "pspp_tool.py")
pspp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pspp)

rng = random.Random(20261026)
def csv(header, rows):
    return header + "\n" + "\n".join(",".join(str(x) for x in r) for r in rows)

# deterministic datasets
reg_rows = []
for _ in range(40):
    x1 = round(rng.gauss(0, 1), 2); x2 = round(rng.gauss(0, 1), 2)
    y = round(3 + 2*x1 - 1.5*x2 + rng.gauss(0, 1), 2)
    b = 1 if (x1 + rng.gauss(0, 0.5)) > 0 else 0
    cnt = max(0, int(round(5 + 2*x1 + rng.gauss(0, 2))))
    reg_rows.append((x1, x2, y, b, cnt))
reg_d = csv("x1,x2,y,bin,cnt", reg_rows)
# zero-inflated counts: many zeros
zi_rows = []
for _ in range(40):
    x1 = round(rng.gauss(0, 1), 2)
    cnt = 0 if rng.random() < 0.45 else max(0, int(round(3 + x1 + rng.gauss(0, 1))))
    zi_rows.append((x1, cnt))
zi_d = csv("x1,cnt", zi_rows)
# mediation: X -> M -> Y
med_rows = []
for _ in range(40):
    x = round(rng.gauss(0, 1), 2); m = round(0.8*x + rng.gauss(0, 1), 2)
    y = round(1.5*m + 0.4*x + rng.gauss(0, 1), 2)
    med_rows.append((x, m, y))
med_d = csv("x,m,y", med_rows)
# dca: outcome + predicted prob
dca_rows = []
for _ in range(40):
    s = round(rng.random(), 2); ev = 1 if rng.random() < s else 0
    dca_rows.append((ev, s))
dca_d = csv("ev,score", dca_rows)
# adf series: random walk-ish + stationary series
rw = [10.0]
for _ in range(49): rw.append(round(rw[-1] + rng.gauss(0, 1), 2))
adf_d = csv("v", [(v,) for v in rw])
st = [round(rng.gauss(0, 1), 2) for _ in range(50)]
adf2_d = csv("v", [(v,) for v in st])
# beta outcome in (0,1)
beta_rows = []
for _ in range(40):
    x1 = round(rng.gauss(0, 1), 2)
    mu = 1/(1+math.exp(-(0.3 + 0.8*x1)))
    yy = round(min(0.99, max(0.01, mu + rng.gauss(0, 0.1))), 3)
    beta_rows.append((x1, yy))
beta_d = csv("x1,yy", beta_rows)

CASES = []
def add(name, py_fn, rust_args):
    CASES.append((name, py_fn, rust_args))
def pc(name, t, d, extra, **kw):
    add(name, lambda: pspp.pspp_run(t=t, d=d, **kw), ["pspp", t, "--csv", d] + extra)

pc("dca", "dca", dca_d, ["--a", "ev", "--v", "score"], a="ev", v="score")
pc("quantreg_med", "quantreg", reg_d, ["--o", "y", "--p", "x1,x2"], o="y", p=["x1", "x2"], tau=0.5)
pc("quantreg_25", "quantreg", reg_d, ["--o", "y", "--p", "x1,x2", "--tau", "0.25"], o="y", p=["x1", "x2"], tau=0.25)
pc("quantreg_75", "quantreg", reg_d, ["--o", "y", "--p", "x1", "--tau", "0.75"], o="y", p=["x1"], tau=0.75)
pc("negbin", "negbin", reg_d, ["--o", "cnt", "--p", "x1,x2"], o="cnt", p=["x1", "x2"])
pc("negbin_1p", "negbin", reg_d, ["--o", "cnt", "--p", "x1"], o="cnt", p=["x1"])
pc("mediation", "mediation", med_d, ["--o", "y", "--a", "x", "--v", "m"], o="y", a="x", v="m")
pc("mediation_cov", "mediation", med_d, ["--o", "y", "--a", "x", "--v", "m", "--p", "x1"], o="y", a="x", v="m", p=["x1"])
pc("pscore", "pscore", reg_d, ["--a", "bin", "--p", "x1,x2"], a="bin", p=["x1", "x2"])
pc("gam", "gam", reg_d, ["--o", "y", "--p", "x1,x2"], o="y", p=["x1", "x2"], df=5)
# NOTE: Python pspp_run's `df` arg is shadowed by `df_val` (default 5):
# args["df"] = df_val if df_val else df. The registry never passes df_val,
# so via the registry knots are ALWAYS 5. Rust --df honours the df_val path.
pc("gam_df3", "gam", reg_d, ["--o", "y", "--p", "x1", "--df", "3"], o="y", p=["x1"], df_val=3)
pc("zip", "zip", zi_d, ["--o", "cnt", "--p", "x1"], o="cnt", p=["x1"])
pc("zip_reg", "zip", reg_d, ["--o", "cnt", "--p", "x1,x2"], o="cnt", p=["x1", "x2"])
pc("lasso", "lasso", reg_d, ["--o", "y", "--p", "x1,x2"], o="y", p=["x1", "x2"], lam=0.1)
pc("lasso_big", "lasso", reg_d, ["--o", "y", "--p", "x1,x2", "--lam", "5.0"], o="y", p=["x1", "x2"], lam=5.0)
pc("adf_rw", "adf", adf_d, ["--v", "v"], v="v", lags=1)
pc("adf_st", "adf", adf2_d, ["--v", "v", "--lags", "2"], v="v", lags=2)
pc("beta", "beta", beta_d, ["--o", "yy", "--p", "x1"], o="yy", p=["x1"])
pc("beta_clip", "beta", csv("x1,yy", [(round(rng.gauss(0,1),2), v) for v in [0,1,0.5,0.2,0.9]*8]),
   ["--o", "yy", "--p", "x1"], o="yy", p=["x1"])

def close(a, b):
    if isinstance(a, bool) or isinstance(b, bool): return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a-b) <= 5e-4 or (abs(b) > 0 and abs(a-b)/abs(b) <= 5e-3)
    return a == b

def cmp(py, rs, path=""):
    diffs = []
    if isinstance(py, dict) and isinstance(rs, dict):
        for k in py:
            if k not in rs: diffs.append(f"{path}.{k}: missing in Rust")
            else: diffs += cmp(py[k], rs[k], f"{path}.{k}")
        for k in rs:
            if k not in py: diffs.append(f"{path}.{k}: extra in Rust")
    elif isinstance(py, list) and isinstance(rs, list):
        if len(py) != len(rs): diffs.append(f"{path}: len {len(py)} vs {len(rs)}")
        for i, (a, b) in enumerate(zip(py, rs)): diffs += cmp(a, b, f"{path}[{i}]")
    elif not close(py, rs):
        diffs.append(f"{path}: py={py} rs={rs}")
    return diffs

npass = nfail = 0
for name, py_fn, rust_args in CASES:
    try: py = json.loads(py_fn())["k"]
    except Exception as e: py = {"e": f"py-error {e}"}
    r = subprocess.run([RUST] + rust_args, capture_output=True, text=True)
    try: rs = json.loads(r.stdout)
    except Exception: rs = {"e": f"rust unparsable: {r.stdout[:200]} {r.stderr[:200]}"}
    rs = rs.get("k", rs)
    diffs = cmp(py, rs)
    if not diffs:
        npass += 1; print(f"PASS {name}")
    else:
        nfail += 1; print(f"FAIL {name}")
        for d in diffs[:12]: print("   ", d)
print(f"\n{npass} passed, {nfail} failed of {len(CASES)}")
