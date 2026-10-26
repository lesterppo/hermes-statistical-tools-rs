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

try:
    import numpy as _np  # only for the gee-poisson dataset (matches verify_sm.py)
except ImportError:
    _np = None

REPO = Path(__file__).resolve().parent.parent
import os
# Fixed Python clone (hst-py-fix) when set; default is the original scratch copy.
SCRATCH = Path(os.environ.get("HST_PY_TOOLS", str(Path.home() / "workspace" / "hst-py-orig")))
PYBIN = os.environ.get("HST_PYBIN", str(SCRATCH / ".venv" / "bin" / "python"))
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
sm = load("statsmodels_tool", TOOLS / "statsmodels_tool.py")
# SEM runs under its own venv (semopy lives there, not in the scratch venv).
SEMBIN = os.environ.get("HST_SEMBIN", str(Path.home() / "workspace" / "hst-sem-venv" / "bin" / "python"))
# SEM Python calls must share ONE process: sem_inspect reads sem_fit's
# in-process cache. SEM_SEQ collects (key, calldict); main() runs them
# once upfront and SEM_PY[key] holds each JSON string.
SEM_SEQ = []
SEM_PY = {}

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


# Bounded cases: same optimum / same method, but the reference
# implementation's own approximation stops short of the exact value.
# name -> (abs_tol, rel_tol, note). Used ONLY for the two cases below;
# every other case uses the strict deep-test tolerance.
BOUNDED = {}


def add_bounded(name, py_fn, rust_args, abs_tol, rel_tol, note):
    BOUNDED[name] = (abs_tol, rel_tol, note)
    CASES.append((name, py_fn, rust_args))


# Distributional / identifiable-only cases: (name, py_fn, rust_args,
# check_fn, note). check_fn(py_unwrapped, rs_unwrapped) returns a list
# of diff strings (empty = pass). Used ONLY for gee_ord (identifiable
# quantities; individual threshold/const splits are unidentifiable by
# construction) and mice (Python unseeded, Rust fixed-seed PMM draw).
DIST_CASES = []


def add_dist(name, py_fn, rust_args, check_fn, note):
    DIST_CASES.append((name, py_fn, rust_args, check_fn, note))


def sem_case(name, calldict, rust_args):
    # Registers a normal CASES entry whose Python output comes from the
    # shared SEM subprocess (see SEM_SEQ / main()).
    key = "sem/" + name
    SEM_SEQ.append((key, calldict))
    add(key, lambda: SEM_PY[key], rust_args)


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

# --- Fix-verification cases (2026-10-26 fix pass) ---
# ks2 with heavy ties (integer data): exact lattice DP must match scipy
# exact ks_2samp, including the n==n square-method case.
ties_d = csv("g,v", [(1, x) for x in [1, 1, 2, 2, 3, 3, 4, 5, 5, 6] * 2] + [(2, x) for x in [1, 2, 2, 3, 4, 4, 5, 6, 6, 7] * 2])
pspp_case("ks2_ties", "ks", ties_d, ["--a", "g", "--v", "v", "--g", "1,2"], a="g", v="v", g=[1, 2])
# forest with tau2 < 0 (homogeneous effects): Python previously raised
# "math domain error" via sqrt(negative var_re); fixed to RE = FE.
forest_neg_d = csv("label,yi,vi", [("S1", 0.50, 0.04), ("S2", 0.52, 0.0225), ("S3", 0.48, 0.09), ("S4", 0.51, 0.0625), ("S5", 0.49, 0.0324)])
add("med/forest_neg", lambda: med.medical_ext_run(action="forest", d=forest_neg_d),
    ["medical_ext", "forest", "--csv", forest_neg_d])
# cox with tied event times: Rust previously used Breslow, lifelines uses
# Efron; both coefficients, LL and concordance must now match.
cox_tied_d = csv("time,event,x1,x2", [(5, 1, 0.5, -0.3), (5, 0, -0.2, 0.8), (8, 1, 1.2, 0.1), (8, 1, -0.7, -0.5), (3, 1, 0.3, 0.4), (12, 0, 0.9, -0.2), (6, 1, -1.1, 0.6), (6, 0, 0.4, 0.0), (10, 1, 0.2, -0.9), (4, 1, -0.4, 0.3), (9, 0, 1.0, 0.5), (7, 1, 0.6, -0.4)])
add("med/cox_tied", lambda: med.medical_ext_run(action="cox", d=cox_tied_d, t="time", e="event", p=["x1", "x2"]),
    ["medical_ext", "cox", "--csv", cox_tied_d, "--params", json.dumps({"t": "time", "e": "event", "p": ["x1", "x2"]})])
# factor on a second dataset: eigenvalues/var_pct exact, loadings equal up
# to per-component sign (handled by the loadings branch in compare()).
# --- 10 new PSPP actions (2026-10-26 fix pass; datasets replicate
# scripts/verify_new10.py exactly: fresh Random(20261026), same draw order) ---
import math as _math
_rng2 = random.Random(20261026)
_reg_rows = []
for _ in range(40):
    _x1 = round(_rng2.gauss(0, 1), 2); _x2 = round(_rng2.gauss(0, 1), 2)
    _y = round(3 + 2*_x1 - 1.5*_x2 + _rng2.gauss(0, 1), 2)
    _b = 1 if (_x1 + _rng2.gauss(0, 0.5)) > 0 else 0
    _cnt = max(0, int(round(5 + 2*_x1 + _rng2.gauss(0, 2))))
    _reg_rows.append((_x1, _x2, _y, _b, _cnt))
_reg_d = csv("x1,x2,y,bin,cnt", _reg_rows)
_zi_rows = []
for _ in range(40):
    _x1 = round(_rng2.gauss(0, 1), 2)
    _cnt = 0 if _rng2.random() < 0.45 else max(0, int(round(3 + _x1 + _rng2.gauss(0, 1))))
    _zi_rows.append((_x1, _cnt))
_zi_d = csv("x1,cnt", _zi_rows)
_med_rows = []
for _ in range(40):
    _x = round(_rng2.gauss(0, 1), 2); _m = round(0.8*_x + _rng2.gauss(0, 1), 2)
    _y = round(1.5*_m + 0.4*_x + _rng2.gauss(0, 1), 2)
    _med_rows.append((_x, _m, _y))
_med_d = csv("x,m,y", _med_rows)
_dca_rows = []
for _ in range(40):
    _s = round(_rng2.random(), 2); _ev = 1 if _rng2.random() < _s else 0
    _dca_rows.append((_ev, _s))
_dca_d = csv("ev,score", _dca_rows)
_rw = [10.0]
for _ in range(49): _rw.append(round(_rw[-1] + _rng2.gauss(0, 1), 2))
_adf_d = csv("v", [(_v,) for _v in _rw])
_st = [round(_rng2.gauss(0, 1), 2) for _ in range(50)]
_adf2_d = csv("v", [(_v,) for _v in _st])
_beta_rows = []
for _ in range(40):
    _x1 = round(_rng2.gauss(0, 1), 2)
    _mu = 1/(1+_math.exp(-(0.3 + 0.8*_x1)))
    _yy = round(min(0.99, max(0.01, _mu + _rng2.gauss(0, 0.1))), 3)
    _beta_rows.append((_x1, _yy))
_beta_d = csv("x1,yy", _beta_rows)
pspp_case("dca", "dca", _dca_d, ["--a", "ev", "--v", "score"], a="ev", v="score")
pspp_case("quantreg_med", "quantreg", _reg_d, ["--o", "y", "--p", "x1,x2"], o="y", p=["x1", "x2"], tau=0.5)
pspp_case("quantreg_25", "quantreg", _reg_d, ["--o", "y", "--p", "x1,x2", "--tau", "0.25"], o="y", p=["x1", "x2"], tau=0.25)
pspp_case("quantreg_75", "quantreg", _reg_d, ["--o", "y", "--p", "x1", "--tau", "0.75"], o="y", p=["x1"], tau=0.75)
pspp_case("negbin", "negbin", _reg_d, ["--o", "cnt", "--p", "x1,x2"], o="cnt", p=["x1", "x2"])
pspp_case("negbin_1p", "negbin", _reg_d, ["--o", "cnt", "--p", "x1"], o="cnt", p=["x1"])
pspp_case("mediation", "mediation", _med_d, ["--o", "y", "--a", "x", "--v", "m"], o="y", a="x", v="m")
pspp_case("mediation_cov", "mediation", _med_d, ["--o", "y", "--a", "x", "--v", "m", "--p", "x1"], o="y", a="x", v="m", p=["x1"])
pspp_case("pscore", "pscore", _reg_d, ["--a", "bin", "--p", "x1,x2"], a="bin", p=["x1", "x2"])
pspp_case("gam", "gam", _reg_d, ["--o", "y", "--p", "x1,x2"], o="y", p=["x1", "x2"], df=5)
# NOTE: Python pspp_run's `df` is shadowed by `df_val` (default 5); the
# registry never passes df_val, so via the registry knots are ALWAYS 5.
# Rust --df follows the df_val path, so gam_df3 passes df_val=3 to Python.
pspp_case("gam_df3", "gam", _reg_d, ["--o", "y", "--p", "x1", "--df", "3"], o="y", p=["x1"], df_val=3)
pspp_case("zip", "zip", _zi_d, ["--o", "cnt", "--p", "x1"], o="cnt", p=["x1"])
pspp_case("zip_reg", "zip", _reg_d, ["--o", "cnt", "--p", "x1,x2"], o="cnt", p=["x1", "x2"])
pspp_case("lasso", "lasso", _reg_d, ["--o", "y", "--p", "x1,x2"], o="y", p=["x1", "x2"], lam=0.1)
pspp_case("lasso_big", "lasso", _reg_d, ["--o", "y", "--p", "x1,x2", "--lam", "5.0"], o="y", p=["x1", "x2"], lam=5.0)
pspp_case("adf_rw", "adf", _adf_d, ["--v", "v"], v="v", lags=1)
pspp_case("adf_st", "adf", _adf2_d, ["--v", "v", "--lags", "2"], v="v", lags=2)
pspp_case("beta", "beta", _beta_d, ["--o", "yy", "--p", "x1"], o="yy", p=["x1"])
pspp_case("beta_clip", "beta", csv("x1,yy", [(round(_rng2.gauss(0, 1), 2), _v) for _v in [0, 1, 0.5, 0.2, 0.9]*8]),
    ["--o", "yy", "--p", "x1"], o="yy", p=["x1"])

# --- IRT model fits (2026-10-26 fix pass) ---
# Datasets from verify_irt/gen.py (numpy default_rng(7)): BIN 200x6,
# BIN3 300x4 (with guessing), GRM 200x4 x 3 cats, PCM 200x4 x 3 cats,
# SMALL 20x4 hand matrix. Rows are digit strings, one per respondent.
_bin_d = 'i1,i2,i3,i4,i5,i6\n1,1,1,0,0,0\n1,1,1,1,0,0\n1,0,0,1,0,0\n1,0,0,0,1,0\n1,0,0,0,0,0\n0,1,1,0,0,0\n0,1,1,0,0,1\n1,1,1,0,1,1\n1,1,1,0,0,0\n0,0,1,1,0,0\n1,1,1,0,1,0\n0,1,1,1,0,0\n1,1,0,0,0,0\n1,1,0,0,0,0\n1,0,0,1,1,0\n1,1,0,0,0,0\n1,0,0,0,0,0\n1,1,0,1,0,0\n1,0,0,0,0,0\n1,1,0,0,0,0\n0,0,0,1,0,0\n1,0,0,0,1,0\n1,0,1,0,1,0\n1,0,1,1,0,0\n0,1,1,1,1,1\n1,1,1,0,0,0\n0,0,0,0,0,0\n1,1,0,0,0,0\n1,0,0,1,0,0\n1,1,0,1,0,0\n1,1,0,0,0,0\n0,0,0,0,0,0\n1,0,1,0,0,0\n0,1,0,1,0,0\n1,1,0,0,0,0\n1,1,0,0,0,0\n1,1,0,1,0,1\n0,1,1,0,0,0\n1,0,0,0,1,0\n1,1,1,0,1,0\n0,1,1,0,0,0\n0,1,0,0,0,0\n1,0,0,0,0,0\n1,1,1,0,0,0\n1,1,1,1,1,0\n0,0,0,0,0,0\n1,1,1,1,0,0\n1,1,0,0,0,0\n1,0,0,0,1,0\n1,1,1,1,0,0\n1,1,1,1,0,1\n1,0,0,1,0,0\n1,1,0,1,0,0\n1,1,1,0,0,1\n0,1,1,1,1,0\n1,1,1,1,1,0\n1,0,0,0,0,0\n1,1,1,1,1,0\n1,1,1,1,1,0\n1,1,1,0,1,0\n1,1,0,1,0,0\n1,0,1,0,0,0\n1,0,0,0,1,0\n0,1,0,1,0,0\n0,0,0,0,0,0\n1,1,0,1,0,0\n1,0,0,1,1,1\n1,1,1,0,1,0\n0,0,0,1,0,0\n1,0,0,1,0,0\n1,1,1,0,1,0\n1,0,0,1,0,0\n1,1,0,0,0,0\n0,0,0,0,0,0\n1,1,0,0,1,0\n1,1,0,1,1,0\n1,1,1,1,1,0\n1,0,0,1,0,0\n1,1,0,0,0,1\n1,1,0,1,1,1\n0,1,1,0,0,0\n0,1,1,0,1,1\n1,0,1,0,1,0\n0,1,0,0,1,0\n1,1,0,0,1,0\n0,0,0,0,0,0\n1,1,0,0,1,0\n1,0,0,1,0,0\n1,0,1,1,1,0\n1,1,1,1,0,0\n1,0,1,0,1,0\n1,1,1,1,1,1\n1,1,1,1,0,0\n1,1,1,1,0,1\n1,0,1,0,0,0\n1,1,1,1,0,0\n1,0,0,1,0,0\n1,0,1,1,0,0\n1,0,0,1,0,0\n0,0,0,0,0,0\n1,0,0,0,0,0\n0,0,0,0,0,0\n1,1,1,1,0,1\n1,1,1,1,0,1\n1,1,0,0,0,0\n1,1,1,0,0,0\n1,1,1,0,1,0\n1,1,1,0,1,0\n1,1,1,0,0,0\n0,0,1,0,0,0\n1,1,1,1,0,1\n1,1,1,1,0,0\n1,0,1,0,0,0\n1,1,1,1,1,1\n1,0,1,1,0,0\n1,1,0,0,0,0\n1,1,0,0,0,0\n1,1,0,1,0,0\n1,1,1,1,1,0\n0,1,0,1,0,0\n1,1,1,0,0,0\n0,1,1,0,0,0\n1,0,0,1,0,0\n0,0,0,0,0,0\n0,0,0,0,0,0\n1,1,1,0,0,0\n1,0,0,1,0,0\n1,1,1,1,0,1\n0,0,0,0,0,0\n1,1,0,1,0,0\n1,0,0,0,0,0\n1,1,1,0,0,0\n0,0,1,0,1,0\n0,0,0,0,0,0\n1,1,1,0,1,1\n1,1,1,1,1,1\n1,0,1,0,0,0\n1,0,1,0,0,0\n0,0,1,1,0,1\n1,0,1,1,0,0\n1,1,1,1,0,0\n1,1,0,0,0,0\n1,1,1,0,0,0\n1,0,0,0,0,0\n1,0,0,0,0,0\n1,1,0,0,0,0\n0,0,1,0,0,1\n1,1,0,1,0,0\n1,0,0,1,0,1\n1,0,1,0,0,0\n0,1,0,1,1,0\n1,1,0,1,0,0\n0,1,0,0,0,0\n1,0,0,0,0,0\n1,0,1,1,0,0\n1,0,1,1,0,0\n0,0,0,0,0,0\n1,0,0,1,0,0\n1,1,0,1,1,0\n0,0,0,0,0,0\n1,1,0,0,0,0\n1,1,0,1,0,0\n1,1,0,1,1,0\n1,0,0,0,0,0\n1,1,0,1,0,0\n1,1,0,0,0,0\n1,1,0,0,0,0\n0,1,1,1,0,0\n1,0,0,0,0,0\n0,1,1,0,1,0\n0,0,0,0,0,0\n0,1,1,1,0,1\n1,0,0,0,0,1\n1,1,1,0,0,0\n0,1,0,0,0,0\n1,0,1,0,0,0\n0,1,1,1,0,0\n1,0,1,0,0,0\n0,1,1,0,0,0\n1,1,1,1,1,0\n1,0,0,0,0,0\n1,0,0,0,0,0\n0,1,0,0,0,0\n1,1,0,0,0,1\n0,0,0,1,0,1\n1,1,0,0,1,0\n1,1,1,1,1,0\n0,0,1,0,1,0\n1,1,1,0,0,0\n1,1,0,0,0,0\n0,1,0,0,0,0\n1,0,1,1,0,1\n0,1,1,1,1,1\n1,0,0,0,0,0\n1,1,1,0,0,1\n1,0,0,0,0,0\n1,1,1,1,0,0\n1,1,1,0,1,0\n1,0,0,0,0,0\n1,1,1,1,1,0'
_bin3_d = 'i1,i2,i3,i4\n1,1,1,0\n1,0,0,0\n0,1,1,0\n1,0,1,1\n1,1,1,1\n1,1,0,0\n0,0,0,0\n1,1,1,1\n0,1,0,0\n1,1,0,1\n1,0,0,0\n1,1,1,1\n1,0,1,1\n1,1,0,1\n0,1,1,0\n0,1,0,0\n0,0,0,0\n1,1,1,1\n1,1,1,0\n1,1,1,1\n0,1,0,0\n1,1,0,1\n1,1,1,1\n1,1,1,1\n1,1,1,0\n0,0,0,0\n1,1,0,1\n0,0,1,0\n1,0,0,0\n0,0,0,1\n1,1,1,0\n1,0,1,0\n1,1,0,0\n1,1,1,0\n1,1,0,1\n0,1,0,0\n1,1,1,1\n1,0,0,1\n0,0,1,0\n1,1,0,1\n1,0,0,0\n1,0,0,1\n1,0,1,1\n1,1,0,0\n1,1,1,0\n1,0,1,0\n0,1,1,0\n1,0,0,1\n0,0,0,0\n1,0,0,0\n1,1,1,1\n1,1,1,1\n0,1,0,1\n0,1,0,0\n1,1,1,0\n1,0,1,1\n1,1,0,0\n1,0,0,1\n1,0,1,0\n1,1,0,1\n1,0,1,0\n0,1,1,0\n0,0,0,0\n1,1,0,1\n1,0,1,0\n1,0,0,0\n0,0,0,0\n1,1,0,0\n1,0,1,0\n1,1,1,0\n0,1,1,0\n0,0,1,1\n1,1,1,1\n1,0,1,1\n1,1,1,1\n1,1,0,0\n1,1,0,1\n0,0,0,0\n1,1,1,1\n1,1,1,0\n1,0,1,1\n0,1,1,1\n1,1,1,1\n0,0,0,0\n1,1,1,0\n1,0,0,1\n1,0,0,1\n0,1,0,1\n1,1,1,0\n1,1,0,0\n1,0,1,1\n1,1,1,0\n1,1,1,1\n1,1,0,1\n0,0,1,0\n0,0,1,0\n0,1,0,1\n0,0,1,0\n1,1,1,0\n1,0,0,0\n1,1,1,0\n1,0,0,0\n1,0,1,0\n1,0,1,1\n1,1,0,0\n0,1,1,0\n1,1,1,1\n1,1,1,1\n1,0,1,0\n1,0,1,0\n0,0,0,1\n0,1,0,0\n1,0,1,0\n1,0,1,1\n1,1,0,1\n0,1,0,0\n1,1,1,1\n0,0,0,0\n0,1,0,0\n1,0,0,0\n0,0,0,0\n0,0,1,0\n1,1,1,0\n1,1,1,1\n0,1,1,0\n1,1,0,0\n1,1,1,1\n1,1,0,0\n1,1,0,0\n1,1,1,0\n1,1,1,1\n1,1,0,1\n1,0,1,0\n1,1,1,1\n1,0,0,0\n1,1,0,0\n1,1,0,1\n1,1,1,0\n1,1,1,0\n1,1,0,0\n0,0,1,0\n1,0,0,1\n1,0,0,1\n1,0,1,0\n1,0,1,0\n1,0,0,1\n0,0,1,0\n1,0,1,0\n1,0,1,0\n1,1,1,1\n1,1,1,1\n1,0,1,0\n1,0,1,1\n1,1,1,0\n1,1,1,0\n1,1,0,0\n0,0,0,0\n1,1,1,1\n1,1,1,1\n0,0,0,0\n1,0,0,1\n1,1,1,1\n0,1,1,0\n1,1,0,0\n1,1,1,0\n1,1,0,1\n1,1,1,1\n1,0,0,0\n0,1,1,0\n0,1,0,0\n0,1,0,0\n1,1,1,1\n0,0,0,1\n1,1,0,1\n1,0,0,0\n0,1,1,0\n0,0,1,1\n1,0,0,0\n0,1,1,0\n1,1,1,1\n1,1,0,1\n1,1,1,1\n1,0,0,0\n1,0,0,0\n1,1,1,0\n0,1,0,0\n1,1,1,0\n0,0,1,0\n1,0,0,1\n1,0,1,0\n1,1,1,0\n1,1,1,0\n1,0,0,0\n1,1,0,0\n1,0,0,0\n1,1,0,1\n1,1,1,1\n1,0,0,0\n0,0,1,0\n0,0,1,1\n0,0,1,0\n1,0,1,1\n1,1,0,1\n0,0,1,0\n1,0,1,1\n0,0,0,0\n1,0,0,0\n1,1,1,0\n1,0,1,1\n1,0,1,1\n1,1,1,0\n1,0,0,1\n1,0,1,0\n1,1,1,1\n1,0,1,0\n1,1,0,0\n1,0,1,1\n1,1,1,1\n1,0,0,0\n0,0,0,0\n0,1,1,1\n0,0,0,0\n1,1,1,0\n1,1,0,0\n0,0,0,1\n1,0,0,0\n1,1,0,0\n1,0,1,0\n1,1,1,0\n1,0,1,1\n1,1,0,1\n0,0,1,1\n0,1,0,0\n1,1,1,1\n1,0,0,0\n1,1,0,1\n1,1,1,1\n1,0,0,1\n1,1,0,1\n1,1,0,0\n1,1,1,1\n0,0,1,0\n0,0,1,1\n1,0,0,0\n0,0,0,1\n1,1,1,1\n1,1,0,1\n1,0,0,0\n0,0,0,0\n1,1,0,1\n1,1,1,0\n0,0,1,0\n1,1,0,0\n0,1,0,1\n1,0,1,0\n1,1,0,0\n0,1,1,0\n1,0,0,0\n1,1,1,1\n1,0,1,1\n1,0,1,0\n1,1,1,0\n1,0,1,1\n1,0,0,1\n1,1,0,0\n0,0,0,0\n0,0,0,1\n1,0,1,0\n1,1,1,1\n1,1,0,0\n1,1,1,0\n0,1,1,0\n1,1,0,0\n0,0,1,1\n1,0,1,0\n1,1,1,1\n1,1,1,0\n1,1,0,1\n0,1,0,0\n1,1,1,1\n0,1,1,0\n0,0,1,1\n1,1,1,0\n1,0,0,0\n0,0,0,0\n1,1,1,0\n1,0,1,1\n1,1,1,1\n0,1,1,0\n1,1,1,0\n1,1,0,0\n1,0,0,1\n1,1,0,1\n1,1,0,0\n1,0,0,1\n0,1,0,1\n1,0,0,0\n1,1,0,0\n0,0,1,1\n1,1,0,0'
_grm_d = 'i1,i2,i3,i4\n0,0,0,0\n1,2,1,1\n2,1,0,1\n1,0,2,2\n1,2,0,1\n2,1,0,2\n2,1,0,0\n0,0,0,1\n2,1,2,2\n2,1,1,1\n2,0,0,2\n1,2,2,2\n1,2,0,1\n2,0,0,2\n1,2,2,2\n0,0,1,2\n0,0,0,0\n0,0,0,0\n1,1,1,1\n2,0,0,1\n2,2,1,2\n0,0,1,2\n0,1,0,1\n0,0,0,2\n2,2,2,2\n1,2,0,2\n0,2,2,2\n2,2,0,2\n2,2,2,2\n0,1,0,0\n0,1,0,1\n2,2,0,2\n2,1,1,0\n2,0,0,1\n1,1,0,2\n2,2,1,0\n0,1,1,2\n0,0,0,1\n2,0,2,2\n1,2,1,2\n1,1,0,2\n0,0,0,1\n2,2,1,2\n1,0,0,0\n0,0,1,0\n2,2,1,1\n0,0,1,1\n2,2,2,2\n0,0,1,0\n2,2,1,2\n1,2,0,2\n2,1,0,2\n0,1,1,1\n0,1,0,2\n0,1,0,1\n2,0,2,2\n1,0,0,0\n2,1,0,0\n1,2,1,2\n1,1,0,1\n2,0,0,0\n2,0,1,1\n1,0,1,2\n2,2,0,1\n2,2,2,2\n1,0,0,0\n2,1,2,2\n2,1,2,2\n1,0,1,2\n2,0,0,1\n2,1,0,1\n2,0,0,2\n1,1,0,2\n1,2,2,1\n2,2,2,0\n2,2,2,1\n0,1,0,0\n0,1,0,2\n1,0,2,2\n2,2,1,1\n2,1,1,2\n2,2,2,2\n2,1,1,2\n2,1,0,0\n2,2,0,2\n0,1,1,1\n2,2,0,2\n2,1,1,2\n2,2,0,0\n2,0,0,1\n2,0,0,0\n2,1,0,2\n2,1,0,1\n1,1,0,1\n2,1,1,2\n0,2,0,1\n1,0,0,2\n0,0,2,2\n1,0,0,1\n1,2,0,2\n0,0,0,2\n0,2,0,1\n0,2,0,2\n1,2,2,1\n1,0,0,0\n2,1,0,0\n2,0,0,0\n1,1,1,1\n0,1,0,0\n1,1,1,0\n1,1,0,2\n1,1,0,2\n2,2,2,2\n0,2,0,1\n0,1,2,2\n2,1,2,2\n2,1,1,2\n1,1,0,2\n2,2,0,1\n0,0,0,0\n2,2,0,2\n1,1,1,2\n2,1,1,2\n2,2,0,1\n0,2,0,1\n0,2,0,1\n2,2,2,2\n1,2,1,1\n0,1,1,1\n1,1,2,2\n2,2,1,1\n0,0,0,2\n0,0,0,1\n0,0,0,2\n1,0,2,1\n1,0,0,0\n0,1,0,2\n2,1,0,2\n1,1,1,2\n0,0,0,0\n0,0,0,1\n1,0,0,1\n1,0,0,2\n0,1,0,1\n2,1,2,2\n0,1,1,0\n0,2,0,0\n0,0,0,1\n2,2,2,2\n0,2,0,2\n0,0,0,0\n0,1,0,1\n1,1,1,1\n1,1,1,0\n2,0,2,0\n0,0,2,2\n2,1,0,1\n0,0,1,0\n2,0,0,2\n0,1,0,1\n1,1,2,2\n1,2,2,1\n2,0,1,1\n1,1,1,2\n2,2,0,1\n2,1,1,2\n1,1,2,1\n0,1,0,0\n2,1,2,2\n2,0,2,0\n0,0,1,1\n0,0,1,1\n2,0,2,2\n1,0,0,1\n2,1,0,2\n0,0,0,0\n1,1,0,2\n2,1,2,2\n2,1,1,1\n1,1,0,2\n1,1,1,2\n2,2,1,2\n1,0,0,1\n1,0,1,1\n1,1,2,0\n0,1,2,0\n2,0,1,0\n2,1,1,1\n0,0,0,0\n0,1,0,2\n1,0,1,2\n1,0,0,0\n1,0,1,0\n2,0,1,2\n0,0,0,0\n2,0,2,1\n0,0,1,1\n0,0,1,1\n2,1,1,2\n1,0,2,0'
_pcm_d = 'i1,i2,i3,i4\n1,1,2,2\n2,0,0,1\n1,1,0,2\n2,0,1,2\n1,1,1,2\n2,1,2,2\n2,1,1,2\n0,1,1,2\n1,1,1,2\n0,0,1,1\n2,1,0,0\n0,0,0,0\n1,0,1,2\n2,2,2,2\n2,2,1,2\n1,1,1,2\n1,0,1,1\n2,0,1,1\n1,1,0,1\n1,0,0,2\n1,2,1,2\n0,0,0,2\n0,1,1,2\n0,0,1,1\n0,0,0,0\n0,1,0,2\n0,0,0,0\n1,1,1,0\n0,2,2,1\n2,0,1,1\n2,2,0,2\n1,1,0,2\n2,0,1,2\n0,2,0,1\n2,2,1,2\n2,0,0,1\n0,2,1,2\n2,1,0,2\n0,0,0,0\n1,0,2,0\n1,0,0,1\n2,2,0,2\n1,0,0,2\n1,1,0,2\n1,1,1,1\n2,2,1,2\n1,1,1,2\n2,2,2,2\n1,1,1,2\n1,0,1,1\n1,2,0,2\n0,0,0,2\n1,1,0,2\n1,1,1,2\n1,1,0,2\n2,0,0,2\n2,2,0,2\n1,2,2,2\n1,1,0,0\n2,0,2,2\n2,1,0,0\n1,0,0,2\n2,1,0,2\n2,0,0,1\n2,2,1,2\n0,1,0,1\n2,2,2,2\n1,0,0,0\n1,0,1,0\n1,2,2,2\n1,0,2,1\n2,0,0,2\n1,0,0,1\n1,0,1,2\n0,0,0,1\n2,2,2,1\n1,0,0,0\n0,0,0,2\n0,1,1,0\n1,2,0,1\n0,0,0,0\n2,1,1,2\n1,0,2,1\n2,1,1,2\n1,0,0,2\n1,2,0,1\n1,1,1,1\n0,0,1,1\n1,0,0,2\n0,0,1,1\n0,0,0,0\n1,1,1,0\n1,1,0,2\n2,0,1,0\n2,1,0,1\n1,0,2,1\n2,2,1,2\n2,2,1,1\n1,0,0,2\n1,1,0,1\n1,0,2,0\n1,1,0,2\n1,1,2,2\n1,0,0,0\n2,1,2,2\n0,0,0,2\n1,1,1,1\n1,0,0,0\n2,2,2,2\n2,1,1,2\n1,1,1,2\n0,0,1,1\n2,2,1,2\n0,0,1,0\n2,1,0,2\n2,2,2,2\n2,1,1,2\n2,0,0,2\n1,2,1,2\n0,1,1,2\n1,1,1,1\n2,2,2,2\n2,2,0,2\n2,2,0,2\n2,0,1,0\n1,1,1,2\n1,0,1,0\n1,1,2,2\n2,2,2,2\n2,0,0,2\n0,0,0,0\n1,2,1,1\n0,0,0,1\n0,0,0,1\n0,0,0,1\n1,0,0,0\n2,1,1,2\n1,1,0,0\n0,1,1,1\n0,1,0,1\n2,2,1,2\n1,1,0,1\n2,1,0,2\n2,0,0,1\n1,2,1,2\n1,2,1,2\n0,2,1,0\n0,0,0,0\n2,0,1,1\n2,1,1,2\n1,0,0,2\n1,1,1,2\n2,1,0,1\n0,0,0,1\n2,1,1,1\n0,2,0,1\n1,0,0,2\n0,1,0,1\n2,2,2,2\n0,1,0,0\n0,1,1,2\n2,1,0,0\n1,2,0,2\n2,1,1,1\n0,1,0,1\n1,2,0,2\n2,1,2,2\n2,1,1,2\n1,1,1,2\n1,1,2,2\n0,0,0,2\n0,2,1,1\n1,0,1,1\n2,1,1,2\n1,2,0,2\n2,2,1,2\n2,0,1,1\n2,1,2,1\n0,0,1,1\n2,0,0,2\n1,0,1,1\n2,0,1,1\n2,1,2,1\n1,1,1,2\n1,0,0,2\n2,1,1,2\n1,2,0,2\n2,0,1,1\n0,0,0,0\n2,0,0,1\n0,0,0,0\n0,1,0,2\n1,1,0,1\n2,1,0,1\n1,0,0,1\n0,0,0,1\n1,1,0,1\n2,2,1,1\n1,1,2,0\n1,1,2,1'
_small_d = 'i1,i2,i3,i4\n1,1,0,0\n1,0,1,0\n0,1,1,1\n1,1,1,0\n0,0,0,1\n1,0,0,1\n0,1,0,0\n1,1,0,1\n0,0,1,1\n1,0,1,1\n0,1,1,0\n1,1,1,1\n0,0,0,0\n1,0,0,0\n0,1,0,1\n1,1,0,0\n0,0,1,0\n1,0,1,0\n0,1,1,1\n1,1,1,0'
add("irt/rasch_jml", lambda: irt.irt_run(action="irt_rasch", d=_bin_d, method="jml"),
    ["irt", "rasch", "--csv", _bin_d, "--params", json.dumps({"method": "jml"})])
add("irt/rasch_mml", lambda: irt.irt_run(action="irt_rasch", d=_bin_d, method="mml"),
    ["irt", "rasch", "--csv", _bin_d, "--params", json.dumps({"method": "mml"})])
add("irt/rasch_small", lambda: irt.irt_run(action="irt_rasch", d=_small_d, method="jml"),
    ["irt", "rasch", "--csv", _small_d, "--params", json.dumps({"method": "jml"})])
add("irt/2pl", lambda: irt.irt_run(action="irt_2pl", d=_bin_d),
    ["irt", "2pl", "--csv", _bin_d])
add("irt/2pl_small", lambda: irt.irt_run(action="irt_2pl", d=_small_d),
    ["irt", "2pl", "--csv", _small_d])
# 3pl: BOUNDED — same optimum, flat (b,g) tradeoff at the g=0 bound for
# item 2 (girth sits exactly on the bound; EM approaches it asymptotically).
# Max observed: d diff 2.72e-3, g diff 5.75e-4; LLs agree to 1e-4.
add_bounded("irt/3pl", lambda: irt.irt_run(action="irt_3pl", d=_bin3_d),
    ["irt", "3pl", "--csv", _bin3_d], 0.005, 0.01,
    "flat (b,g) tradeoff at g=0 bound; LL agrees to 1e-4")
add("irt/pcm", lambda: irt.irt_run(action="irt_pcm", d=_pcm_d),
    ["irt", "pcm", "--csv", _pcm_d])
# grm: BOUNDED — Rust is the exact MML optimum (LL -828.2533 vs girth
# -828.2653, Rust HIGHER by 0.0120, confirmed by independent scipy
# L-BFGS-B from three starts); girth stops early (500-point lookup
# interpolation, disc-change < 1e-3 stopping). Max threshold diff 0.0153.
add_bounded("irt/grm", lambda: irt.irt_run(action="irt_grm", d=_grm_d),
    ["irt", "grm", "--csv", _grm_d], 0.02, 5e-3,
    "Rust = exact MML optimum; girth stops early (LL +0.0120)")
add("irt/score_eap", lambda: irt.irt_run(action="irt_score", d=irt_d, diff="-1,-0.5,0,0.5,1", method="eap", model="2pl"),
    ["irt", "score", "--csv", irt_d, "--params", json.dumps({"diff": "-1,-0.5,0,0.5,1", "method": "eap", "model": "2pl"})])
add("irt/score_map", lambda: irt.irt_run(action="irt_score", d=irt_d, diff="-1,-0.5,0,0.5,1", method="map", model="2pl"),
    ["irt", "score", "--csv", irt_d, "--params", json.dumps({"diff": "-1,-0.5,0,0.5,1", "method": "map", "model": "2pl"})])
# --- StatsModels + SEM actions (2026-10-26 fix pass) ---
# Datasets replicate scripts/verify_sm.py exactly (fresh Random(20261026)
# for anova/gee/mlm in the same order; separate seeds for ord/mice/sem).
_sm_rng = random.Random(20261026)
_rows = []
for _s in range(1, 11):
    _base = _sm_rng.gauss(0, 2)
    for _lvl, _eff in enumerate([0, 3, 6]):
        _rows.append((_s, _lvl + 1, round(10 + _base + _eff + _sm_rng.gauss(0, 1), 2)))
_sm_anova1 = csv("subj,cond,y", _rows)
_rows = []
for _s in range(1, 11):
    _base = _sm_rng.gauss(0, 2)
    for _a in [0, 1]:
        for _b in [0, 1]:
            _rows.append((_s, _a, _b, round(10 + _base + 2*_a + 3*_b + 1*_a*_b + _sm_rng.gauss(0, 1), 2)))
_sm_anova2 = csv("subj,A,B,y", _rows)
_rows = []
for _s in range(1, 21):
    for _t in range(3):
        _x = round(_sm_rng.gauss(0, 1), 2)
        _rows.append((_s, _t, _x, round(2 + 1.5*_x + _sm_rng.gauss(0, 1), 2)))
_sm_gee_g = csv("id,t,x,y", _rows)
_rows = []
for _s in range(1, 31):
    for _t in range(2):
        _x = round(_sm_rng.gauss(0, 1), 2)
        _pr = 1 / (1 + math.exp(-(-0.5 + 1.2*_x)))
        _rows.append((_s, _t, _x, 1 if _sm_rng.random() < _pr else 0))
_sm_gee_b = csv("id,t,x,y", _rows)
_rows = []
for _s in range(1, 31):
    for _t in range(2):
        _x = round(_sm_rng.gauss(0, 1), 2)
        _lam = math.exp(1.0 + 0.4*_x)
        _cnt = int(_np.random.default_rng(_s * 10 + _t).poisson(_lam))
        _rows.append((_s, _t, _x, _cnt))
_sm_gee_p = csv("id,t,x,y", _rows)
_rows = []
for _s in range(1, 31):
    for _t in range(2):
        _x = round(_sm_rng.gauss(0, 1), 2)
        _mu = math.exp(0.5 + 0.3*_x)
        _rows.append((_s, _t, _x, round(_mu * (0.7 + 0.6*_sm_rng.random()), 3)))
_sm_gee_ga = csv("id,t,x,y", _rows)
_rows = []
for _s in range(1, 21):
    _re = _sm_rng.gauss(0, 1.5)
    for _t in range(4):
        _x = round(_sm_rng.gauss(0, 1), 2)
        _rows.append((_s, _x, round(3 + 2*_x + _re + _sm_rng.gauss(0, 1), 2)))
_sm_mlm = csv("id,x,y", _rows)
_r2 = random.Random(1000)
_rows = []
for _s in range(1, 31):
    for _t in range(2):
        _x = round(_r2.gauss(0, 1), 2)
        _lat = 0.8*_x + _r2.gauss(0, 1)
        _cat = 0 if _lat < -0.5 else (1 if _lat < 0.5 else 2)
        _rows.append((_s, _t, _x, _cat))
_sm_ord = csv("id,t,x,y", _rows)
_r3 = random.Random(3)
_rows = []
for _i in range(40):
    _x1 = round(_r3.gauss(0, 1), 2); _x2 = round(_r3.gauss(0, 1), 2)
    _y = round(2 + 1.5*_x1 - 0.8*_x2 + _r3.gauss(0, 1), 2)
    _rows.append(("" if _i in (5, 12, 20) else _x1, _x2, "" if _i in (8, 25) else _y))
_sm_mice = "x1,x2,y\n" + "\n".join(",".join(str(v) for v in r) for r in _rows)
_r4 = random.Random(5)
_rows = []
for _i in range(120):
    _f1 = _r4.gauss(0, 1); _f2 = 0.6*_f1 + _r4.gauss(0, 0.8)
    _rows.append((round(_f1 + _r4.gauss(0, 0.5), 3), round(0.9*_f1 + _r4.gauss(0, 0.5), 3),
                  round(1.1*_f1 + _r4.gauss(0, 0.5), 3), round(_f2 + _r4.gauss(0, 0.5), 3),
                  round(0.8*_f2 + _r4.gauss(0, 0.5), 3), round(1.2*_f2 + _r4.gauss(0, 0.5), 3)))
_sm_sem = csv("x1,x2,x3,y1,y2,y3", _rows)
_sm_sem_desc = "f1 =~ x1 + x2 + x3\nf2 =~ y1 + y2 + y3\nf2 ~ f1"
_r5 = random.Random(4)
_rows = []
for _i in range(150):
    _f1 = _r5.gauss(0, 1); _f2 = 0.4*_f1 + _r5.gauss(0, 1)
    _rows.append((round(_f1 + _r5.gauss(0, 0.5), 3), round(0.9*_f1 + _r5.gauss(0, 0.5), 3),
                  round(_f2 + _r5.gauss(0, 0.5), 3), round(0.8*_f2 + _r5.gauss(0, 0.5), 3)))
_sm_sem2 = csv("x1,x2,y1,y2", _rows)
_sm_sem2_desc = "f1 =~ x1 + x2\nf2 =~ y1 + y2"


def _sm_case(name, action, d, rust_extra, **kw):
    add("sm/" + name, lambda: sm.statsmodels_run(action=action, d=d, **kw),
        ["statsmodels", action, "--csv", d] + rust_extra)


_sm_case("anova_rm_1f", "anova_rm", _sm_anova1, ["--o", "y", "--a", "subj", "--w", "cond"],
         o="y", g="subj", w=["cond"])
_sm_case("anova_rm_2f", "anova_rm", _sm_anova2, ["--o", "y", "--a", "subj", "--params", '{"w":["A","B"]}'],
         o="y", g="subj", w=["A", "B"])
_sm_case("gee_gauss_ex", "gee", _sm_gee_g, ["--o", "y", "--p", "x", "--a", "id", "--fam", "gaussian", "--cov", "exchangeable"],
         o="y", p=["x"], g="id", fam="gaussian", cov="exchangeable")
_sm_case("gee_gauss_ind", "gee", _sm_gee_g, ["--o", "y", "--p", "x", "--a", "id", "--fam", "gaussian", "--cov", "independence"],
         o="y", p=["x"], g="id", fam="gaussian", cov="independence")
_sm_case("gee_gauss_ar1", "gee", _sm_gee_g, ["--o", "y", "--p", "x", "--a", "id", "--fam", "gaussian", "--cov", "ar1"],
         o="y", p=["x"], g="id", fam="gaussian", cov="ar1")
_sm_case("gee_binom_ex", "gee", _sm_gee_b, ["--o", "y", "--p", "x", "--a", "id", "--fam", "binomial", "--cov", "exchangeable"],
         o="y", p=["x"], g="id", fam="binomial", cov="exchangeable")
_sm_case("gee_binom_ind", "gee", _sm_gee_b, ["--o", "y", "--p", "x", "--a", "id", "--fam", "binomial", "--cov", "independence"],
         o="y", p=["x"], g="id", fam="binomial", cov="independence")
_sm_case("gee_pois_ex", "gee", _sm_gee_p, ["--o", "y", "--p", "x", "--a", "id", "--fam", "poisson", "--cov", "exchangeable"],
         o="y", p=["x"], g="id", fam="poisson", cov="exchangeable")
_sm_case("gee_gamma_ex", "gee", _sm_gee_ga, ["--o", "y", "--p", "x", "--a", "id", "--fam", "gamma", "--cov", "exchangeable"],
         o="y", p=["x"], g="id", fam="gamma", cov="exchangeable")
_sm_case("gee_nom_ex", "gee_nom", _sm_ord, ["--o", "y", "--p", "x", "--a", "id", "--cov", "exchangeable"],
         o="y", p=["x"], g="id", cov="exchangeable")
_sm_case("gee_nom_ind", "gee_nom", _sm_ord, ["--o", "y", "--p", "x", "--a", "id", "--cov", "independence"],
         o="y", p=["x"], g="id", cov="independence")
_sm_case("mlm", "mlm", _sm_mlm, ["--o", "y", "--p", "x", "--a", "id"],
         o="y", p=["x"], g="id")


def _gee_ord_check(py, rs):
    # Only identifiable quantities: threshold+const sums, slope, ll/aic/bic.
    # Individual threshold/const values are a null-space shift that varies
    # even between Python's own GLM/GEE/Ordinal fits (rank-deficient design:
    # const = sum of threshold dummies on every expanded row).
    diffs = []
    for key in ["ll", "aic", "bic"]:
        if key in py and key in rs and not num_close(py[key], rs[key]):
            diffs.append(f"{key}: py={py[key]} rs={rs[key]}")
    pf, rf = py.get("f", {}), rs.get("f", {})
    for k in ["x"]:
        for sub in ["c", "s"]:
            if k in pf and k in rf and not num_close(pf[k][sub], rf[k][sub]):
                diffs.append(f"f.{k}.{sub}: py={pf[k][sub]} rs={rf[k][sub]}")
    th = sorted([k for k in pf if k.startswith("I(y>")])
    for t in th:
        ps = pf[t]["c"] + pf["const"]["c"]
        rsum = rf[t]["c"] + rf["const"]["c"]
        if not num_close(ps, rsum):
            diffs.append(f"{t}+const: py={ps} rs={rsum}")
    return diffs


for _cov in ["exchangeable", "independence"]:
    add_dist(f"sm/gee_ord_{'ex' if _cov == 'exchangeable' else 'ind'}",
             lambda _c=_cov: sm.statsmodels_run(action="gee_ord", d=_sm_ord, o="y", p=["x"], g="id", cov=_c),
             ["statsmodels", "gee_ord", "--csv", _sm_ord, "--o", "y", "--p", "x", "--a", "id", "--cov", _cov],
             _gee_ord_check,
             "identifiable quantities only; threshold/const splits unidentifiable")


def _mice_check(py, rs):
    # Python MICE is unseeded (stochastic); Rust is a fixed-seed (20261026)
    # PMM draw from the same procedure. Bands: coefficients within 0.15
    # (~4 sd of Python's run-to-run spread, sd 0.02-0.03), fmi in [0, 0.6].
    diffs = []
    pf, rf = py.get("f", {}), rs.get("f", {})
    for k in ["Intercept", "x1", "x2"]:
        if k not in pf or k not in rf:
            diffs.append(f"missing coef {k}")
            continue
        if abs(pf[k]["c"] - rf[k]["c"]) > 0.15:
            diffs.append(f"f.{k}.c: py={pf[k]['c']} rs={rf[k]['c']} (band 0.15)")
    for k in ["Intercept", "x1", "x2"]:
        for o, tag in ((py, "py"), (rs, "rs")):
            v = o.get("fmi", {}).get(k)
            if v is None or not (0.0 <= v <= 0.6):
                diffs.append(f"fmi.{k}.{tag} out of [0, 0.6]: {v}")
    return diffs


add_dist("sm/mice",
         lambda: sm.statsmodels_run(action="mice", d=_sm_mice, o="y", p=["x1", "x2"], n=5, mod="ols"),
         ["statsmodels", "mice", "--csv", _sm_mice, "--o", "y", "--p", "x1,x2", "--n", "5", "--model-type", "ols"],
         _mice_check,
         "Python unseeded PMM vs Rust fixed-seed PMM; band 0.15")
sem_case("fit_2fac", {"action": "sem_fit", "d": _sm_sem, "desc": _sm_sem_desc},
         ["sem", "sem_fit", "--csv", _sm_sem, "--params", json.dumps({"desc": _sm_sem_desc})])
sem_case("inspect_raw", {"action": "sem_inspect", "d": _sm_sem, "desc": _sm_sem_desc, "mode": "raw"},
         ["sem", "sem_inspect", "--csv", _sm_sem, "--params", json.dumps({"desc": _sm_sem_desc, "mode": "raw"})])
sem_case("inspect_std_est", {"action": "sem_inspect", "d": _sm_sem, "desc": _sm_sem_desc, "mode": "std_est"},
         ["sem", "sem_inspect", "--csv", _sm_sem, "--params", json.dumps({"desc": _sm_sem_desc, "mode": "std_est"})])
sem_case("fit_autocov", {"action": "sem_fit", "d": _sm_sem2, "desc": _sm_sem2_desc},
         ["sem", "sem_fit", "--csv", _sm_sem2, "--params", json.dumps({"desc": _sm_sem2_desc})])

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
SKIP_KEYS = set()  # Shapiro-Wilk is now implemented in Rust (Royston AS R94)


CUR_TOL = (5e-4, 5e-3)  # (abs, rel); main() sets this per case from BOUNDED


def num_close(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return False
        return abs(a - b) <= CUR_TOL[0] or (b != 0 and abs(a - b) / abs(b) <= CUR_TOL[1])
    return a == b


def canon_params_list(rows):
    # SEM param tables: semopy's row ORDER and ~~ direction vary run to
    # run (hash order). Canonicalise each ~~ row's direction and sort
    # by (op, lv, rv) — keys are unique across loadings/variances/
    # covariances, so this only removes ordering noise, not values.
    out = []
    for r in rows:
        r = dict(r)
        if r.get("op") == "~~" and r.get("lv") != r.get("rv"):
            a, b = sorted([r["lv"], r["rv"]])
            r["lv"], r["rv"] = a, b
        out.append(r)
    return sorted(out, key=lambda r: (r.get("op", ""), r.get("lv", ""), r.get("rv", "")))


def compare(py, rs, path, diffs):
    if isinstance(py, list) and isinstance(rs, list) and py and rs \
            and all(isinstance(x, dict) and "lv" in x and "op" in x and "rv" in x for x in py + rs):
        # SEM params list: compare canonically (see canon_params_list).
        py, rs = canon_params_list(py), canon_params_list(rs)
        if len(py) != len(rs):
            diffs.append(f"{path}: length {len(py)} vs {len(rs)}")
        else:
            for i, (a, b) in enumerate(zip(py, rs)):
                compare(a, b, f"{path}[{i}]", diffs)
        return
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


def run_sem_py():
    # One subprocess for all SEM Python calls (shared fit cache).
    code = (
        "import sys, types, importlib.util, json\n"
        "class _Reg:\n    def register(self, **kw): pass\n"
        "_tools = types.ModuleType('tools'); _tools.__path__ = []\n"
        "_tools.registry = _Reg(); sys.modules['tools'] = _tools\n"
        "sys.modules['tools.registry'] = types.SimpleNamespace(registry=_Reg())\n"
        f"spec = importlib.util.spec_from_file_location('sem_tool', {str(TOOLS / 'sem_tool.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "calls = json.load(sys.stdin)\n"
        "for c in calls:\n"
        "    print(mod.sem_run(**c))\n"
    )
    proc = subprocess.run([SEMBIN, "-c", code], input=json.dumps([c for _, c in SEM_SEQ]),
                          capture_output=True, text=True, timeout=600)
    lines = proc.stdout.strip().splitlines() if proc.stdout.strip() else []
    for (key, _), line in zip(SEM_SEQ, lines):
        SEM_PY[key] = line
    for key, _ in SEM_SEQ:
        SEM_PY.setdefault(key, json.dumps({"e": f"sem subprocess produced no output: {proc.stderr[-300:]}"}))


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else ""
    if SEM_SEQ and (not only or "sem/" in only):
        run_sem_py()
    results = []
    for name, py_fn, rust_args in CASES:
        if only and only not in name:
            continue
        global CUR_TOL
        CUR_TOL = BOUNDED[name][:2] if name in BOUNDED else (5e-4, 5e-3)
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
        bnote = f"  [bounded: {BOUNDED[name][2]}]" if name in BOUNDED and verdict == "PASS" else ""
        print(f"{verdict:10s} {name}" + (f"  :: {diffs[0]}" if diffs else "") + bnote)
    for name, py_fn, rust_args, check_fn, note in DIST_CASES:
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
        py_cmp = py_out.get("k", py_out) if isinstance(py_out, dict) and "k" in py_out else py_out
        rs_cmp = rs_out.get("k", rs_out) if isinstance(rs_out, dict) and "k" in rs_out else rs_out
        if "e" in py_out or "e" in rs_out:
            diffs = [f"error mismatch: py={py_out.get('e')} rs={rs_out.get('e')}"] if ("e" in py_out) != ("e" in rs_out) else []
        else:
            diffs = check_fn(py_cmp, rs_cmp)
        verdict = "PASS" if not diffs else "DIVERGENCE"
        results.append({"name": name, "verdict": verdict, "diffs": diffs[:20],
                        "py": py_out if verdict != "PASS" else None,
                        "rs": rs_out if verdict != "PASS" else None})
        print(f"{verdict:10s} {name}" + (f"  :: {diffs[0]}" if diffs else "") + (f"  [distributional: {note}]" if verdict == "PASS" else ""))
    npass = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"\n{npass}/{len(results)} PASS")
    (REPO / "scripts" / "deep_test_report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
