"""Parallel, cached experiment runner. One BLAS thread per worker so that
wall-clock timings are comparable across methods."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import multiprocessing as mp

import numpy as np
from threadpoolctl import threadpool_limits

# The env vars above are too late if a script imported numpy first; limit
# BLAS threads at runtime too (4 workers x 4 BLAS threads thrashed ~100x).
threadpool_limits(1)

from . import data
from .core import BP, SSFA, SparseBP

ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
_DATA = {}


def _key(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def dims_for(ds, width=256, depth=6):
    d_in = {"digits": 64}.get(ds, 196 if ds.endswith("_small") else 784)
    return [d_in] + [width] * depth + [10]


def run_one(spec):
    path = os.path.join(ROOT, _key(spec) + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    k = (spec["dataset"], spec.get("val", False))
    if k not in _DATA:
        _DATA[k] = data.load(*k)
    D = _DATA[k]
    kw = dict(spec.get("kw", {}))
    dims = dims_for(spec["dataset"], kw.pop("width", 256), kw.pop("depth", 6))
    cls = {"bp": BP, "ssfa": SSFA, "sbp": SparseBP}[spec["method"]]
    tr = cls(dims, spec["seed"], **kw)
    hist = tr.fit(D, spec["epochs"], eval_every_steps=spec.get("eval_every"))
    out = dict(spec=spec, hist=hist)
    if hasattr(tr, "bwd_layers"):
        out["bwd_layers_per_step"] = tr.bwd_layers / max(tr.t, 1)
    if spec["method"] == "ssfa" and spec.get("align"):
        Xtr, Ytr = D[0], D[1]
        out["align"] = tr.alignment(Xtr[:256], Ytr[:256])
    os.makedirs(ROOT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def run(specs, procs=4):
    specs = list(specs)
    todo = [s for s in specs if not os.path.exists(os.path.join(ROOT, _key(s) + ".json"))]
    if todo:
        with mp.get_context("fork").Pool(procs) as p:
            for _ in p.imap_unordered(run_one, todo):
                pass
    return [run_one(s) for s in specs]


def final_acc(results, last=1):
    """Mean test accuracy over the last `last` evaluations, per run."""
    return np.array([np.mean([h[3] for h in r["hist"][-last:]]) for r in results])


def summarize(results, last=1):
    a = final_acc(results, last) * 100
    return a.mean(), a.std()
