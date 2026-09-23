"""E7: depth sensitivity on digits (3 / 6 / 12 hidden layers, width 256),
5 seeds x 40 epochs; learning rates from E1 (no preconditioner for DFA/SSFA,
which E1/E2 showed is equivalent once the learning rate is tuned)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ssfa.runner import run, summarize

CONF = {"bp": ("bp", {"lr": 0.3}),
        "dfa": ("ssfa", {"lr": 0.5, "schedule": "dense", "precond": False}),
        "ssfa P5": ("ssfa", {"lr": 0.5, "P": 5, "precond": False})}
if __name__ == "__main__":
    sp = lambda m, kw, d: [dict(dataset="digits", method=m, kw={**kw, "depth": d}, seed=s, epochs=40, align=m == "ssfa") for s in range(5)]
    run([s for d in [3, 6, 12] for m, kw in CONF.values() for s in sp(m, kw, d)])
    out = {}
    for d in [3, 6, 12]:
        for n, (m, kw) in CONF.items():
            res = run(sp(m, kw, d))
            mu, sd = summarize(res)
            al = np.mean([[c for _, c in r["align"]] for r in res], 0).round(2).tolist() if "align" in res[0] else None
            out[f"depth{d}/{n}"] = dict(acc=mu, std=sd, align=al)
            print(f"hidden layers={d:2d} {n:8s} {mu:6.2f} +/- {sd:.2f} align={al}", flush=True)
    json.dump(out, open("results/e7_depth.json", "w"), indent=1)
