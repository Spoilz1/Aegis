"""E1: per-method learning-rate selection on a validation split (digits).
Selection = mean val accuracy over the last 5 of 40 epochs, 3 seeds."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ssfa.runner import run, final_acc

GRID = {
    "bp":            ("bp",   {},                                   [0.01, 0.03, 0.1, 0.3, 1.0]),
    "bp_b8":         ("bp",   {"batch": 8},                         [0.003, 0.01, 0.03, 0.1, 0.3]),
    "bp_accum5":     ("bp",   {"accum": 5},                         [0.03, 0.1, 0.3, 1.0]),
    "dfa":           ("ssfa", {"schedule": "dense"},                [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]),
    "dfa_noprec":    ("ssfa", {"schedule": "dense", "precond": False}, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]),
    "ssfa_p5":       ("ssfa", {"P": 5},                             [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]),
    "ssfa_p5_noprec":("ssfa", {"P": 5, "precond": False},           [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]),
}

def specs(name, lr, seeds=(0, 1, 2)):
    m, kw, _ = GRID[name]
    return [dict(dataset="digits", val=True, method=m, kw={**kw, "lr": lr}, seed=s, epochs=40) for s in seeds]

if __name__ == "__main__":
    best = {}
    for name, (m, kw, lrs) in GRID.items():
        rows = []
        for lr in lrs:
            a = final_acc(run(specs(name, lr)), last=5) * 100
            rows.append((a.mean(), lr))
            print(f"{name:15s} lr={lr:<6} val={a.mean():.2f} +/- {a.std():.2f}", flush=True)
        best[name] = max(rows)[1]
        if best[name] in (lrs[0], lrs[-1]):
            print(f"  WARNING: best lr for {name} at grid edge")
    print(best)
    os.makedirs("results", exist_ok=True)
    json.dump(best, open("results/best_lr.json", "w"), indent=1)
