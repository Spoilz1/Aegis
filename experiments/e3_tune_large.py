"""E3: learning-rate selection on MNIST / Fashion-MNIST validation splits
(1 seed, 8 epochs; selection by final val accuracy)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ssfa.runner import run, final_acc

GRID = {
    "bp":        ("bp",   {},                                   [0.03, 0.1, 0.3]),
    "bp_accum5": ("bp",   {"accum": 5},                         [0.1, 0.3, 1.0]),
    "dfa":       ("ssfa", {"schedule": "dense", "precond": False}, [0.1, 0.2, 0.5, 1.0]),
    "ssfa_p5":   ("ssfa", {"P": 5, "precond": False},           [0.2, 0.5, 1.0, 2.0]),
}
if __name__ == "__main__":
    best = {}
    for ds in ["mnist", "fashion"]:
        allspecs = {(n, lr): [dict(dataset=ds, val=True, method=m, kw={**kw, "lr": lr}, seed=0, epochs=8)]
                    for n, (m, kw, lrs) in GRID.items() for lr in lrs}
        run([s for v in allspecs.values() for s in v])
        for n, (m, kw, lrs) in GRID.items():
            rows = [(final_acc(run(allspecs[(n, lr)]))[0] * 100, lr) for lr in lrs]
            print(ds, n, [(lr, round(a, 2)) for a, lr in rows], flush=True)
            best[f"{ds}/{n}"] = max(rows)[1]
    print(best)
    json.dump(best, open("results/best_lr_large.json", "w"), indent=1)
