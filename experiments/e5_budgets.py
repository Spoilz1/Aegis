"""E5: accuracy at matched wall-clock and matched weight-write budgets, from the
per-epoch histories of E2 (digits) and E6 (MNIST/Fashion, if present)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ssfa.runner import run

def acc_at(hist, budget, col):
    """Test accuracy at the last evaluation with hist[col] <= budget
    (col 2 = train seconds, col 4 = layer writes). NaN if none. No
    best-so-far selection, which would be early stopping on the test set."""
    ok = [h[3] for h in hist if h[col] <= budget]
    return ok[-1] if ok else np.nan

def first_reach(hist, target, col):
    for h in hist:
        if h[3] >= target:
            return h[col]
    return np.nan

def table(ds, configs, specs_fn, time_budgets, write_budgets, targets):
    out = {}
    for name, (m, kw) in configs.items():
        res = run(specs_fn(ds, m, kw))
        H = [r["hist"] for r in res]
        out[name] = dict(
            time={b: np.nanmean([acc_at(h, b, 2) for h in H]) * 100 for b in time_budgets},
            writes={b: np.nanmean([acc_at(h, b, 4) for h in H]) * 100 for b in write_budgets},
            reach={t: [first_reach(h, t / 100, 2) for h in H] for t in targets},
            reach_writes={t: [first_reach(h, t / 100, 4) for h in H] for t in targets})
    print(f"\n== {ds}: test accuracy at the last eval within a TRAIN-TIME budget (s) ==")
    print(f"{'method':28s}" + "".join(f"{b:>9}" for b in time_budgets))
    for n, o in out.items():
        print(f"{n:28s}" + "".join(f"{v:9.2f}" for v in o["time"].values()))
    print(f"\n== {ds}: test accuracy at the last eval within a WEIGHT-WRITE budget (layer-matrix writes) ==")
    print(f"{'method':28s}" + "".join(f"{b:>9}" for b in write_budgets))
    for n, o in out.items():
        print(f"{n:28s}" + "".join(f"{v:9.2f}" for v in o["writes"].values()))
    print(f"\n== {ds}: median seconds / layer-writes to first reach target (n reached/seeds) ==")
    for n, o in out.items():
        cells = []
        for t in targets:
            r, w = np.array(o["reach"][t]), np.array(o["reach_writes"][t])
            k = np.isfinite(r).sum()
            cells.append(f"{t}%: {np.nanmedian(r) if k else float('nan'):6.2f}s {np.nanmedian(w) if k else float('nan'):8.0f}w ({k}/{len(r)})")
        print(f"{n:28s} " + " | ".join(cells))
    return out

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from e2_digits import CONFIGS as C2, SEEDS
    pick = ["bp", "bp accum 5 (write-matched)", "bp width 64", "dfa", "dfa noprec", "ssfa P5", "ssfa P5 noprec", "sched shuffled", "ssfa P2"]
    cfg = {k: C2[k] for k in pick}
    f2 = lambda ds, m, kw: [dict(dataset="digits", method=m, kw=kw, seed=s, epochs=40, align=m == "ssfa") for s in SEEDS]
    res = {"digits": table("digits", cfg, f2, [0.25, 0.5, 1, 2, 4], [500, 1000, 2000, 5000, 12000], [95.5, 96.0, 96.5])}
    if os.path.exists("results/e6_large.json"):
        from e6_large import CONFIGS as C6, spec6
        for ds in ["mnist", "fashion"]:
            res[ds] = table(ds, C6(ds), spec6, [10, 20, 40, 80, 160], [5000, 10000, 20000, 50000, 100000],
                            [97.0, 97.5, 98.0] if ds == "mnist" else [87.0, 88.0, 89.0])
    json.dump(res, open("results/e5_budgets.json", "w"), indent=1, default=float)
