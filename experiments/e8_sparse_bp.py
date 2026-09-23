"""E8: sparse backprop (exact gradients, ceil(L/P) layers written per step).
Stage 1: LR selection on the digits validation split (3 seeds x 40 epochs)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ssfa.runner import run, final_acc

LRS = [0.1, 0.3, 1.0, 2.0]
SCHED = ["stagger", "shuffled", "subset"]

if __name__ == "__main__":
    procs = int(os.environ.get("PROCS", 4))
    sp = lambda sch, lr, P=5: [dict(dataset="digits", val=True, method="sbp", kw={"lr": lr, "P": P, "schedule": sch},
                                    seed=s, epochs=40) for s in range(3)]
    run([s for sch in SCHED for lr in LRS for s in sp(sch, lr)], procs=procs)
    best = {}
    for sch in SCHED:
        rows = []
        for lr in LRS:
            a = final_acc(run(sp(sch, lr)), last=5) * 100
            rows.append((a.mean(), lr))
            print(f"sbp {sch:9s} lr={lr:<4} val={a.mean():.2f} +/- {a.std():.2f}", flush=True)
        best[sch] = max(rows)[1]
    print(best)
    json.dump(best, open("results/best_lr_sbp_digits.json", "w"), indent=1)
