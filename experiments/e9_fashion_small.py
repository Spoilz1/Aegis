"""E9: small Fashion-MNIST (5k train / 2k test, 14x14). Full BP vs
write-matched BP vs SSFA vs exact-gradient sparse BP (P=5, stagger).
Stage 1: LR on validation split (3 seeds x 20 epochs). Stage 2: test, 5 seeds."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ssfa.runner import run, final_acc, summarize
import numpy as np

DS, EP = "fashion_small", 20
M = {"bp": ("bp", {}), "bp accum5": ("bp", {"accum": 5}),
     "ssfa P5": ("ssfa", {"P": 5, "precond": False}), "sparse bp P5": ("sbp", {"P": 5})}
LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 0.6]
sp = lambda m, kw, lr, val, seeds: [dict(dataset=DS, val=val, method=m, kw={**kw, "lr": lr}, seed=s, epochs=EP) for s in seeds]

if __name__ == "__main__":
    run([s for m, kw in M.values() for lr in LRS for s in sp(m, kw, lr, True, range(3))])
    best = {}
    for n, (m, kw) in M.items():
        rows = [(final_acc(run(sp(m, kw, lr, True, range(3))), last=3).mean() * 100, lr) for lr in LRS]
        best[n] = max(rows)[1]
        print(f"val {n:13s} " + "  ".join(f"lr={lr}:{a:.2f}" for a, lr in rows) + ("  EDGE" if best[n] in (LRS[0], LRS[-1]) else ""), flush=True)
    run([s for n, (m, kw) in M.items() for s in sp(m, kw, best[n], False, range(5))])
    out = {}
    for n, (m, kw) in M.items():
        res = run(sp(m, kw, best[n], False, range(5)))
        mu, sd = summarize(res)
        pe = np.mean([[h[3] for h in r["hist"]] for r in res], 0) * 100
        w = res[0]["hist"][0][4]
        out[n] = dict(lr=best[n], acc=mu, std=sd, per_epoch=pe.tolist(), writes_per_epoch=w)
        print(f"TEST {n:13s} lr={best[n]:<5} {mu:.2f} +/- {sd:.2f}  writes/epoch={w:5d}  acc@ep1,2,5,10,20: "
              + ", ".join(f"{pe[i]:.1f}" for i in [0, 1, 4, 9, 19]), flush=True)
    json.dump(out, open("results/e9_fashion_small.json", "w"), indent=1)
