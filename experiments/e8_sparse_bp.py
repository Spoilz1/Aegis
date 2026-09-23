"""E8: sparse backprop (exact gradients, ceil(L/P) layers written per step).
Stage 1: LR selection on the digits validation split (3 seeds x 40 epochs).
Stage 2: test set, 5 seeds x 40 epochs, at the selected LR, vs E2 baselines.
Pre-registered prediction (docs/theory_sparse_updates.md section 7): best LR
0.9-1.5 for stagger P=5; test accuracy within noise of full BP (96.94)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ssfa.runner import run, final_acc

LRS = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
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

    # stage 2: test set, plus the untuned lr=0.3 (BP's) to isolate the lr-gain effect
    from ssfa.runner import summarize
    tst = lambda sch, lr: [dict(dataset="digits", method="sbp", kw={"lr": lr, "P": 5, "schedule": sch},
                                seed=s, epochs=40) for s in range(5)]
    cfgs = [(sch, best[sch]) for sch in SCHED] + [("stagger", 0.3)]
    run([s for c in cfgs for s in tst(*c)], procs=procs)
    out = {}
    for sch, lr in cfgs:
        res = run(tst(sch, lr))
        mu, sd = summarize(res)
        H = [r["hist"] for r in res]
        per_ep = np.mean([[h[3] for h in hh] for hh in H], 0) * 100
        out[f"{sch} lr={lr}"] = dict(acc=mu, std=sd, writes=H[0][-1][4], per_epoch=per_ep.tolist(),
                                     bwd_layers_per_step=res[0]["bwd_layers_per_step"])
        print(f"TEST sbp {sch:9s} lr={lr:<4} {mu:.2f} +/- {sd:.2f}  writes={H[0][-1][4]}  "
              f"bwd layers/step={res[0]['bwd_layers_per_step']:.2f}  acc@ep1,2,5,10,20: "
              + ", ".join(f"{per_ep[i]:.1f}" for i in [0, 1, 4, 9, 19]), flush=True)
    json.dump(out, open("results/e8_sparse_bp.json", "w"), indent=1)
