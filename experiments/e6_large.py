"""E6: MNIST and Fashion-MNIST, 784-256x6-10, 3 seeds x 20 epochs, learning
rates from E3 (validation). Preconditioner-on runs reuse the no-precond lr."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ssfa.runner import run, summarize

SEEDS = range(3)
EPOCHS = 20

def CONFIGS(ds):
    lr = json.load(open("results/best_lr_large.json"))
    g = lambda n: lr[f"{ds}/{n}"]
    s = g("ssfa_p5")
    return {
        "bp":                         ("bp",   {"lr": g("bp")}),
        "bp accum 5 (write-matched)": ("bp",   {"lr": g("bp_accum5"), "accum": 5}),
        "bp width 64":                ("bp",   {"lr": g("bp"), "width": 64}),
        "dfa":                        ("ssfa", {"lr": g("dfa"), "schedule": "dense", "precond": False}),
        "ssfa P5":                    ("ssfa", {"lr": s, "P": 5, "precond": False}),
        "ssfa P5 +precond":           ("ssfa", {"lr": s, "P": 5}),
        "ssfa P5 shuffled":           ("ssfa", {"lr": s, "P": 5, "precond": False, "schedule": "shuffled"}),
        "ssfa P2":                    ("ssfa", {"lr": s, "P": 2, "precond": False}),
        "ssfa P7":                    ("ssfa", {"lr": s, "P": 7, "precond": False}),
    }

def spec6(ds, m, kw):
    return [dict(dataset=ds, method=m, kw=kw, seed=s, epochs=EPOCHS, align=m == "ssfa") for s in SEEDS]

if __name__ == "__main__":
    out = {}
    for ds in ["mnist", "fashion"]:
        C = CONFIGS(ds)
        run([s for m, kw in C.values() for s in spec6(ds, m, kw)])
        for name, (m, kw) in C.items():
            res = run(spec6(ds, m, kw))
            mu, sd = summarize(res)
            t = np.mean([r["hist"][-1][2] for r in res])
            al = np.mean([[c for _, c in r["align"]] for r in res], 0).round(3).tolist() if "align" in res[0] else None
            out[f"{ds}/{name}"] = dict(acc=mu, std=sd, time=t, writes=res[0]["hist"][-1][4], align=al, kw=kw)
            print(f"{ds:7s} {name:28s} {mu:6.2f} +/- {sd:.2f}  t={t:6.1f}s  writes={res[0]['hist'][-1][4]}  align={al}", flush=True)
    json.dump(out, open("results/e6_large.json", "w"), indent=1)
