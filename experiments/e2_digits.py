"""E2: test-set comparison on digits, 5 seeds x 40 epochs, learning rates from E1
(plus the spec's untuned settings for reproduction). Also schedule-pattern and
period ablations for SSFA."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ssfa.runner import run, summarize
import numpy as np

SEEDS = range(5)
CONFIGS = {
    # reproduction of spec settings
    "bp (spec lr=0.1)":            ("bp",   {"lr": 0.1}),
    "ssfa P5 (spec eta=0.05)":     ("ssfa", {"lr": 0.05, "P": 5}),
    "ssfa P5 noprec (eta=0.05)":   ("ssfa", {"lr": 0.05, "P": 5, "precond": False}),
    "dfa (eta=0.05)":              ("ssfa", {"lr": 0.05, "schedule": "dense"}),
    # tuned
    "bp":                          ("bp",   {"lr": 0.3}),
    "bp batch 8":                  ("bp",   {"lr": 0.1, "batch": 8}),
    "bp accum 5 (write-matched)":  ("bp",   {"lr": 0.3, "accum": 5}),
    "bp width 128":                ("bp",   {"lr": 0.3, "width": 128}),
    "bp width 64":                 ("bp",   {"lr": 0.3, "width": 64}),
    "dfa":                         ("ssfa", {"lr": 0.5, "schedule": "dense"}),
    "dfa noprec":                  ("ssfa", {"lr": 0.5, "schedule": "dense", "precond": False}),
    "ssfa P5":                     ("ssfa", {"lr": 0.5, "P": 5}),
    "ssfa P5 noprec":              ("ssfa", {"lr": 0.5, "P": 5, "precond": False}),
    # schedule patterns (P=5, lr=0.5)
    "sched ascending":             ("ssfa", {"lr": 0.5, "P": 5, "schedule": "ascending"}),
    "sched blocked":               ("ssfa", {"lr": 0.5, "P": 5, "schedule": "blocked"}),
    "sched shuffled":              ("ssfa", {"lr": 0.5, "P": 5, "schedule": "shuffled"}),
    "sched stagger+out":           ("ssfa", {"lr": 0.5, "P": 5, "schedule": "stagger+out"}),
    "sched bernoulli":             ("ssfa", {"lr": 0.5, "P": 5, "schedule": "bernoulli"}),
    # period sweep (stagger, lr=0.5)
    "ssfa P2":                     ("ssfa", {"lr": 0.5, "P": 2}),
    "ssfa P3":                     ("ssfa", {"lr": 0.5, "P": 3}),
    "ssfa P7":                     ("ssfa", {"lr": 0.5, "P": 7}),
}

def specs(m, kw):
    return [dict(dataset="digits", method=m, kw=kw, seed=s, epochs=40, align=m == "ssfa") for s in SEEDS]

if __name__ == "__main__":
    allspecs = [s for m, kw in CONFIGS.values() for s in specs(m, kw)]
    run(allspecs)
    out = {}
    for name, (m, kw) in CONFIGS.items():
        res = run(specs(m, kw))
        mu, sd = summarize(res)
        mu5, sd5 = summarize(res, last=5)
        t = np.mean([r["hist"][-1][2] for r in res])
        al = None
        if "align" in res[0]:
            al = np.mean([[c for _, c in r["align"]] for r in res], 0).round(3).tolist()
        out[name] = dict(acc=mu, std=sd, acc_last5=mu5, std_last5=sd5, time=t,
                         writes=res[0]["hist"][-1][4], align=al)
        print(f"{name:30s} {mu:6.2f} +/- {sd:.2f}  (last5 {mu5:.2f} +/- {sd5:.2f})  t={t:5.2f}s  writes={out[name]['writes']}  align={al}", flush=True)
    json.dump(out, open("results/e2_digits.json", "w"), indent=1)
