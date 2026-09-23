"""E4: memory accounting. (a) where the spec's 26.4% comes from; (b) full
per-step memory for SSFA vs. backprop variants at B=32; (c) how the total
saving depends on batch size."""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ssfa import memory as M
from ssfa.core import make_schedule

D = [64] + [256] * 6 + [10]
DM = [784] + [256] * 6 + [10]
L = 7
out = {}

# (a) spec formula: c_l = B (d_{l-1} + d_l), BP = sum_l c_l (each hidden activation counted twice)
c = [D[l - 1] + D[l] for l in range(1, L + 1)]
worst = max(c[a - 1] + c[b - 1] for a, b in [(1, 6), (2, 7)])
spec_ratio = worst / sum(c)
bp = M.bp(D, 32)
ss = M.ssfa_worst(D, 32, make_schedule("stagger", L, 5), 5)
out["a"] = dict(spec_formula=spec_ratio, distinct_retained=ss["act_retained"] / bp["act_retained"],
                peak_live=ss["act_peak"] / bp["act_peak"], total=M.total(ss) / M.total(bp))
print("(a) spec formula %.3f | distinct-tensor retained %.3f | peak live %.3f | total incl. weights %.3f" % tuple(out["a"].values()))

# (b) table at B=32
def row(name, r, ref):
    return dict(name=name, retained=r["act_retained"] / ref["act_retained"], act_peak=r["act_peak"] / ref["act_peak"],
                total=M.total(r) / M.total(ref), flops=r.get("flops_mean", r["flops"]) / ref["flops"],
                act_peak_floats=r["act_peak"], total_floats=M.total(r))
rows = [row("bp", bp, bp)]
for res, keep in M.checkpoint_pareto(D, 32):
    if keep != tuple(range(L)):
        rows.append(row(f"bp ckpt keep={list(keep)}", res, bp))
rows.append(row("bp batch 8", M.bp(D, 8), bp))
rows.append(row("bp accum 5 (m=4 microbatches)", M.bp(D, 32, microbatches=4), bp))
rows.append(row("bp width 128", M.bp([64] + [128] * 6 + [10], 32), bp))
rows.append(row("bp width 64", M.bp([64] + [64] * 6 + [10], 32), bp))
for P in [2, 3, 5, 7]:
    rows.append(row(f"ssfa P{P}", M.ssfa_worst(D, 32, make_schedule("stagger", L, P), P), bp))
    rows.append(row(f"ssfa P{P} (Omega from seed)", M.ssfa_worst(D, 32, make_schedule("stagger", L, P), P, count_omega=False), bp))
rows.append(row("ssfa P5 blocked", M.ssfa_worst(D, 32, make_schedule("blocked", L, 5), 5), bp))
out["b"] = rows
print("\n(b) B=32, digits net; ratios vs plain backprop")
print(f"{'method':38s} {'retained':>8s} {'actpeak':>8s} {'total':>8s} {'flops':>6s}")
for r in rows:
    print(f"{r['name']:38s} {r['retained']:8.3f} {r['act_peak']:8.3f} {r['total']:8.3f} {r['flops']:6.2f}")

# (c) total-memory ratio vs batch size
print("\n(c) total memory SSFA(P=5)/BP and best-checkpoint/BP vs batch size")
out["c"] = []
for name, dims in [("digits", D), ("mnist", DM)]:
    for B in [1, 8, 32, 128, 512, 2048]:
        b = M.bp(dims, B)
        s = M.ssfa_worst(dims, B, make_schedule("stagger", L, 5), 5)
        ck = min(M.checkpoint_pareto(dims, B), key=lambda x: M.total(x[0]))[0]
        frac = b["act_peak"] / M.total(b)
        rec = dict(net=name, B=B, act_share_bp=frac, ssfa=M.total(s) / M.total(b), ckpt=M.total(ck) / M.total(b),
                   ckpt_flops=ck["flops"] / b["flops"])
        out["c"].append(rec)
        print(f"{name:6s} B={B:5d}  activations are {frac:5.1%} of BP total | SSFA {rec['ssfa']:.3f} | best ckpt {rec['ckpt']:.3f} (flops x{rec['ckpt_flops']:.2f})")
json.dump(out, open("results/e4_memory.json", "w"), indent=1)
