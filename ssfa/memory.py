"""Structural peak-memory model (in floats) for one training step.

Memory depends only on shapes and the schedule, never on values, so it is
computed by replaying each method's allocate/free sequence. Components:

  params      all W_l, b_l                                   (every method)
  state       method-specific persistent state (SSFA: s_l; Omega_l optionally,
              since it can be regenerated from a seed; accumulation: grad buffer)
  act_peak    peak of live activation / error tensors during the step
  upd_peak    largest transient weight-gradient tensor (one layer at a time,
              same assumption for every method)

`act_retained` is the spec's Proposition-2 quantity: activations held after
the forward pass, waiting for an update.
"""
import math


def n_params(dims):
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


class Live:
    def __init__(self):
        self.live, self.cur, self.peak = {}, 0, 0

    def alloc(self, k, n):
        assert k not in self.live, k
        self.live[k] = n
        self.cur += n
        self.peak = max(self.peak, self.cur)

    def free(self, k):
        self.cur -= self.live.pop(k)


def bp(dims, B, segments=0, microbatches=1, keep=None):
    """Backprop. segments=k>0: gradient checkpointing, keeping only k segment
    boundary activations and recomputing each segment during backward.
    microbatches=m>1: gradient accumulation over B/m-row microbatches (exact
    same update, but needs a parameter-sized gradient buffer).
    keep: explicit checkpoint set (activation indices, must contain 0)."""
    L = len(dims) - 1
    b = math.ceil(B / microbatches)
    if keep is not None:
        C = sorted(set(keep) | {0})
    else:
        C = sorted({(i * L) // segments for i in range(segments)}) if segments else list(range(L))
    m = Live()
    m.alloc(("A", 0), b * dims[0])
    for l in range(1, L + 1):
        m.alloc(("A", l), b * dims[l])
        if l - 1 not in C:
            m.free(("A", l - 1))
    retained = m.cur
    m.alloc(("d", L), b * dims[L])
    m.free(("A", L))
    edges, extra = C + [L], 0
    for si in range(len(edges) - 2, -1, -1):
        lo, hi = edges[si], edges[si + 1]
        for l in range(lo + 1, hi):                      # recompute segment interior
            m.alloc(("A", l), b * dims[l])
            extra += 2 * b * dims[l - 1] * dims[l]
        for l in range(hi, lo, -1):                      # backward through W_l
            if l - 1 > 0:
                m.alloc(("d", l - 1), b * dims[l - 1])
            m.free(("d", l))
            m.free(("A", l - 1))
    fwd = sum(2 * B * dims[i] * dims[i + 1] for i in range(L))
    return dict(params=n_params(dims), state=n_params(dims) if microbatches > 1 else 0,
                act_peak=m.peak, act_retained=retained,
                upd_peak=max(dims[i] * dims[i + 1] for i in range(L)),
                flops=3 * fwd + extra * microbatches, fwd_flops=fwd)


def ssfa(dims, B, S, count_omega=True):
    """One SSFA step with scheduled layer set S (1-based layer ids 1..L)."""
    L = len(dims) - 1
    S = set(S)
    keep = lambda i: (i + 1) in S or i in S   # A_i is input of layer i+1 / output of layer i
    m = Live()
    m.alloc(("A", 0), B * dims[0])
    for l in range(1, L + 1):
        m.alloc(("A", l), B * dims[l])
        if not keep(l - 1):
            m.free(("A", l - 1))
    retained = sum(v for (k, i), v in m.live.items() if keep(i))
    m.alloc("E", B * dims[L])
    if not keep(L):
        m.free(("A", L))
    for l in sorted(S):
        if l < L:
            m.alloc(("dZ", l), B * dims[l])   # (E Omega^T) * sigma'
            m.free(("dZ", l))
    act_peak = m.peak
    om = sum(dims[l] * dims[L] for l in range(1, L)) if count_omega else 0
    state = om + sum(dims[l - 1] for l in range(1, L + 1))
    upd = max(dims[l - 1] * dims[l] for l in S) if S else 0
    fwd = sum(2 * B * dims[i] * dims[i + 1] for i in range(L))
    upd_flops = sum(2 * B * dims[l - 1] * dims[l] + (2 * B * dims[l] * dims[L] if l < L else 0)
                    for l in S)
    return dict(params=n_params(dims), state=state, act_peak=act_peak,
                act_retained=retained, upd_peak=upd, flops=fwd + upd_flops, fwd_flops=fwd)


def ssfa_worst(dims, B, sched, P, horizon=None, **kw):
    """Worst case over one full period (schedules here are periodic in P or
    cycle through a permutation of residues, so P*L steps cover all cases)."""
    horizon = horizon or P * len(dims) * 4
    runs = [ssfa(dims, B, [l + 1 for l in sched(t)], **kw) for t in range(horizon)]
    worst = max(runs, key=lambda r: r["act_peak"])
    worst = dict(worst)
    worst["act_retained"] = max(r["act_retained"] for r in runs)
    worst["act_retained_mean"] = sum(r["act_retained"] for r in runs) / len(runs)
    worst["flops_mean"] = sum(r["flops"] for r in runs) / len(runs)
    return worst


def total(r):
    return r["params"] + r["state"] + r["act_peak"] + r["upd_peak"]


def checkpoint_pareto(dims, B):
    """All checkpoint plans (subsets of activations 1..L-1 to keep), reduced to
    the Pareto front of (act_peak, flops)."""
    import itertools
    L = len(dims) - 1
    plans = []
    for r in range(L):
        for c in itertools.combinations(range(1, L), r):
            plans.append((bp(dims, B, keep=(0,) + c), (0,) + c))
    plans.sort(key=lambda x: (x[0]["act_peak"], x[0]["flops"]))
    front, best = [], float("inf")
    for res, c in plans:
        if res["flops"] < best:
            front.append((res, c))
            best = res["flops"]
    return front
