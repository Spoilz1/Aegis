"""The structural memory model must agree with what the real SSFA step caches,
and the schedule propositions must hold."""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ssfa import memory as M
from ssfa.core import SSFA, make_schedule

DIMS = [64] + [256] * 6 + [10]


def test_cache_matches_model():
    rng = np.random.RandomState(0)
    X, Y = rng.randn(32, 64), np.eye(10)[rng.randint(0, 10, 32)]
    for P in [2, 3, 5, 7]:
        net = SSFA(DIMS, 0, P=P)
        for t in range(2 * P):
            S = net.sched(net.t)
            net.step(X, Y)
            net.t += 1
            model = M.ssfa(DIMS, 32, [l + 1 for l in S])["act_retained"]
            assert net.last_cache_floats == model, (P, t)


def test_proposition_1():
    L = 7
    for kind in ["stagger", "ascending", "blocked", "shuffled"]:
        for P in [1, 2, 3, 5, 7]:
            f = make_schedule(kind, L, P, np.random.RandomState(0))
            counts = np.zeros(L)
            for t in range(P * 20):
                S = f(t)
                assert len(S) <= math.ceil(L / P)
                counts[S] += 1
            assert (counts == 20).all(), (kind, P, counts)


def test_bp_checkpoint_default_is_plain_bp():
    a = M.bp(DIMS, 32)
    b = M.bp(DIMS, 32, keep=tuple(range(7)))
    assert a["act_peak"] == b["act_peak"] and a["flops"] == b["flops"]


if __name__ == "__main__":
    test_cache_matches_model(); test_proposition_1(); test_bp_checkpoint_default_is_plain_bp()
    print("ok")
