"""Core implementation: MLP, backprop, (sparse) direct feedback alignment, SSFA.

Row-major convention: activations are (B, d); W_l is (d_l, d_{l-1}).
This is the transpose of the column convention in the spec; the math is identical.
"""
import time
import numpy as np


# ---------------------------------------------------------------- utilities

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def semi_orthogonal(rows, cols, rng):
    """rows x cols matrix with orthonormal columns (rows >= cols)."""
    q, r = np.linalg.qr(rng.randn(rows, cols))
    return q * np.sign(np.diag(r))


def init_net(dims, rng):
    Ws = [rng.randn(dims[i + 1], dims[i]) / np.sqrt(dims[i]) for i in range(len(dims) - 1)]
    bs = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
    return Ws, bs


def predict(Ws, bs, X):
    a = X
    L = len(Ws)
    for l in range(L):
        z = a @ Ws[l].T + bs[l]
        a = z if l == L - 1 else np.tanh(z)
    return a.argmax(1)


# ---------------------------------------------------------------- schedules

def make_schedule(kind, L, P, rng=None):
    """Return f(t) -> sorted list of layer indices (0-based) updated at step t."""
    idx = list(range(L))
    if kind == "dense":
        return lambda t: idx
    if kind == "stagger":          # spec: (t + l) mod P == 0  -> fires top-down
        return lambda t: [l for l in idx if (t + l) % P == 0]
    if kind == "ascending":        # (t - l) mod P == 0        -> fires bottom-up
        return lambda t: [l for l in idx if (t - l) % P == 0]
    if kind == "blocked":          # contiguous groups, one group per step
        groups = [list(g) for g in np.array_split(np.arange(L), P)]
        return lambda t: groups[t % P]
    if kind == "shuffled":         # stagger residues, random order each cycle
        state = {"cycle": -1, "perm": None}
        def f(t):
            c = t // P
            if c != state["cycle"]:
                state["cycle"], state["perm"] = c, rng.permutation(P)
            r = state["perm"][t % P]
            return [l for l in idx if l % P == r]
        return f
    if kind == "stagger+out":      # stagger on hidden layers, output layer every step
        def f(t):
            s = [l for l in idx[:-1] if (t + l) % P == 0]
            return s + [L - 1]
        return f
    if kind == "subset":           # uniform random ceil(L/P)-subset each step (LISA-style)
        k = -(-L // P)
        return lambda t: sorted(rng.choice(L, k, replace=False).tolist())
    if kind == "bernoulli":        # stochastic, rate 1/P, no hard bound
        return lambda t: [l for l in idx if rng.rand() < 1.0 / P]
    raise ValueError(kind)


# ---------------------------------------------------------------- trainers

class Trainer:
    """Common loop: epochs of shuffled minibatches, eval after each epoch."""

    def __init__(self, dims, seed, batch=32):
        self.rng = np.random.RandomState(seed)
        self.dims = dims
        self.L = len(dims) - 1
        self.Ws, self.bs = init_net(dims, self.rng)
        self.B = batch
        self.t = 0
        self.writes = 0          # number of (layer) weight-matrix writes

    def fit(self, data, epochs, eval_every_steps=None, time_budget=None):
        Xtr, Ytr, Xte, yte = data
        n = len(Xtr)
        hist = []  # (epoch, step, train_time_s, test_acc, layer_writes)
        train_time = 0.0
        for ep in range(epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n - self.B + 1, self.B):
                j = perm[i:i + self.B]
                t0 = time.perf_counter()
                self.step(Xtr[j], Ytr[j])
                train_time += time.perf_counter() - t0
                self.t += 1
                if eval_every_steps and self.t % eval_every_steps == 0:
                    acc = (predict(self.Ws, self.bs, Xte) == yte).mean()
                    hist.append((ep + 1, self.t, train_time, acc, self.writes))
            if not eval_every_steps:
                acc = (predict(self.Ws, self.bs, Xte) == yte).mean()
                hist.append((ep + 1, self.t, train_time, acc, self.writes))
            if time_budget and train_time > time_budget:
                break
        return hist


class BP(Trainer):
    """Plain SGD backprop, softmax cross-entropy. Layer updates are applied
    during the backward sweep (no full gradient buffer).
    accum > 1: gradients summed over `accum` consecutive minibatches and
    applied once (large-batch / write-matched baseline)."""

    def __init__(self, dims, seed, batch=32, lr=0.1, accum=1):
        super().__init__(dims, seed, batch)
        self.lr, self.accum = lr, accum
        if accum > 1:
            self.gW = [np.zeros_like(W) for W in self.Ws]
            self.gb = [np.zeros_like(b) for b in self.bs]

    def step(self, X, Y):
        Ws, bs, L = self.Ws, self.bs, self.L
        A = [X]
        for l in range(L):
            z = A[-1] @ Ws[l].T + bs[l]
            A.append(softmax(z) if l == L - 1 else np.tanh(z))
        B = X.shape[0]
        delta = (A[-1] - Y) / B
        apply = self.accum == 1 or (self.t + 1) % self.accum == 0
        for l in range(L - 1, -1, -1):
            gW, gb = delta.T @ A[l], delta.sum(0)
            if l > 0:
                delta = (delta @ Ws[l]) * (1 - A[l] ** 2)
            if self.accum == 1:
                Ws[l] -= self.lr * gW
                bs[l] -= self.lr * gb
                self.writes += 1
            else:
                self.gW[l] += gW
                self.gb[l] += gb
                if apply:
                    Ws[l] -= self.lr * self.gW[l] / self.accum
                    bs[l] -= self.lr * self.gb[l] / self.accum
                    self.gW[l][:] = 0
                    self.gb[l][:] = 0
                    self.writes += 1


class SSFA(Trainer):
    """Sparse direct feedback alignment with a pre-committed layer schedule.

    schedule='dense' -> ordinary DFA. precond toggles the diagonal input
    preconditioner. Only scheduled layers retain activations (cache)."""

    def __init__(self, dims, seed, batch=32, lr=0.05, P=5, schedule="stagger",
                 precond=True, beta=0.05, eps=1e-6):
        super().__init__(dims, seed, batch)
        self.lr, self.P, self.precond, self.beta, self.eps = lr, P, precond, beta, eps
        d_out = dims[-1]
        self.Om = [semi_orthogonal(dims[l + 1], d_out, self.rng) for l in range(self.L - 1)]
        self.s = [np.ones(dims[l]) for l in range(self.L)]
        self.sched = make_schedule(schedule, self.L, P, np.random.RandomState(seed + 10_000))
        self.last_cache_floats = 0

    def step(self, X, Y):
        Ws, bs, L, beta = self.Ws, self.bs, self.L, self.beta
        S = self.sched(self.t)
        Sset = set(S)
        cache = {}
        a = X
        for l in range(L):
            if self.precond:
                self.s[l] = (1 - beta) * self.s[l] + beta * (a ** 2).mean(0)
            z = a @ Ws[l].T + bs[l]
            a_next = softmax(z) if l == L - 1 else np.tanh(z)
            if l in Sset:
                cache[l] = (a, a_next)
            a = a_next
        self.last_cache_floats = sum(u.size + v.size for u, v in cache.values())
        E = Y - a
        B = X.shape[0]
        for l in S:
            a_prev, a_out = cache[l]
            dZ = E if l == L - 1 else (E @ self.Om[l].T) * (1 - a_out ** 2)
            gW = dZ.T @ a_prev
            if self.precond:  # A~ = A / (sqrt(s)+eps)  ==  column-scale of dZ^T A
                gW /= (np.sqrt(self.s[l]) + self.eps)
            Ws[l] += self.lr / B * gW
            bs[l] += self.lr / B * dZ.sum(0)
            self.writes += 1

    def alignment(self, X, Y):
        """Cosine similarity between each hidden layer's SSFA update direction
        (unpreconditioned) and the true negative gradient, on one batch."""
        Ws, bs, L = self.Ws, self.bs, self.L
        A = [X]
        for l in range(L):
            z = A[-1] @ Ws[l].T + bs[l]
            A.append(softmax(z) if l == L - 1 else np.tanh(z))
        E = Y - A[-1]
        out, delta = [], -E
        for l in range(L - 1, -1, -1):
            g_true = -(delta.T @ A[l])
            if l < L - 1:
                dZ = (E @ self.Om[l].T) * (1 - A[l + 1] ** 2)
                g_fa = dZ.T @ A[l]
                c = (g_true * g_fa).sum() / (np.linalg.norm(g_true) * np.linalg.norm(g_fa) + 1e-12)
                out.append((l, c))
            if l > 0:
                delta = (delta @ Ws[l]) * (1 - A[l] ** 2)
        return sorted(out)


class SparseBP(Trainer):
    """Exact-gradient sparse updates: only layers in S_t are written; the
    backward sweep stops at min(S_t), so activations below the lowest
    scheduled layer are discarded during the forward pass (the schedule is
    known in advance). Plain SGD."""

    def __init__(self, dims, seed, batch=32, lr=0.1, P=5, schedule="stagger"):
        super().__init__(dims, seed, batch)
        self.lr, self.P = lr, P
        self.sched = make_schedule(schedule, self.L, P, np.random.RandomState(seed + 10_000))
        self.bwd_layers = 0      # layers traversed by backward sweeps (compute proxy)

    def step(self, X, Y):
        Ws, bs, L = self.Ws, self.bs, self.L
        S = self.sched(self.t)
        lo = min(S)
        A = {}
        a = X
        for l in range(L):
            if l >= lo:
                A[l] = a                      # input of layer l, needed for l >= lo
            z = a @ Ws[l].T + bs[l]
            a = softmax(z) if l == L - 1 else np.tanh(z)
        delta = (a - Y) / X.shape[0]
        Sset = set(S)
        for l in range(L - 1, lo - 1, -1):
            if l in Sset:
                gW, gb = delta.T @ A[l], delta.sum(0)
            if l > lo:
                delta = (delta @ Ws[l]) * (1 - A[l] ** 2)
            if l in Sset:
                Ws[l] -= self.lr * gW
                bs[l] -= self.lr * gb
                self.writes += 1
        self.bwd_layers += L - lo
