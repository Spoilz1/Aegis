"""Datasets: sklearn digits (1,797 8x8 images), MNIST, Fashion-MNIST.

All returned as (Xtr, Ytr_onehot, Xte, yte) float64, standardized with
training-set statistics. The digits split is fixed (random_state=0) so that
seeds vary only initialization and minibatch order.
"""
import gzip
import os
import urllib.request

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "..", ".data")

URLS = {
    "mnist": "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "fashion": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
}
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def _onehot(y, k=10):
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1
    return Y


def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    return (Xtr - mu) / sd, (Xte - mu) / sd


def digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    Xtr, Xte = _standardize(Xtr, Xte)
    return Xtr, _onehot(ytr), Xte, yte


def _idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def idx_dataset(name):
    d = os.path.join(CACHE, name)
    os.makedirs(d, exist_ok=True)
    arrs = []
    for f in FILES:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(URLS[name] + f, p)
        arrs.append(_idx(p))
    Xtr, ytr, Xte, yte = arrs
    Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float64) / 255
    Xte = Xte.reshape(len(Xte), -1).astype(np.float64) / 255
    Xtr, Xte = _standardize(Xtr, Xte)
    return Xtr, _onehot(ytr.astype(int)), Xte, yte.astype(int)


def small(name, n_train=5000, n_test=2000):
    """Small, fast variant of an idx dataset: fixed random subset, 2x2
    average-pooled to 14x14 (196 inputs), standardized on the subset."""
    d = os.path.join(CACHE, name)
    idx_dataset(name)  # ensure downloaded
    Xtr, ytr, Xte, yte = [_idx(os.path.join(d, f)) for f in FILES]
    r = np.random.RandomState(0)
    tr, te = r.permutation(len(Xtr))[:n_train], r.permutation(len(Xte))[:n_test]
    pool = lambda X: X.reshape(len(X), 14, 2, 14, 2).mean((2, 4)).reshape(len(X), -1) / 255
    Xtr, Xte = _standardize(pool(Xtr[tr].astype(np.float64)), pool(Xte[te].astype(np.float64)))
    return Xtr, _onehot(ytr[tr].astype(int)), Xte, yte[te].astype(int)


def load(name, val=False):
    """val=True: return (train minus 20% holdout, holdout) for hyperparameter
    selection; the test set is never used for tuning."""
    if name == "digits":
        Xtr, Ytr, Xte, yte = digits()
    elif name.endswith("_small"):
        Xtr, Ytr, Xte, yte = small(name[:-6])
    else:
        Xtr, Ytr, Xte, yte = idx_dataset(name)
    if not val:
        return Xtr, Ytr, Xte, yte
    idx = np.random.RandomState(123).permutation(len(Xtr))
    k = len(Xtr) // 5
    va, tr = idx[:k], idx[k:]
    return Xtr[tr], Ytr[tr], Xtr[va], Ytr[va].argmax(1)
