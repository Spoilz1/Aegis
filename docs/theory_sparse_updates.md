# When can you train by updating only some layers per step?

A first-order theory of exact-gradient sparse layer updates, checked on hand
examples and one measurement on the 7-layer digits network. Everything below is
a prediction until the pre-registered test at the end has been run.

## 1. Model

Linearize the network around its current weights. Stack every training output
into a residual vector r (predictions minus targets). A gradient step of size
eta on a set S of layers changes the residual by

    r  <-  (I - eta * K_S) r,        K_S = sum_{l in S} K_l,        K_l = J_l J_l^T

where J_l is the Jacobian of all outputs with respect to layer l's parameters.
K_l is layer l's *tangent kernel*. Each K_l is positive semi-definite, and the
full network's kernel is K = sum_l K_l. Full backprop is the case S = all layers.

Two facts follow directly.

- **Stability.** A step is stable only if eta < 2 / lambda_max(K_S).
- **Progress.** Along an eigendirection v of K, one step removes the fraction
  eta * v^T K_S v of the residual.

## 2. Results

**R1: at equal learning rate, sparse and full training make the same progress
per weight write.** Take a cyclic schedule that writes every layer once per
cycle. For small eta, the product of the cycle's steps is
prod (I - eta K_{S_t}) = I - eta K + O(eta^2). That is one full backprop step, and
it uses the same number of layer writes. The cost is that sparse training needs
L/|S| times as many forward passes.

**R2: sparse steps can use a larger learning rate.** Since K ⪰ K_S, the largest
stable learning rate grows by the factor

    g_S = lambda_max(K) / lambda_max(K_S),    with 1 <= g_S <= L/|S|.

The upper limit is reached when all layers are *coherent*: each layer's kernel
points the same way as the total, K_l ≈ K/L. The lower limit is reached when the
layers are orthogonal, each fitting different directions of the residual.

**R3: in the coherent regime, sparsity costs nothing per step.** Scale the
learning rate by g_S ≈ L/|S|. A step that writes |S| of the L layers then removes
the same residual as a full backprop step. Writes per step drop by the factor
|S|/L.

**R4: in the orthogonal regime, sparsity costs sample efficiency.** Writes per
unit of progress stay the same, but reaching a given loss takes L/|S| times as
many forward passes. This is the likely cause of the widening gap on
MNIST/Fashion-MNIST, where each layer effectively sees only 1/P of the data.

## 3. Hand checks

**Coherent case.** Take the scalar chain o = w1*w2*x with x = 1, y = 1.2,
starting at w = (1, 1).

- The residual is r = -0.2 and both gradients are -0.2.
- Both layers' Jacobians are (1, 1), so K_1 = K_2 = 1 and K = 2. The gain is
  g = 2 = L/|S|, the maximum.
- Full GD at eta = 1/2 moves to w = (1.1, 1.1). The new residual is 0.01,
  the loss falls from 0.02 to 5e-5, and it takes **2 writes**.
- Updating only w1, at eta = 1, moves to w = (1.2, 1). The residual is
  **exactly 0**, after **1 write**.
- Full GD at eta = 1 overshoots to a residual of 0.24. So the doubled
  learning rate is safe for one layer and unsafe for both, as R2 predicts.

All three numbers were confirmed by direct computation.

**Orthogonal case.** Suppose K_1 = diag(1, 0) and K_2 = diag(0, 1): layer 1 can
only fit sample A and layer 2 only sample B.

- Full GD at eta = 1 zeroes the residual in 1 step with 2 writes.
- Sparse updating needs 2 steps with 2 writes: layer 1 fits A, then layer 2
  fits B.
- The gain is g = 1. Writes are the same and steps are doubled, as R4 predicts.

## 4. Measurement: the real network is coherent

The network is the digits MLP, 64-256x6-10 with tanh. I computed the exact
per-layer kernels K_l on 48 training images (a 480x480 matrix per layer, using
the logits):

| | init | after 10 epochs of BP |
|---|---|---|
| coherence lambda_max(K) / sum_l lambda_max(K_l) (1 = coherent, 1/7 = orthogonal) | 0.92 | 0.84 |
| gain g for a single layer (upper limit 7) | 5.1 | 4.8 |
| g for the stagger sets {0,5}, {4}, {3}, {2}, {1,6} | 5.0, 5.9, 5.1, 5.1, 3.4 | 3.9, 6.2, 6.1, 6.4, 3.2 |

The flatter directions are shared too. Along K's eigendirections number 0 to
400, which span eigenvalues from 10,570 down to 35, no layer carries more than
27% of any direction. An even split would be 14%.

**Conclusion.** This network sits close to the coherent regime, so R3 applies.

## 5. Why SSFA did not benefit

In DFA, layer l's update direction is not J_l^T r. The output change is
J_l Btilde_l^T r, whose "kernel" is not positive semi-definite. It was measured
as only about 0.3 to 0.4 aligned with the true gradient. The misaligned part of
each step behaves like noise, so the learning rate can't be raised.

This matches E1/E3: SSFA's best learning rate equalled dense DFA's on digits
(0.5 vs 0.5) and on MNIST (0.2 vs 0.2) instead of being several times larger.
It is a hypothesis consistent with the data, not a derivation.

## 6. Compute and memory per step (hand count, L = 7)

Units: one forward matmul = 1. Backward costs 1 per delta propagation plus
1 per weight gradient.

| | forward | delta props | weight grads | total |
|---|---|---|---|---|
| full BP | 7 | 6 | 7 | 20 |
| sparse BP, stagger P=5 (sets {0,5},{4},{3},{2},{1,6}) | 7 | mean (6+2+3+4+5)/5 = 4 | mean 1.4 | 12.4 (62%) |
| sparse BP, bottom-up blocks {0,1},{2,3},{4},{5},{6} | 7 | (6+4+2+1+0)/5 = 2.6 | 1.4 | 11.0 (55%) |

The backward sweep has to reach the lowest scheduled layer. Any step that
includes layer 0 therefore holds every activation, so **exact-gradient sparse
updates give no peak-memory bound**. The benefits are write count and compute.

## 7. Pre-registered prediction (not yet run)

The test is sparse BP with the stagger schedule, P = 5, on digits, with the
learning rate selected on the validation split.

1. Best learning rate: between 0.9 and 1.5, which is 3 to 5 times BP's tuned
   0.3, following g ≈ 4 to 6 for single-layer sets.
2. At that learning rate, test accuracy per epoch is within seed noise of full
   BP (96.9 ± 0.5), while writing 20% of the layers per step.
3. At matched weight writes, it beats both full BP and write-matched BP with
   accumulation (96.56).

**The theory is falsified if** the best learning rate is ≤ 0.3, or if accuracy
per epoch is clearly below BP (more than 1 point).

## Caveats

- This is linearized, first-order analysis. It ignores the O(eta^2) cross-terms
  in a cycle and assumes the kernels change slowly.
- The logit kernel stands in for cross-entropy curvature. Only the *ratio* of
  learning rates is predicted, not the absolute value.
- Minibatch noise and edge-of-stability behaviour can shift the optimum.
- Coherence was measured on one small MLP. Convolutional and residual networks
  still need checking.
