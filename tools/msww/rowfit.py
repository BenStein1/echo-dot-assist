"""Fit one row of the multi-wakeword decoder's [15,512] head (kernel/Dense:20) to a new word.

New word = class K (a slot the stock set never uses). The engine's posterior of K is softmax over 15 logits, so with the other
14 logits fixed (the stock engine's own output, minus the slot's old logit) it is  p = sigmoid(z - L),  L = logsumexp(others),
z = w . x + b, x the engine's 512-d input of the head. That is a logistic regression with an offset. Positives: the cycles of a
word's accept (found by multiple-instance refinement inside windows the word was said in); negatives: all other cycles.
The int8 row is then the same asymmetric per-row quantisation the head uses; forward() is the engine's exact integer path.
"""
import numpy as np
import spotter

K_DEFAULT = 12


def base_logit(Y, k):
    o = np.delete(Y, k, axis=1)
    m = o.max(1)
    return m + np.log(np.exp(o - m[:, None]).sum(1))


def quantize_row(w):
    w = np.asarray(w, np.float64)
    lo, hi = min(w.min(), 0.0), max(w.max(), 0.0)
    s = max((hi - lo) / 255.0, 1e-8)
    z = float(np.clip(np.rint(-128 - lo / s), -128, 127))
    q = np.clip(np.rint(w / s + z), -128, 127).astype(np.int8)
    return q, np.float32(s), np.int32(z), np.int32(q.astype(np.int64).sum())


def forward(X, q, sw, zw, bias):
    """The engine's HybridGemm for one row: per-cycle dynamic int8 quantisation of x, integer dot with (q - zp)."""
    X = X.astype(np.float32)
    lo, hi = X.min(1), X.max(1)
    sc = ((hi - lo) / np.float32(127)).astype(np.float32)
    sc = np.where(sc == 0, np.float32(1), sc)
    inv = np.float32(1) / sc
    z = np.rint(-lo * inv)
    xq = np.minimum(np.rint(X * inv[:, None]) + z[:, None], 127) - z[:, None]
    acc = xq.astype(np.int64) @ (q.astype(np.int64) - int(zw))
    return acc.astype(np.float32) * (sc * np.float32(sw)) + np.float32(bias)


def fit_logistic(X, y, wt, L, l2, v0=None, iters=200):
    """minimise sum wt*BCE(sigmoid(X v + c - L), y) + l2*|v|^2 ; returns v, c."""
    from scipy.optimize import minimize
    n, d = X.shape
    def f(theta):
        v, c = theta[:d], theta[d]
        z = X @ v + c - L
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        loss = -(wt * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() + l2 * (v @ v)
        g = wt * (p - y)
        return loss, np.concatenate([X.T @ g + 2 * l2 * v, [g.sum()]])
    th0 = np.zeros(d + 1) if v0 is None else v0
    r = minimize(f, th0, jac=True, method='L-BFGS-B', options={'maxiter': iters})
    return r.x


def mil_fit(pos_windows, neg, k=K_DEFAULT, l2=1.0, rounds=4, decay=0.6, hard=0.02, neg_weight=1.0, pos_cycles=3):
    """pos_windows: list of (X[n,512], L[n], burst_mask[n]) one per positive window (word said somewhere inside);
    neg: (X, L) all negative cycles.  Returns (w, b) float row (applied to raw x)."""
    from scipy.optimize import minimize  # noqa (fit_logistic imports it too)
    Xn, Ln = neg
    allX = np.concatenate([Xn] + [w[0] for w in pos_windows])
    mu, sd = allX.mean(0), allX.std(0) + 1e-3
    zn = (Xn - mu) / sd
    pw = [((w[0] - mu) / sd, w[1], w[2]) for w in pos_windows]
    # init: mean of burst cycles of positive windows against the negatives
    v0 = np.concatenate([np.mean([x[m].mean(0) for x, _, m in pw if m.any()], 0) - zn.mean(0) * 1.0, [0.0]])
    v0[:-1] /= np.linalg.norm(v0[:-1]) / 3
    theta = v0
    wneg = np.full(len(zn), neg_weight)
    for r in range(rounds):
        v, c = theta[:-1], theta[-1]
        pos_idx = []
        for x, L, m in pw:
            s = spotter.smooth(1 / (1 + np.exp(-np.clip(x @ v + c - L, -30, 30))), decay)
            s = np.where(m, s, -1)
            t = int(np.argmax(s)); pos_idx.append(range(max(0, t - pos_cycles + 1), t + 1))
        Xp = np.concatenate([pw[i][0][list(ix)] for i, ix in enumerate(pos_idx)])
        Lp = np.concatenate([pw[i][1][list(ix)] for i, ix in enumerate(pos_idx)])
        zneg = zn @ v + c - Ln
        thr = np.quantile(zneg, 1 - hard)
        wn = wneg * np.where(zneg >= thr, 5.0, 1.0)
        X = np.concatenate([Xp, zn]); y = np.concatenate([np.ones(len(Xp)), np.zeros(len(zn))])
        wt = np.concatenate([np.full(len(Xp), len(zn) / len(Xp) * 0.05), wn])
        L = np.concatenate([Lp, Ln])
        theta = fit_logistic(X, y, wt, L, l2, theta)
    v, c = theta[:-1], theta[-1]
    w = v / sd
    b = c - (mu * w).sum()
    return w, b
