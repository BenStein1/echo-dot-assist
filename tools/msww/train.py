"""Row trainer for the multi-wakeword decoder's [15,512] head: fit one class row to a new word from a few spoken examples.

  fit()      logistic row against the engine's 14 other logits (competing classes), multiple-instance choice of the word's cycles,
             hard-negative mining on the streams it is given (the caller passes DEV streams only), optional quantisation-aware stage that
             optimises the int8 row through the engine's integer path.
  choose()   (decay, accept) for the word from streams the row was not fitted on.
  categorize()/measure()  the same accounting for python-simulated spotter events and for RESULT lines of the real engine.
"""
import numpy as np
import ds, rowfit as F, spotter
from ds import DT

K = 12  # the class slot (unused by every stock set: 1 4 6 8 9 10 12 13)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def xhat(X):
    """The engine's per-cycle dynamic int8 quantisation of the input, as floats: x_hat = xq * scale."""
    X = X.astype(np.float32)
    lo, hi = X.min(1), X.max(1)
    sc = ((hi - lo) / np.float32(127)).astype(np.float32)
    sc = np.where(sc == 0, np.float32(1), sc)
    inv = np.float32(1) / sc
    z = np.rint(-lo * inv)
    xq = np.minimum(np.rint(X * inv[:, None]) + z[:, None], 127) - z[:, None]
    return (xq * sc[:, None]).astype(np.float32)


def _logreg(X, y, wt, L, l2, theta0, iters=150):
    from scipy.optimize import minimize
    n, d = X.shape
    X = np.ascontiguousarray(X, np.float32)
    def f(th):
        v, c = th[:d], th[d]
        z = X @ v.astype(np.float32) + np.float32(c) - L
        p = sigmoid(z)
        loss = -(wt * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() + l2 * float(v @ v)
        g = (wt * (p - y)).astype(np.float32)
        return loss, np.concatenate([X.T @ g + 2 * l2 * v, [g.sum()]]).astype(np.float64)
    return minimize(f, theta0, jac=True, method='L-BFGS-B', options={'maxiter': iters}).x


def _pos_windows(streams, names):
    out = []
    for n in names:
        s = streams[n]
        for a, b in s['win']['pos']:
            sel = (s['t'] >= a) & (s['t'] < b)
            mask = (s['t'][sel] >= a + 0.3) & (s['t'][sel] <= b - 0.2)
            if mask.any():
                out.append((n, s['X'][sel], s['L'][sel], mask, s['Xh'][sel] if 'Xh' in s else None))
    return out


def _neg_pool(streams, names):
    """-> X, L, Xh, weight, segments [(start, end, sequential)].  A stream may carry 'wt' (per-cycle weight, a coreset's counts) and
    'seq': False (cycles are not in time order: no temporal smoothing when mining)."""
    Xs, Ls, Xh, Ws, seg, o = [], [], [], [], [], 0
    for n in names:
        s = streams[n]
        drop = ds.in_win(s['t'], s['win']['pos']) | ds.in_win(s['t'], s['win']['ignore'])
        Xs.append(s['X'][~drop]); Ls.append(s['L'][~drop])
        Ws.append(s['wt'][~drop] if 'wt' in s else np.ones(int((~drop).sum())))
        seg.append((o, o + int((~drop).sum()), s.get('seq', True))); o += int((~drop).sum())
        if 'Xh' in s: Xh.append(s['Xh'][~drop])
    return np.concatenate(Xs), np.concatenate(Ls), (np.concatenate(Xh) if Xh else None), np.concatenate(Ws), seg


def _smooth_pool(p, seg, decay):
    out = p.copy()
    for a, b, q in seg:
        if q: out[a:b] = spotter.smooth(p[a:b], decay)
    return out


def fit(streams, ben_names, neg_names, decay=0.6, l2=3.0, rounds=5, pos_cycles=3, pos_frac=0.05, mine_thr=0.08, mine_w=8.0,
        qat=0, seed=0):
    """-> dict(w, b, row) ; w,b float row on the raw 512-d input (z = x.w + b); row = quantised (q, s, zp, sums, bias)."""
    pw = _pos_windows(streams, ben_names)
    Xn, Ln, Xnh, Wn, seg = _neg_pool(streams, list(ben_names) + list(neg_names))
    allX = np.concatenate([Xn] + [p[1] for p in pw])
    mu, sd = allX.mean(0), allX.std(0) + 1e-3
    zn = ((Xn - mu) / sd).astype(np.float32)
    zp = [((p[1] - mu) / sd).astype(np.float32) for p in pw]
    v0 = np.mean([zp[i][pw[i][3]].mean(0) for i in range(len(pw))], 0) - zn.mean(0)
    v0 = v0 / np.linalg.norm(v0) * 3
    theta = np.concatenate([v0, [0.0]])
    wn = np.ones(len(zn))
    for r in range(rounds):
        v, c = theta[:-1], theta[-1]
        idx_p, y_p = [], []
        for i, (n, X, L, m, _) in enumerate(pw):
            sm = spotter.smooth(sigmoid(zp[i] @ v + c - L), decay)
            sm = np.where(m, sm, -1)
            t = int(np.argmax(sm))
            idx_p.append(list(range(max(0, t - pos_cycles + 1), t + 1)))
        Xp = np.concatenate([zp[i][ix] for i, ix in enumerate(idx_p)])
        Lp = np.concatenate([pw[i][2][ix] for i, ix in enumerate(idx_p)])
        # hard-negative mining on the given negatives: every cycle of a smoothed score above mine_thr, and its neighbours
        zneg = zn @ v + c - Ln
        sm = _smooth_pool(sigmoid(zneg), seg, decay)
        hard = np.convolve((sm >= mine_thr).astype(float), np.ones(5), 'same') > 0
        wn = Wn * np.where(hard, mine_w, 1.0)
        X = np.concatenate([Xp, zn]); y = np.concatenate([np.ones(len(Xp)), np.zeros(len(zn))])
        wt = np.concatenate([np.full(len(Xp), pos_frac * wn.sum() / len(Xp)), wn])
        L = np.concatenate([Lp, Ln]).astype(np.float32)
        theta = _logreg(X, y, wt, L, l2, theta)
    v, c = theta[:-1], theta[-1]
    w = v / sd
    b = c - float((mu * w).sum())
    out = dict(w=w, b=b)
    if qat:
        out = _qat(out, pw, idx_p, Xn, Xnh, Ln, wn, pos_frac, qat, mu, sd, l2)
    q, s, z, sm_ = F.quantize_row(out['w'])
    out['row'] = dict(q=q, s=s, zp=z, sums=sm_, bias=np.float32(out['b']))
    return out


def _qat(m, pw, idx_p, Xn, Xnh, Ln, wn, pos_frac, steps, mu, sd, l2):
    """Straight-through fine-tuning of the int8 row: forward uses the engine's quantised input x_hat and the dequantised row."""
    Xp = np.concatenate([pw[i][1][ix] for i, ix in enumerate(idx_p)])
    Lp = np.concatenate([pw[i][2][ix] for i, ix in enumerate(idx_p)])
    Xall = xhat(np.concatenate([Xp, Xn])).astype(np.float32)
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))]).astype(np.float32)
    wt = np.concatenate([np.full(len(Xp), pos_frac * wn.sum() / len(Xp)), wn]).astype(np.float32)
    L = np.concatenate([Lp, Ln]).astype(np.float32)
    w = m['w'].astype(np.float32).copy(); b = np.float32(m['b'])
    mom1, mom2, mb1, mb2 = np.zeros_like(w), np.zeros_like(w), 0.0, 0.0
    lr = 0.02 * float(np.abs(w).mean())
    best = (1e30, w.copy(), b)
    for t in range(1, steps + 1):
        q, s, zq, _ = F.quantize_row(w)
        wd = (np.float32(s) * (q.astype(np.float32) - np.float32(zq))).astype(np.float32)
        z = Xall @ wd + b - L
        p = sigmoid(z)
        loss = float(-(wt * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum())
        if loss < best[0]: best = (loss, w.copy(), b)
        g = wt * (p - y)
        gw = Xall.T @ g
        gb = float(g.sum())
        mom1 = 0.9 * mom1 + 0.1 * gw; mom2 = 0.999 * mom2 + 0.001 * gw * gw
        w = w - lr * (mom1 / (1 - 0.9 ** t)) / (np.sqrt(mom2 / (1 - 0.999 ** t)) + 1e-8)
        mb1 = 0.9 * mb1 + 0.1 * gb; mb2 = 0.999 * mb2 + 0.001 * gb * gb
        b = np.float32(b - lr * 10 * (mb1 / (1 - 0.9 ** t)) / (np.sqrt(mb2 / (1 - 0.999 ** t)) + 1e-8))
    return dict(w=best[1].astype(np.float64), b=float(best[2]))


def zscore(s, m, quant=True):
    """Row output z for a stream: the engine's own integer path (quant) or float."""
    if quant:
        r = m['row']
        return F.forward(s['X'], r['q'], r['s'], r['zp'], r['bias'])
    return s['X'] @ m['w'] + m['b']


def posterior(s, m, quant=True):
    return sigmoid(zscore(s, m, quant) - s['L'])


def events_of(s, m, decay, accept, quant=True, notify=0.12):
    p = posterior(s, m, quant)
    return [(e[2] / 16000.0, e[1]) for e in spotter.events(p, decay, accept, notify) if e[0] == 2]


def categorize(s, ev):
    """ev: [(time_s, score)] accepts of the word in stream s -> dict of counts and the event list with categories."""
    w = s['win']
    res = dict(pos=len(w['pos']), pos_hit=0, dot=len(w['dot']), dot_fa=0, speech=len(w['speech']), speech_fa=0, reply_fa=0, unl_fa=0,
               ign=0, corpus_fa=0, events=[], missed=[])
    used = set()
    for t, sc in ev:
        cat = None
        for kind in ('pos', 'dot', 'speech', 'reply', 'ignore'):
            for i, (a, b) in enumerate(w[kind]):
                if a <= t <= b + 0.5 * (kind in ('pos', 'dot', 'speech')):
                    cat = (kind, i); break
            if cat: break
        if s['kind'] == 'corpus':
            res['corpus_fa'] += 1; cat = ('corpus', 0)
        elif cat is None:
            res['unl_fa'] += 1; cat = ('unlabelled', 0)
        elif cat[0] == 'pos':
            if cat not in used: res['pos_hit'] += 1
            used.add(cat)
        elif cat[0] == 'dot':
            if cat not in used: res['dot_fa'] += 1
            used.add(cat)
        elif cat[0] == 'speech':
            if cat not in used: res['speech_fa'] += 1
            used.add(cat)
        elif cat[0] == 'reply':
            res['reply_fa'] += 1
        else:
            res['ign'] += 1
        res['events'].append((round(t, 2), round(float(sc), 4), cat[0]))
    for i, (a, b) in enumerate(w['pos']):
        if ('pos', i) not in used:
            res['missed'].append((round(a, 2), round(b, 2)))
    return res


def measure(streams, names, m, decay, accept, quant=True):
    tot = dict(pos=0, pos_hit=0, dot=0, dot_fa=0, speech=0, speech_fa=0, reply_fa=0, unl_fa=0, ign=0, corpus_fa=0, hours=0.0)
    per = {}
    for n in names:
        s = streams[n]
        r = categorize(s, events_of(s, m, decay, accept, quant))
        per[n] = r
        for k in tot:
            if k == 'hours': tot[k] += len(s['X']) * DT / 3600.0
            else: tot[k] += r[k]
    return tot, per


def fa_count(r):
    return r['dot_fa'] + r['speech_fa'] + r['reply_fa'] + r['unl_fa'] + r['corpus_fa']


GRID_DECAY = (0.0, 0.3, 0.6)
GRID_ACCEPT = (0.35, 0.45, 0.523, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def choose(streams, m, names, quant=True, grid_decay=GRID_DECAY, grid_accept=GRID_ACCEPT):
    """Pick (decay, accept) for the word on streams that the row m was NOT fitted on: fewest false accepts, then most words found,
    then the widest margin (the higher accept)."""
    best = None
    for d in grid_decay:
        for a in grid_accept:
            tot, _ = measure(streams, names, m, d, a, quant)
            key = (fa_count(tot), -tot['pos_hit'], -a)
            if best is None or key < best[0]:
                best = (key, d, a, tot)
    return best[1], best[2], best[3]
