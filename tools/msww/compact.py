#!/usr/bin/env python3
"""Does a compact shared negative pool keep the held-out false-accept rate?  (design for shipping the pool to a Dot)

The pool is the DEV corpus (11 TTS voices, 1.48 h, 66.6k cycles x 512 float32 = 136 MB; the same pool for 17 voices / 2.26 h is ~200 MB).
Variants (all leave-one-recording-out over Ben's recordings, fixed decay/accept, scored on the HELD voices that nothing was fitted on):
   full        every cycle                                          (baseline)
   rand N%     uniform random cycles, weight 100/N
   core        k-means centroids of the pool (K of them), weight = cluster size  (summary statistics: means + counts)
   hard+core   every 'hard' cycle (engine says 1-P(background) >= tau: speech that already looks like some stock word, i.e. where a new
               class can steal mass) kept as is + core K of the remaining easy cycles
Sizes are what would ship: int8 per cycle (the engine's own per-cycle quantisation of the 512 inputs: 1 byte/dim + scale).

  compact.py [--folds w171828,w172114,live,rdot] [--accept 0.8] [--decay 0.6] OUT.json
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ds, train as T, loro, rowfit as F
from feat import softmax


def pool_stream(name, X, L, W, seq=False):
    return dict(name=name, X=X.astype(np.float32), Y=np.zeros((len(X), 15)), L=L, t=np.arange(len(X)) * ds.DT, kind='pool', wt=W, seq=seq,
                win=dict(pos=[], dot=[], speech=[], reply=[], ignore=[]))


def kmeans(X, k, iters=8, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = (C * C).sum(1)[None, :] - 2 * X @ C.T
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = X[m].mean(0)
    d = (C * C).sum(1)[None, :] - 2 * X @ C.T
    a = d.argmin(1)
    return C, np.bincount(a, minlength=k).astype(float), a


def build(S, dev, variant, kcore, tau, seed=0):
    X = np.concatenate([S[v]['X'] for v in dev]); L = np.concatenate([S[v]['L'] for v in dev]); Y = np.concatenate([S[v]['Y'] for v in dev])
    n = len(X); rng = np.random.RandomState(seed)
    if variant == 'full':
        return {v: S[v] for v in dev}, n, n
    if variant.startswith('rand'):
        frac = float(variant[4:]) / 100.0
        idx = np.sort(rng.choice(n, int(n * frac), replace=False))
        return {'pool': pool_stream('pool', X[idx], L[idx], np.full(len(idx), 1.0 / frac))}, len(idx), n
    if variant.startswith('hardrand'):          # every hard cycle + a uniform fraction of the easy ones (weight 1/frac)
        frac = float(variant[8:]) / 100.0
        hard = (1 - softmax(Y)[:, 0]) >= tau
        easy = np.flatnonzero(~hard)
        idx = np.sort(rng.choice(easy, int(len(easy) * frac), replace=False))
        sel = np.concatenate([np.flatnonzero(hard), idx]); W = np.concatenate([np.ones(int(hard.sum())), np.full(len(idx), 1.0 / frac)])
        return {'pool': pool_stream('pool', X[sel], L[sel], W)}, len(sel), n
    if variant == 'core':
        C, cnt, a = kmeans(X, kcore)
        Lc = np.bincount(a, weights=L, minlength=kcore) / np.maximum(cnt, 1)   # the engine's competing logit the row is compared against
        return {'pool': pool_stream('pool', C, Lc, cnt)}, kcore, n
    if variant == 'hard+core':
        danger = 1 - softmax(Y)[:, 0]
        hard = danger >= tau
        Xe, Le = X[~hard], L[~hard]
        C, cnt, a = kmeans(Xe, kcore)
        Lc = np.bincount(a, weights=Le, minlength=kcore) / np.maximum(cnt, 1)
        Xp = np.concatenate([X[hard], C]); Lp = np.concatenate([L[hard], Lc]); W = np.concatenate([np.ones(int(hard.sum())), cnt])
        return {'pool': pool_stream('pool', Xp, Lp, W)}, int(hard.sum()) + kcore, n
    raise ValueError(variant)


def run(a):
    S = loro.load(T.K, True)
    dev = [v for v in ds.DEV if v in S]; held = [v for v in ds.HELD if v in S]
    folds = a.folds.split(',')
    res = {}
    for variant in a.variants.split(','):
        t0 = time.time()
        P, kept, n = build(S, dev, variant, a.kcore, a.tau)
        S2 = dict(S); S2.update(P)
        hit = pos = ben_fa = cfa = 0; extra = {x: [0, 0] for x in a.sens}
        for h in folds:
            tr = [x for x in ds.BEN if x != h]
            m = T.fit(S2, loro.with_aug(tr, S2, True), list(P), rounds=5)
            t1, _ = T.measure(S, [h], m, a.decay, a.accept, True)
            c, _ = T.measure(S, held, m, a.decay, a.accept, True)
            for x in a.sens:
                t2, _ = T.measure(S, [h], m, a.decay, x, True); c2, _ = T.measure(S, held, m, a.decay, x, True)
                extra[x][0] += t2['dot_fa'] + t2['speech_fa'] + t2['reply_fa'] + t2['unl_fa']; extra[x][1] += c2['corpus_fa']
            hit += t1['pos_hit']; pos += t1['pos']; ben_fa += t1['dot_fa'] + t1['speech_fa'] + t1['reply_fa'] + t1['unl_fa']; cfa += c['corpus_fa']
        size_mb = kept * 513 / 1e6
        res[variant] = dict(kept_cycles=kept, of=n, ship_mb_int8=round(size_mb, 1), nova=f'{hit}/{pos}', ben_nonword_fa=ben_fa, held_corpus_fa=cfa, at_accept={str(x): dict(ben_nonword_fa=v[0], held_corpus_fa=v[1]) for x, v in extra.items()},
                            held_hours_per_fold=round(ds.hours({v: S[v] for v in held}), 2), folds=len(folds), secs=round(time.time() - t0))
        print(variant, res[variant], flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('out')
    ap.add_argument('--folds', default=','.join(ds.BEN)); ap.add_argument('--accept', type=float, default=0.8); ap.add_argument('--decay', type=float, default=0.6)
    ap.add_argument('--variants', default='full,rand10,rand2,core,hard+core'); ap.add_argument('--sens', type=lambda v: [float(x) for x in v.split(',')], default=[0.3, 0.5]); ap.add_argument('--kcore', type=int, default=1500); ap.add_argument('--tau', type=float, default=0.05)
    a = ap.parse_args()
    json.dump(run(a), open(a.out, 'w'), indent=1)
