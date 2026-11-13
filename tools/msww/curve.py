#!/usr/bin/env python3
"""Enrollment-size curve: how many spoken examples does a new word need?

Per held-out recording (leave-one-recording-out over Ben's four) the row is fitted from n positive examples drawn from the other recordings
and a FIXED shared negative pool (the DEV corpus voices: 1.48 h of other speech, identical for every n), then measured on the held-out
recording and on the HELD corpus voices.  Columns:
   real n      n real utterances, nothing else
   aug N       the same real utterances plus augmented copies (gain / speed of those utterances) up to N positive clips in total
   +own        the shared pool plus the user's own non-word speech (what an enrolling user can say during the session) as extra negatives
The decay and accept are fixed (0.6 / 0.8, what the inner folds picked for most words) so that only the number of examples changes.

  curve.py [--draws 3] [--accept 0.8] [--decay 0.6]
"""
import argparse, copy, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ds, train as T, loro


def restrict(s, keep_idx, kind):
    """copy of a stream whose word windows are only keep_idx; the other windows are ignored (neither word nor negative)."""
    t = dict(s)
    w = copy.deepcopy(s['win'])
    keep = [w['pos'][i] for i in keep_idx]
    drop = [w['pos'][i] for i in range(len(w['pos'])) if i not in keep_idx]
    w['pos'] = keep
    w['ignore'] = w['ignore'] + drop
    t['win'] = w
    return t


def draw(S, train_recs, n_real, n_total, rng):
    """-> dict name -> stream for the chosen positives (real windows of Ben streams, augmented clips of the same windows)."""
    reals = [(r, i) for r in train_recs for i in range(len(S[r]['win']['pos']))]
    pick = [reals[j] for j in rng.permutation(len(reals))[:n_real]]
    out = {}
    for r in train_recs:
        idx = [i for rr, i in pick if rr == r]
        out[r] = restrict(S[r], idx, 'real')
    if n_total > n_real:
        extra = n_total - n_real
        cands = []
        for r in train_recs:
            a = S.get('aug_' + r)
            if a is None: continue
            for ci, c in enumerate(a['idx']):
                if (r, c['src']) in pick and not (c['gain'] == 0.0 and c['speed'] == 1.0):
                    cands.append((r, ci))
        sel = [cands[j] for j in rng.permutation(len(cands))[:extra]]
        for r in train_recs:
            if 'aug_' + r in S:
                out['aug_' + r] = restrict(S['aug_' + r], [ci for rr, ci in sel if rr == r], 'aug')
    return out


def run(a):
    S = loro.load(T.K, True)
    dev = [v for v in ds.DEV if v in S]
    held = [v for v in ds.HELD if v in S]
    cells = [('real', 2, 2), ('real', 5, 5), ('real', 10, 10), ('aug', 5, 20), ('aug', 10, 20), ('aug', 10, 40), ('aug', 11, 200)]
    res = {}
    for own in (False, True):
        for kind, n_real, n_tot in cells:
            acc = dict(hit=0, pos=0, fa=0, cfa=0, margin=[])
            t0 = time.time()
            for h in ds.BEN:
                tr = [n for n in ds.BEN if n != h]
                for d in range(a.draws if n_tot < 200 else 1):
                    rng = np.random.RandomState(1000 * d + 7)
                    P = draw(S, tr, min(n_real, sum(len(S[r]['win']['pos']) for r in tr)), n_tot, rng)
                    S2 = dict(S); S2.update(P)
                    # negatives: shared DEV pool always; the user's own non-word audio (training recordings) only in the +own variant
                    names = [n for n in P if not n.startswith('aug_')] + [n for n in P if n.startswith('aug_')]
                    m = T.fit(S2, names, dev, rounds=5) if own else fit_shared(S2, names, dev)
                    tot, per = T.measure(S2 if False else S, [h], m, a.decay, a.accept, True)
                    c, _ = T.measure(S, held, m, a.decay, a.accept, True)
                    acc['hit'] += tot['pos_hit']; acc['pos'] += tot['pos']
                    acc['fa'] += tot['dot_fa'] + tot['speech_fa'] + tot['reply_fa'] + tot['unl_fa']; acc['cfa'] += c['corpus_fa']
            res[(own, kind, n_real, n_tot)] = acc
            print(f"{'+own' if own else 'shared'} {kind:4s} real={n_real:2d} total={n_tot:3d}: Nova {acc['hit']}/{acc['pos']}  Ben non-word FA {acc['fa']}  HELD-corpus FA {acc['cfa']} (0.78 h x folds)  [{time.time() - t0:.0f}s]", flush=True)
    return res


def fit_shared(S2, pos_names, dev):
    """Shared negative pool only: the positives come from Ben streams, but those streams' non-word audio is NOT used as negatives."""
    S3 = dict(S2)
    for n in pos_names:
        if n.startswith('aug_'): continue
        s = dict(S3[n]); w = copy.deepcopy(s['win']); w['ignore'] = w['ignore'] + [(0.0, s['t'][-1] + 1)]   # everything but the kept word windows
        # (the pool drops cycles inside word windows AND ignore zones; the word windows themselves are read by _pos_windows)
        s['win'] = w; S3[n] = s
    return T.fit(S3, pos_names, dev, rounds=5)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=3); ap.add_argument('--accept', type=float, default=0.8); ap.add_argument('--decay', type=float, default=0.6)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    r = run(a)
    if a.out:
        json.dump({str(k): v for k, v in r.items()}, open(a.out, 'w'), indent=1, default=str)
