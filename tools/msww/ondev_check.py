#!/usr/bin/env python3
"""Check the on-device row trainer (src/tools/msww_row.c) against the scipy fit on identical data, per leave-one-recording-out fold.

Dumps (build/msww/ondev/): POOL.bin = the shipped compact pool (int8 + scale per cycle, weights) + the enrolling user's own non-word audio
(as it would be on the Dot: from the enrollment session), POS.bin = the word windows (real + augmented copies).  Runs the C trainer, scores its
row exactly as the engine would (T.measure, quantised) on the held-out recording and on the HELD voices, and does the same for T.fit on the
same (dequantised) pool.

  ondev_check.py BIN [--variant hardrand5] [--folds ...]       BIN = build/msww_row (cc -O2 -o build/msww_row src/tools/msww_row.c -lm)
"""
import argparse, json, os, struct, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ds, train as T, loro, compact, rowfit as F

OUT = os.path.join(ds.M, 'ondev')


def q8(X):
    sc = np.maximum(np.abs(X).max(1), 1e-6) / 127.0
    return np.rint(X / sc[:, None]).astype(np.int8), sc.astype(np.float32)


def write_pool(path, X, L, W, seq):
    q, sc = q8(X)
    with open(path, 'wb') as f:
        f.write(b'MSWP' + struct.pack('<I', len(X)))
        for i in range(len(X)):
            f.write(struct.pack('<fff4B', float(L[i]), float(W[i]), float(sc[i]), int(seq[i]), 0, 0, 0)); f.write(q[i].tobytes())
    return q.astype(np.float32) * sc[:, None]        # what the trainer sees


def write_pos(path, S, names):
    wins = T._pos_windows(S, names)
    with open(path, 'wb') as f:
        f.write(b'MSWQ' + struct.pack('<I', len(wins)))
        for n, X, L, mask, _ in wins:
            f.write(struct.pack('<I', len(X)))
            for i in range(len(X)):
                f.write(struct.pack('<fI', float(L[i]), int(mask[i]))); f.write(X[i].astype('<f4').tobytes())
    return len(wins)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bin'); ap.add_argument('--variant', default='hardrand5'); ap.add_argument('--folds', default=','.join(ds.BEN))
    ap.add_argument('--accepts', default='0.8,0.5,0.3'); ap.add_argument('--decay', type=float, default=0.6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    S = loro.load(T.K, True)
    dev = [v for v in ds.DEV if v in S]; held = [v for v in ds.HELD if v in S]
    P, kept, n = compact.build(S, dev, a.variant, 1500, 0.05)
    ps = next(iter(P.values()))
    accepts = [float(x) for x in a.accepts.split(',')]
    for h in a.folds.split(','):
        tr = [x for x in ds.BEN if x != h]; names = loro.with_aug(tr, S, True)
        # negatives: the pool (non-sequential) then each training recording's non-word cycles (sequential runs)
        # seq codes (see msww_row.c): 0 pool cycle, not time-ordered (unsmoothed); 1 first cycle of a recording's run; 2 continues it
        Xs, Ls, Ws, Sq, cuts = [ps['X']], [ps['L']], [ps['wt']], [np.zeros(len(ps['X']), np.uint8)], [len(ps['X'])]
        for nm in names:
            s = S[nm]; drop = ds.in_win(s['t'], s['win']['pos']) | ds.in_win(s['t'], s['win']['ignore']); k = ~drop
            Xs.append(s['X'][k]); Ls.append(s['L'][k]); Ws.append(np.ones(int(k.sum()))); sq = np.full(int(k.sum()), 2, np.uint8); sq[0] = 1; Sq.append(sq); cuts.append(cuts[-1] + int(k.sum()))
        X = np.concatenate(Xs); L = np.concatenate(Ls); W = np.concatenate(Ws); Sq = np.concatenate(Sq)
        pf, qf, jf = [os.path.join(OUT, f'{a.variant}-{h}.{e}') for e in ('pool', 'pos', 'json')]
        Xdq = write_pool(pf, X, L, W, Sq); nwin = write_pos(qf, S, names)
        t0 = time.time()
        r = subprocess.run([a.bin, pf, qf, jf], capture_output=True, text=True); dt = time.time() - t0
        if r.returncode: sys.exit('trainer failed: ' + r.stderr)
        j = json.load(open(jf)); rr = j['row']
        mc = dict(w=np.array(j['w']), b=j['b'], row=dict(q=np.array(rr['q'], np.int8), s=np.float32(rr['s']), zp=np.int32(rr['zp']), sums=np.int32(rr['sums']), bias=np.float32(rr['bias'])))
        # reference: the scipy fit on the SAME inputs: the dequantised pool (Xdq) for every negative, the float word windows for positives.
        # Each recording is split in two streams: ':pos' (its word windows, every cycle otherwise ignored) and ':neg' (its non-word cycles, from Xdq).
        S2 = dict(S); negs = ['pool']
        S2['pool'] = compact.pool_stream('pool', Xdq[:cuts[0]], L[:cuts[0]], W[:cuts[0]], seq=False)
        poss = []
        for i, nm in enumerate(names):
            s = S[nm]; c0, c1 = cuts[i], cuts[i + 1]
            S2[nm + ':pos'] = dict(s, win=dict(pos=s['win']['pos'], dot=[], speech=[], reply=[], ignore=[(-1e9, 1e9)])); poss.append(nm + ':pos')
            S2[nm + ':neg'] = compact.pool_stream(nm + ':neg', Xdq[c0:c1], L[c0:c1], W[c0:c1], seq=True); negs.append(nm + ':neg')
        mp = T.fit(S2, poss, negs, rounds=5)
        cor = float(np.corrcoef(mc['w'], mp['w'])[0, 1])
        line = f'fold {h}: pool {len(X)} cycles ({os.path.getsize(pf) / 1e6:.1f} MB) + {nwin} word windows; C trainer {dt:.1f}s on this PC; row corr(C, scipy) {cor:.3f}'
        print(line, flush=True)
        for acc in accepts:
            for tag, m in (('C    ', mc), ('scipy', mp)):
                t1, _ = T.measure(S, [h], m, a.decay, acc, True); c, _ = T.measure(S, held, m, a.decay, acc, True)
                print(f'   accept {acc}: {tag} Nova {t1["pos_hit"]}/{t1["pos"]}  Ben non-word FA {t1["dot_fa"] + t1["speech_fa"] + t1["reply_fa"] + t1["unl_fa"]}  HELD corpus FA {c["corpus_fa"]}', flush=True)


if __name__ == '__main__':
    main()
