#!/usr/bin/env python3
"""Leave-one-recording-out protocol for an msww row (python simulation of the engine's decoder head + spotter; the packaged-engine
gate is gate.py).

For every held-out recording h of Ben's four:
  * fit on the other three (real word windows, their augmented copies, all their non-word audio as negatives) + the DEV corpus voices;
  * choose (decay, accept) from rows that did NOT see the data it is judged on: inner leave-one-recording-out over the three training
    recordings, with a third of the DEV voices held back from every inner fit (the calibration voices);
  * refit on all training data with the chosen values, then measure on h and on the HELD corpus voices (never used by anything above).
The held-out recording and the HELD voices are touched once, for the final numbers.

  loro.py [--aug 1] [--qat 0] [--k 12] [--l2 3] [--pos-frac 0.05] [--mine-w 8] [--fixed DECAY,ACCEPT]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ds, train as T, spotter


def load(k, aug=True):
    S = {}
    S.update(ds.load_ben(k))
    S.update(ds.load_corpus(k))
    if aug:
        S.update(ds.load_aug(k))
    return S


def with_aug(names, S, aug):
    return list(names) + (['aug_' + n for n in names if 'aug_' + n in S] if aug else [])


BLOCK = 37   # cycles (3 s) per negative unit


def peaks(s, m, decay):
    """Smoothed-score peaks: one per word window, and one per 3 s block of everything else (ignored zones and word windows excluded)."""
    sm = spotter.smooth(T.posterior(s, m, False), decay)
    pos = [float(sm[(s['t'] >= a) & (s['t'] < b)].max()) for a, b in s['win']['pos'] if ((s['t'] >= a) & (s['t'] < b)).any()]
    keep = ~(ds.in_win(s['t'], s['win']['pos']) | ds.in_win(s['t'], s['win']['ignore']))
    v = np.where(keep, sm, -1.0)
    n = len(v) // BLOCK
    neg = [float(v[i * BLOCK:(i + 1) * BLOCK].max()) for i in range(n)] if n else []
    return pos, [x for x in neg if x >= 0]


def pick(S, train_recs, dev, cfg):
    """Hyper-parameters, decay and accept from rows that did NOT see the data they are judged on (inner leave-one-recording-out, a
    third of the DEV voices kept out of every inner fit as calibration voices).  Cost = misses + FA_COST * false accepts, where a false
    accept is a negative block (3 s of non-word audio, or a dot window) whose smoothed score reaches the accept threshold."""
    cal = dev[::3]; fit_dev = [v for v in dev if v not in cal]
    best = None
    for hp in cfg['grid']:
        fitkw = dict(cfg['fit']); fitkw.update(hp)
        agg = {d: [] for d in T.GRID_DECAY}
        for r in train_recs:
            inner = [n for n in train_recs if n != r]
            m = T.fit(S, with_aug(inner, S, cfg['aug']), fit_dev, **fitkw)
            for d in T.GRID_DECAY:
                pp, nn = peaks(S[r], m, d)
                for v in cal:
                    pc, nc = peaks(S[v], m, d); nn += nc
                agg[d].append((pp, nn))
        for d in T.GRID_DECAY:
            costs = []
            for a in T.GRID_ACCEPT:
                miss = sum(sum(1 for x in pp if x < a) for pp, nn in agg[d])
                fa = sum(sum(1 for x in nn if x >= a) for pp, nn in agg[d])
                costs.append((miss + cfg['fa_cost'] * fa, a, miss, fa))
            lo = min(c[0] for c in costs)
            plateau = [c for c in costs if c[0] == lo]
            mid = plateau[len(plateau) // 2]        # the centre of the cheapest plateau: the widest margin to both sides
            key = (lo, abs(len(plateau)) * -1)
            if best is None or key < best[0]:
                best = (key, hp, d, mid[1], mid[2], mid[3])
    return best[1], best[2], best[3], best


def run(cfg, verbose=True):
    S = load(cfg['k'], cfg['aug'])
    dev = [v for v in ds.DEV if v in S]
    held = [v for v in ds.HELD if v in S]
    out = {}
    tot_all = None
    for h in ds.BEN:
        t0 = time.time()
        tr = [n for n in ds.BEN if n != h]
        hp = {}
        if cfg['fixed']:
            d, a = cfg['fixed']
        else:
            hp, d, a, info = pick(S, tr, dev, cfg)
            print(f'  [{h}] inner pick: {hp} decay={d} accept={a} (inner cost {info[0][0]:.0f}: {info[4]} misses, {info[5]} FA blocks)', flush=True)
        fitkw = dict(cfg['fit']); fitkw.update(hp)
        m = T.fit(S, with_aug(tr, S, cfg['aug']), dev, **fitkw)
        tb, pb = T.measure(S, [h], m, d, a, cfg['quant'])
        tc, pc = T.measure(S, held, m, d, a, cfg['quant'])
        curve = {}
        for a2 in T.GRID_ACCEPT:
            t2, _ = T.measure(S, [h], m, d, a2, cfg['quant']); c2, _ = T.measure(S, held, m, d, a2, cfg['quant'])
            curve[a2] = dict(hit=t2['pos_hit'], pos=t2['pos'], fa=t2['dot_fa'] + t2['speech_fa'] + t2['reply_fa'] + t2['unl_fa'], corpus_fa=c2['corpus_fa'])
        if verbose:
            print('    operating curve (accept: Nova hits / Ben FA / HELD-corpus FA) at decay', d, ' '.join(f"{a2}:{c['hit']}/{c['fa']}/{c['corpus_fa']}" for a2, c in curve.items()))
        out[h] = dict(curve=curve, decay=d, accept=a, hp=hp, ben=pb[h], ben_tot=tb, corpus_tot=tc, corpus=pc)
        if verbose:
            r = pb[h]
            print(f"held-out {h:8s} decay={d} accept={a:5.3f} | Nova {r['pos_hit']}/{r['pos']} dot-FA {r['dot_fa']}/{r['dot']} speech-FA {r['speech_fa']}/{r['speech']} "
                  f"reply-FA {r['reply_fa']} unlabelled {r['unl_fa']} | HELD corpus {tc['hours']:.2f} h: {tc['corpus_fa']} FA  [{time.time() - t0:.0f}s]", flush=True)
            if r['events']: print('    accepts:', r['events'], ' missed:', r['missed'])
    return out


def summarize(out):
    k = ['pos', 'pos_hit', 'dot', 'dot_fa', 'speech', 'speech_fa', 'reply_fa', 'unl_fa']
    t = {x: sum(o['ben'][x] for o in out.values()) for x in k}
    cfa = [sum(o['corpus'][v]['corpus_fa'] for v in o['corpus']) for o in out.values()]
    hrs = list(out.values())[0]['corpus_tot']['hours']
    print(f"TOTAL Nova {t['pos_hit']}/{t['pos']}  dot-FA {t['dot_fa']}/{t['dot']}  speech-FA {t['speech_fa']}/{t['speech']}  reply-FA {t['reply_fa']}  unlabelled {t['unl_fa']}"
          f"  | HELD corpus {hrs:.2f} h per fold: FA per fold {cfa}")
    return t


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--aug', type=int, default=1); ap.add_argument('--qat', type=int, default=0); ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--l2', type=float, default=3.0); ap.add_argument('--pos-frac', type=float, default=0.05); ap.add_argument('--mine-w', type=float, default=8.0)
    ap.add_argument('--mine-thr', type=float, default=0.08); ap.add_argument('--fixed', default='')
    ap.add_argument('--float', action='store_true', help='score with the float row instead of the engine integer path')
    ap.add_argument('--fa-cost', type=float, default=3.0); ap.add_argument('--out', default='')
    ap.add_argument('--grid', default='3:0.05,3:0.15,1:0.05,10:0.05,10:0.15,1:0.15', help='l2:pos_frac candidates')
    a = ap.parse_args()
    grid = [dict(l2=float(g.split(':')[0]), pos_frac=float(g.split(':')[1])) for g in a.grid.split(',')]
    cfg = dict(k=a.k, grid=grid, fa_cost=a.fa_cost, aug=bool(a.aug), quant=not a.float, fixed=tuple(float(v) for v in a.fixed.split(',')) if a.fixed else None,
               fit=dict(l2=a.l2, pos_frac=a.pos_frac, mine_w=a.mine_w, mine_thr=a.mine_thr, qat=a.qat))
    o = run(cfg)
    summarize(o)
    if a.out:
        json.dump(o, open(a.out, 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
