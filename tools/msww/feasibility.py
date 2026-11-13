#!/usr/bin/env python3
"""Step 4: does a trained Dense:20 row separate Nova from dot on held-out recordings?  Leave-one-recording-out over Ben's captures,
TTS voices split into train / held-out.  usage: feasibility.py [--k 12] [--decay 0.6] [--accept 0.523]"""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import dataset as D, rowfit as F, spotter
from feat import softmax
from bursts import bursts

TTS_HELD = ['en-gb-f-elderly', 'en-us-m-middle', 'de-m-middle', 'en-ja-m-young', 'en-us-f-young']
BEN = ['w171828', 'w172114', 'live', 'rdot']
SINGLE = ['nova1', 'dot1']


def cyc_t(n):
    return (np.arange(n) + 1) * 0.08


def label_cycles(n, wins):
    lab = np.zeros(n, int); t = cyc_t(n)
    for a, b, kind, _ in wins:
        lab[(t >= a) & (t < b)] = {'pos': 1, 'dot': 2, 'speech': 3}[kind]
    return lab


def burst_mask(name, n):
    raw = os.path.join(D.M, 'rec', name + '.raw')
    m = np.zeros(n, bool); t = cyc_t(n)
    for a, b, _ in bursts(raw):
        m[(t >= a) & (t <= b + 0.4)] = True
    return m


def load_all(k):
    data = {}
    W = D.windows()
    for nme in BEN + SINGLE:
        X, Y, t = D.load(nme)
        data[nme] = dict(X=X, Y=Y, L=F.base_logit(Y, k), lab=label_cycles(len(X), W[nme]), win=W[nme], mask=burst_mask(nme, len(X)))
    for v in sorted(f[:-5] for f in os.listdir(os.path.join(D.M, 'tts')) if f.endswith('.feat')):
        X, Y, t = D.load(v)
        data[v] = dict(X=X, Y=Y, L=F.base_logit(Y, k), lab=np.zeros(len(X), int), win=[], mask=None)
    return data


def train(data, names, tts_names, k, l2, pos_cycles=3, rounds=4, max_pos=None):
    pos, negX, negL = [], [], []
    npos = 0
    for n in names:
        d = data[n]
        for a, b, kind, _ in d['win']:
            sel = (cyc_t(len(d['X'])) >= a) & (cyc_t(len(d['X'])) < b)
            if kind == 'pos' and (max_pos is None or npos < max_pos):
                pos.append((d['X'][sel], d['L'][sel], d['mask'][sel])); npos += 1
        keep = ~np.isin(d['lab'], [1]) if True else None
        # cycles of positive windows that are not the chosen cycle are ambiguous; leave them out of the negatives
        negX.append(d['X'][keep]); negL.append(d['L'][keep])
    for n in tts_names:
        negX.append(data[n]['X']); negL.append(data[n]['L'])
    w, b = F.mil_fit(pos, (np.concatenate(negX), np.concatenate(negL)), k=k, l2=l2, pos_cycles=pos_cycles, rounds=rounds)
    return w, b, npos


def scores(d, w, b, k, quant):
    if quant:
        q, s, z, sm = F.quantize_row(w)
        zz = F.forward(d['X'], q, s, z, b)
    else:
        zz = d['X'] @ w + b
    return 1 / (1 + np.exp(-np.clip(zz - d['L'], -30, 30)))


def evaluate(d, p, decay, accept, notify):
    ev = [e for e in spotter.events(p, decay, accept, notify) if e[0] == 2]
    res = dict(pos=0, pos_hit=0, dot=0, dot_fa=0, speech=0, speech_fa=0, unl_fa=0, accepts=len(ev), hours=len(p) * 0.08 / 3600)
    for a, b, kind, _ in d['win']:
        hit = any(a <= e[2] / 16000 <= b + 0.5 for e in ev)
        res[kind] += 1; res[{'pos': 'pos_hit', 'dot': 'dot_fa', 'speech': 'speech_fa'}[kind]] += hit
    res['unl_fa'] = sum(1 for e in ev if not any(a <= e[2] / 16000 <= b + 0.5 for a, b, _, _ in d['win']))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=12); ap.add_argument('--decay', type=float, default=0.6)
    ap.add_argument('--accept', type=float, default=0.523); ap.add_argument('--notify', type=float, default=0.12)
    ap.add_argument('--l2', type=float, default=3.0); ap.add_argument('--quant', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    a = ap.parse_args()
    data = load_all(a.k)
    tts_tr = [n for n in data if n not in BEN + SINGLE + TTS_HELD]
    tot = {}
    for held in BEN:
        names = [n for n in BEN + SINGLE if n != held]
        w, b, npos = train(data, names, tts_tr, a.k, a.l2)
        out = {}
        for tag, nm in [('ben', [held]), ('tts', TTS_HELD)]:
            for n in nm:
                p = scores(data[n], w, b, a.k, a.quant)
                r = evaluate(data[n], p, a.decay, a.accept, a.notify)
                for kk, v in r.items(): out[(tag, kk)] = out.get((tag, kk), 0) + v
        print(f"held-out {held:8s} trained on {npos:2d} Nova windows | Nova {out[('ben','pos_hit')]}/{out[('ben','pos')]}  dot-FA {out[('ben','dot_fa')]}/{out[('ben','dot')]}  "
              f"speech-FA {out[('ben','speech_fa')]}/{out[('ben','speech')]}  unlabelled-accepts {out[('ben','unl_fa')]} | TTS held-out {out[('tts','hours')]:.2f} h: {out[('tts','accepts')]} accepts")
        for kk, v in out.items(): tot[kk] = tot.get(kk, 0) + v
    print(f"TOTAL Nova {tot[('ben','pos_hit')]}/{tot[('ben','pos')]}  dot-FA {tot[('ben','dot_fa')]}/{tot[('ben','dot')]}  speech-FA {tot[('ben','speech_fa')]}/{tot[('ben','speech')]}  "
          f"unlabelled {tot[('ben','unl_fa')]} ; TTS {tot[('tts','hours')]:.2f} h negative speech, accepts {tot[('tts','accepts')]}")


if __name__ == '__main__':
    main()
