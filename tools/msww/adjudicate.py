#!/usr/bin/env python3
"""Listen-free adjudication of engine accepts that the ASR check contradicted: is the audio under an accept more like Ben's 'Nova' or his 'dot'?

Reference classes are the single-word bursts Ben said in w171828 / w172114 / live (labels.py: word identity from prompts + manual timeline
selection, NOT from ASR).  A burst is described by duration, rms, spectral centroid and an 8-segment x 24-band log-mel shape (level removed);
the verdict is the 3-nearest-neighbour distance to each class (leave-one-out accuracy on the references is printed).  Not a transcription: a
similarity statement.  Usage: adjudicate.py [t1-t2 ...]   (rdot seconds; default = the four post-dot words of rdot)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ds

M, SR = ds.M, 16000


def fbank(nf=400, nm=24):
    f = np.fft.rfftfreq(nf, 1 / SR); hz = lambda m: 700 * (10 ** (m / 2595) - 1); mel = lambda h: 2595 * np.log10(1 + h / 700)
    p = hz(np.linspace(mel(100), mel(7600), nm + 2)); fb = np.zeros((nm, len(f)))
    for i in range(nm):
        a, b, c = p[i:i + 3]; fb[i] = np.clip(np.minimum((f - a) / (b - a), (c - f) / (c - b)), 0, None)
    return fb


FB = fbank()


def describe(x, a, b):
    s = x[int(a * SR):int(b * SR)]
    sp = np.abs(np.fft.rfft(s * np.hanning(len(s)))); f = np.fft.rfftfreq(len(s), 1 / SR)
    fr = np.array([np.log(FB @ np.abs(np.fft.rfft(s[i:i + 400] * np.hanning(400))) ** 2 + 1e-3) for i in range(0, len(s) - 400, 160)])
    v = np.concatenate([fr[i].mean(0) for i in np.array_split(np.arange(len(fr)), 8)]); v = v - v.mean()
    return dict(dur=b - a, rms=10 * np.log10(max((s ** 2).mean(), 1)), cen=float((sp * f).sum() / sp.sum())), v


def load(n): return np.fromfile(os.path.join(M, 'rec', n + '.raw'), '<i2').astype(float)


def refs():
    out = []
    for n in ('w171828', 'w172114', 'live'):
        x = load(n)
        for p in json.load(open(os.path.join(M, 'labels', n + '.json')))['prompts']:
            if p['kind'] in ('pos', 'dot') and not p['prompt'].startswith('The dot'):
                for b in p['bursts']:
                    if b['end'] - b['start'] < 0.7:
                        d, v = describe(x, b['start'], b['end']); out.append((p['kind'], n, b['start'], d, v))
    return out


def verdict(R, v, skip=None):
    d = np.array([np.linalg.norm(r[4] - v) for r in R]); k = np.array([r[0] for r in R]); m = np.ones(len(R), bool)
    if skip is not None: m[skip] = False
    return np.sort(d[m & (k == 'pos')])[:3].mean(), np.sort(d[m & (k == 'dot')])[:3].mean()


def main():
    R = refs()
    for k in ('pos', 'dot'):
        d = [r[3] for r in R if r[0] == k]
        print(f"reference {'Nova' if k == 'pos' else 'dot '} n={len(d):2d}  dur {np.mean([e['dur'] for e in d]):.2f}±{np.std([e['dur'] for e in d]):.2f}  rms {np.mean([e['rms'] for e in d]):.1f}±{np.std([e['rms'] for e in d]):.1f}  centroid {np.mean([e['cen'] for e in d]):.0f}±{np.std([e['cen'] for e in d]):.0f} Hz")
    ok = sum(('pos' if verdict(R, r[4], i)[0] < verdict(R, r[4], i)[1] else 'dot') == r[0] for i, r in enumerate(R))
    print(f'leave-one-out 3-NN on the references: {ok}/{len(R)}')
    cases = [('rdot', a, b, t) for a, b, t in ((22.55, 22.95, 'ASR said dot; engine accept 22.96'), (27.30, 27.70, 'ASR said dot; engine accept 27.76'),
                                               (32.30, 32.70, 'ASR said nova; accept 32.88'), (37.55, 37.95, 'ASR said nova; accept 38.08'))]
    cases += [('live', a, b, t) for a, b, t in ((25.9, 26.3, 'old-head dot FA 26.52 ("Dot, dot. Hello, dot.")'), (35.25, 35.65, 'old-head dot FA 35.7 ("Hello, dot")'),
                                               (7.05, 7.55, 'old-head dot FA 7.5 ("Dot, dot. Hello, dot.")'))]
    cases += [('rdot', a, b, 'known dot (prompted)') for a, b in ((13.04, 13.42), (14.12, 14.48), (15.18, 15.60))]
    for n, a, b, t in cases:
        d, v = describe(load(n), a, b); dn, dd = verdict(R, v)
        print(f"{n:5s} {a:5.2f}-{b:5.2f} dur {d['dur']:.2f} rms {d['rms']:.1f} cen {d['cen']:4.0f}  dist Nova {dn:5.1f} dot {dd:5.1f} -> {'Nova-like' if dn < dd else 'dot-like '}  [{t}]")


if __name__ == '__main__':
    main()
