"""Streams for training and measuring an msww row.  Personal and Amazon-derived data are read from build/msww (ignored by git).

Every stream is a continuous recording run through the stock engine once (pryon_extract -w): per decoder cycle (80 ms) the 512-d
input X of kernel/Dense:20 and the engine's 15 logits Y.  Two families:

  corpus  17 TTS voice streams (testdata/corpus, every clip of the voice that checked out: bare commands, wake word + command, wake word
          alone, "stop" kinds; Amazon's own words, hey_jarvis, hey_mycroft, okay_nabu).  All negative for a new word.
          Voices are split ONCE, by name, into HELD (never used for fitting, mining or choosing thresholds) and DEV.
  ben     Ben's recordings with windows from the ACTUAL audio (labels.py): Nova words (pos), dot phrases, other speech, Dot replies
          (the speaker's own TTS, negative), strays/ignored.
"""
import json, os
import numpy as np
from feat import read_feat, CYCLE
import rowfit as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
M = os.path.join(ROOT, 'build', 'msww')
DT = CYCLE / 16000.0   # 0.08 s

CORPUS = sorted(f[:-5] for f in os.listdir(os.path.join(M, 'full')) if f.endswith('.feat')) if os.path.isdir(os.path.join(M, 'full')) else []
HELD = CORPUS[::3]                       # fixed rule, decided before any model was fitted
DEV = [v for v in CORPUS if v not in HELD]
BEN = ['w171828', 'w172114', 'live', 'rdot']


def cyc_end(n):
    return (np.arange(n) + 1) * DT


def _stream(X, Y, k, name, **kw):
    return dict(name=name, X=X, Y=Y.astype(np.float64), L=F.base_logit(Y.astype(np.float64), k), t=cyc_end(len(X)), **kw)


def ben_windows(name):
    """-> dict(pos=[(a,b)], dot=[...], speech=[...], reply=[...], ignore=[...]) seconds."""
    if name == 'rdot':
        from bursts import bursts
        raw = os.path.join(M, 'rec', 'rdot.raw')
        dots = json.load(open(os.path.join(ROOT, 'build', 'false-wake-reopen', 'repeated-dot.json')))
        d = dict(pos=[], dot=[(x['seconds'] - 0.2, x['seconds'] + 0.7) for x in dots], speech=[], reply=[], ignore=[])
        last = max(x['seconds'] for x in dots)
        x = np.fromfile(raw, '<i2')
        sub = _subbursts(x)
        # after the dots the recording holds 3.0 s blocks (Nova, then a command): the first sub-burst is the word, later ones the command
        for a, b, _ in [t for t in bursts(raw) if t[0] > last + 0.5]:
            inside = [(s0, e0) for s0, e0 in sub if a - 0.05 <= s0 and e0 <= b + 0.05]
            if inside:
                d['pos'].append((inside[0][0] - 0.3, inside[0][1] + 0.7))
                d['ignore'] += [(s0, e0 + 0.4) for s0, e0 in inside[1:]]
        return d
    r = json.load(open(os.path.join(M, 'labels', name + '.json')))
    d = dict(pos=[], dot=[], speech=[], reply=[(b['start'] - 0.1, b['end'] + 0.3) for b in r['replies']],
             ignore=[(b['start'] - 0.1, b['end'] + 0.3) for b in r.get('strays', [])])
    for p in r['prompts']:
        if not p['bursts']:
            continue
        a, b = min(x['start'] for x in p['bursts']), max(x['end'] for x in p['bursts'])
        d[{'pos': 'pos', 'dot': 'dot', 'speech': 'speech'}[p['kind']]].append((a - 0.3, b + 0.7))
    return d


def _subbursts(x, fr=800, db=18.0, merge=0.1):
    n = len(x) // fr
    e = 10 * np.log10(np.maximum((x[:n * fr].astype(float).reshape(n, fr) ** 2).mean(1), 1.0))
    on = e > max(np.percentile(e, 20), 20.0) + db
    out, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or on[j:j + 2].any()): j += 1
            out.append((i * fr / 16000, j * fr / 16000)); i = j
        else:
            i += 1
    return [(a, b) for a, b in out if b - a >= 0.1]


def load_ben(k):
    out = {}
    for n in BEN:
        X, Y = read_feat(os.path.join(M, 'feat', n + '.feat'))
        out[n] = _stream(X, Y, k, n, win=ben_windows(n), kind='ben')
    return out


def load_aug(k):
    """Augmented copies of each recording's Nova words (augment.py).  Only positives: the rest of the stream is ignored."""
    out = {}
    for n in BEN:
        p = os.path.join(M, 'augfeat', n + '.feat')
        if not os.path.exists(p):
            continue
        X, Y = read_feat(p)
        idx = json.load(open(os.path.join(M, 'aug', n + '.json')))
        t_end = len(X) * DT
        out['aug_' + n] = _stream(X, Y, k, 'aug_' + n, kind='aug', idx=idx,
                                  win=dict(pos=[(c['word'][0] - 0.3, c['word'][1] + 0.7) for c in idx], dot=[], speech=[], reply=[], ignore=[(0, t_end)]))
    return out


def load_corpus(k, names=None):
    out = {}
    for n in (names or CORPUS):
        X, Y = read_feat(os.path.join(M, 'full', n + '.feat'))
        out[n] = _stream(X, Y, k, n, win=dict(pos=[], dot=[], speech=[], reply=[], ignore=[]), kind='corpus')
    return out


def in_win(t, wins):
    m = np.zeros(len(t), bool)
    for a, b in wins:
        m |= (t >= a) & (t < b)
    return m


def hours(streams):
    return sum(len(s['X']) for s in streams.values()) * DT / 3600.0
