#!/usr/bin/env python3
"""Labels from the ACTUAL audio of Ben's captures (not prompt timing).

1. speech bursts by energy (bursts.py); each gets a spectral centroid.
2. a Dot reply is a burst with centroid < 570 Hz, or a long loud one (>=1.4 s and >=74 dBFS: it can hide a word said over it) -> 'reply'
   (negative for training, never a positive or a dot label).
3. the remaining bursts are Ben's. They are matched to the prompt list by ORDER: the bursts that start in [t_i + 0.5, t_{i+1} + 0.5)
   belong to prompt i. A prompt expecting n words (count of dot/Nova words in the phrase) that gets another number of bursts is flagged.
usage: labels.py OUT.json    (writes build/msww/labels/<name>.json for each capture and prints a table)"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from bursts import bursts

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
B = os.path.join(ROOT, 'build')
CAPS = {'w171828': ('wake-check-20261112-171828', 'msww/rec/w171828.raw'), 'w172114': ('wake-check-20261112-172114', 'msww/rec/w172114.raw'),
        'live': ('live-wake-check-20261112-175530', 'msww/rec/live.raw')}


def centroid(x, a, b):
    s = x[int(a * 16000):int(b * 16000)].astype(float)
    sp = np.abs(np.fft.rfft(s * np.hanning(len(s)))); f = np.fft.rfftfreq(len(s), 1 / 16000)
    return float((sp * f).sum() / max(sp.sum(), 1e-9))


def expected_words(phrase):
    return len(re.findall(r'\b(dot|nova)\b', phrase.lower())) or None


def frames(x, fr=800):
    n = len(x) // fr
    e = np.zeros(n); c = np.zeros(n); f = np.fft.rfftfreq(fr, 1 / 16000); w = np.hanning(fr)
    for i in range(n):
        s = x[i * fr:(i + 1) * fr].astype(float)
        e[i] = 10 * np.log10(max((s ** 2).mean(), 1.0))
        sp = np.abs(np.fft.rfft(s * w)); c[i] = (sp * f).sum() / max(sp.sum(), 1e-9)
    return e, c, fr / 16000.0


def split(x, margin=14.0, dil=0.2):
    """Frame-level separation. A Dot-reply frame is loud (>= 66 dBFS) with a low centroid (< 700 Hz); frames within `dil` s of one are
    reply too. What is left is cut into Ben's bursts (energy > 20th percentile + margin, gaps <= 0.3 s merged)."""
    e, c, dt = frames(x)
    rep = (e >= 66) & (c < 700)
    k = int(round(dil / dt))
    rep = np.convolve(rep.astype(float), np.ones(2 * k + 1), 'same') > 0
    on = (e > np.percentile(e, 20) + margin) & ~rep
    segs = lambda m: _runs(m, dt)
    return segs(on), segs(rep), (e, c, dt)


def _runs(m, dt, merge=0.3, min_s=0.12):
    out, i, n = [], 0, len(m)
    g = int(round(merge / dt))
    while i < n:
        if m[i]:
            j = i
            while j < n and (m[j] or m[j:j + g].any()): j += 1
            while not m[j - 1]: j -= 1
            out.append((i * dt, j * dt))
            i = j
        else:
            i += 1
    return [(a, b) for a, b in out if b - a >= min_s]


# Prompts whose order matching is ambiguous, resolved by reading the frame-level timeline (energy/centroid) and the Dot's own reply
# bursts: which burst of the prompt is Ben's word. Everything else of that prompt's bursts is a tail of a Dot reply or a stray (ignored).
# (capture, prompt time) -> start times of Ben's word bursts
KEEP = {('w171828', 34.1): [36.55], ('w171828', 49.1): [50.55], ('w172114', 9.1): [10.1],
        ('live', 4.1): [5.55, 6.25, 7.05], ('live', 11.1): [13.05], ('live', 42.1): [46.1], ('live', 60.1): [64.95, 65.75],
        ('live', 67.1): [70.4]}


def label(name):
    d, rel = CAPS[name]
    prompts = json.load(open(os.path.join(B, d, 'prompts.json')))
    x = np.fromfile(os.path.join(B, rel), '<i2')
    ben, rep, (e, c, dt) = split(x)
    bs = [dict(start=a, end=b, peak=float(e[int(a / dt):int(b / dt) + 1].max()), cen=centroid(x, a, b), kind='ben') for a, b in ben]
    rs = [dict(start=a, end=b, kind='reply') for a, b in rep]
    ts = [p['seconds'] for p in prompts] + [1e9]
    out = []
    for i, p in enumerate(prompts):
        mine = [b for b in bs if ts[i] + 0.5 <= b['start'] < ts[i + 1] + 0.5]
        raw_flag = ''
        if (name, round(p['seconds'], 1)) in KEEP:
            keep = KEEP[(name, round(p['seconds'], 1))]
            extra = [b for b in mine if not any(abs(b['start'] - k) < 0.06 for k in keep)]
            mine = [b for b in mine if any(abs(b['start'] - k) < 0.06 for k in keep)]
            raw_flag = f'resolved by timeline: {len(extra)} reply-tail/stray burst(s) at {[round(b["start"],2) for b in extra]} set aside'
            for b in extra: b['kind'] = 'stray'
        ph = p['phrase']
        kind = 'pos' if ph.lower().startswith('nova') else 'dot' if 'dot' in ph.lower() else 'speech'
        want = expected_words(ph)
        flag = '' if want is None or want <= len(mine) or raw_flag else f'MISMATCH expected {want} words, found {len(mine)} bursts'
        if not mine: flag = 'NO SPEECH FOUND'
        out.append(dict(prompt=ph, kind=kind, t_prompt=p['seconds'], bursts=mine, flag=flag, note=raw_flag))
    strays = [b for b in bs if b.get('kind') == 'stray']
    return dict(name=name, prompts=out, bursts=bs, replies=rs, strays=strays)


if __name__ == '__main__':
    os.makedirs(os.path.join(B, 'msww', 'labels'), exist_ok=True)
    for n in CAPS:
        r = label(n)
        json.dump(r, open(os.path.join(B, 'msww', 'labels', n + '.json'), 'w'), indent=1)
        print('==', n)
        print('   replies:', [(round(b['start'],2), round(b['end'],2)) for b in r['replies']])
        for p in r['prompts']:
            print(f"  {p['t_prompt']:5.1f} {p['kind']:6s} {p['prompt'][:28]:28s} -> {[(round(b['start'],2), round(b['end'],2)) for b in p['bursts']]} {p['flag']} {p['note']}")
