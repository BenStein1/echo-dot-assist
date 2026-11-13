#!/usr/bin/env python3
"""Augmented copies of the word windows of Ben's recordings (gain, speed, background noise), concatenated per source recording into
build/msww/aug/<rec>.raw (+ .json with the word window of every clip), for pryon_extract.

Why: the three Nova words the first row missed in every fold were the three quietest (55-57 dBFS peak).  A user's enrollment is a few
utterances, so the trainer must see the same word at other levels, speeds and noise.  Each source recording yields its own stream so that a
held-out recording is never trained on (leave-one-recording-out stays honest).

  augment.py [--gains -8,-5,-2.5,3] [--speeds 0.92,1.0,1.08] [--noise 0,18]
"""
import argparse, json, os, sys
import numpy as np
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(__file__))
import ds

OUT = os.path.join(ds.M, 'aug')
PRE, POST = 1.6, 0.45          # context before the window start / after the word end, seconds
GAP = 2.2


def ambient(x, n):
    """n samples of the recording's own background: its quietest 1 s repeated."""
    fr = 16000
    k = len(x) // fr
    e = [np.mean(x[i * fr:(i + 1) * fr].astype(float) ** 2) for i in range(k)]
    i = int(np.argmin(e))
    seg = x[i * fr:(i + 1) * fr].astype(float)
    return np.tile(seg, n // fr + 1)[:n], np.sqrt(np.mean(seg ** 2))


def build(rec, gains, speeds, noises):
    raw = os.path.join(ds.M, 'rec', rec + '.raw')
    x = np.fromfile(raw, '<i2')
    win = ds.ben_windows(rec)['pos']
    parts, index, pos = [], [], 0
    amb, arms = ambient(x, int(GAP * 16000))
    parts.append(amb[:int(GAP * 16000)]); pos += len(parts[-1])
    for wi, (a, b) in enumerate(win):
        s0, s1 = int(max(0, a - PRE) * 16000), int((b - 0.25) * 16000)   # a = word start - 0.3, b = word end + 0.7  -> end at word end + 0.45
        clip = x[s0:s1].astype(float)
        word_a, word_b = (a + 0.3 - max(0, a - PRE)), (b - 0.7 - max(0, a - PRE))     # word position inside the clip
        for sp in speeds:
            c = clip if sp == 1.0 else resample_poly(clip, 100, int(round(100 * sp)))
            wa, wb = word_a / sp, word_b / sp
            for g in gains + [0.0]:
                for snr in noises:
                    y = c * 10 ** (g / 20)
                    if snr:
                        n = ambient(x, len(y))[0]
                        n = n / (np.sqrt(np.mean(n ** 2)) + 1e-9) * np.sqrt(np.mean(y ** 2)) * 10 ** (-snr / 20)
                        y = y + n
                    if sp == 1.0 and g == 0.0 and not snr and False:
                        pass
                    y = np.clip(np.rint(y), -32768, 32767).astype('<i2')
                    index.append(dict(src=wi, gain=g, speed=sp, snr=snr, start=pos / 16000.0, word=[pos / 16000.0 + wa, pos / 16000.0 + wb]))
                    parts.append(y); pos += len(y)
                    gap = ambient(x, int(GAP * 16000))[0].astype('<i2'); parts.append(gap); pos += len(gap)
    os.makedirs(OUT, exist_ok=True)
    np.concatenate(parts).astype('<i2').tofile(os.path.join(OUT, rec + '.raw'))
    json.dump(index, open(os.path.join(OUT, rec + '.json'), 'w'), indent=1)
    return len(index), pos / 16000.0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gains', default='-8,-5,-2.5,3'); ap.add_argument('--speeds', default='0.92,1.0,1.08'); ap.add_argument('--noise', default='0,18')
    a = ap.parse_args()
    g = [float(v) for v in a.gains.split(',')]; sp = [float(v) for v in a.speeds.split(',')]; nz = [float(v) for v in a.noise.split(',')]
    for r in ds.BEN:
        print(r, build(r, g, sp, nz))
