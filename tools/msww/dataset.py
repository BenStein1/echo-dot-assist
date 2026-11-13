"""Labelled recordings for the msww feasibility study. Reads personal recordings from build/ (never from git).

A recording has windows: (start_s, end_s, kind, text) with kind 'pos' (Nova said), 'dot' (dot-like phrases), 'speech' (other speech).
Everything outside a window is unlabelled (silence, replies of the Dot, repeats): negative for training, counted separately in tests.
"""
import json, os
import numpy as np
from feat import read_feat, softmax, CYCLE
from bursts import bursts

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
B = os.path.join(ROOT, 'build')
M = os.path.join(B, 'msww')


def _prompts(d, lead=0.3, span=4.8):
    ps = json.load(open(os.path.join(B, d, 'prompts.json')))
    out = []
    for p in ps:
        ph = p['phrase']
        kind = 'pos' if ph.lower().startswith('nova') else 'dot' if 'dot' in ph.lower() else 'speech'
        out.append((p['seconds'] + lead, p['seconds'] + span, kind, ph))
    return out


def windows():
    w = {}
    w['w171828'] = _prompts('wake-check-20261112-171828')
    w['w172114'] = _prompts('wake-check-20261112-172114')
    w['live'] = _prompts('live-wake-check-20261112-175530', span=6.8)
    raw = os.path.join(M, 'rec', 'rdot.raw')
    bs = bursts(raw)
    dots = json.load(open(os.path.join(B, 'false-wake-reopen', 'repeated-dot.json')))
    last = max(d['seconds'] for d in dots)
    neg, pos = [], []
    for a, b, _ in bs:
        if a <= last + 0.5:
            neg.append([a, b])
        else:
            pos.append([a, b])
    # merge dot bursts of one group (gap < 1.5 s)
    grp = []
    for a, b in neg:
        if grp and a - grp[-1][1] < 1.5: grp[-1][1] = b
        else: grp.append([a, b])
    w['rdot'] = [(a - 0.2, b + 0.3, 'dot', 'repeated dot') for a, b in grp] + \
                [(a - 0.3, a + 3.3, 'pos', 'Nova (+command)') for a, b in pos if b - a > 1.0]
    w['nova1'] = [(0.0, 3.0, 'pos', 'Nova')]
    w['dot1'] = [(0.0, 3.0, 'dot', 'dot')]
    return w


def load(name, k=None):
    """-> X[T,512], Y[T,15] (stock engine's logits), cycle end times (s)."""
    X, Y = read_feat(os.path.join(M, 'feat', name + '.feat') if name in windows_names else os.path.join(M, 'tts', name + '.feat'))
    return X, Y.astype(np.float64), (np.arange(len(X)) + 1) * CYCLE / 16000.0

windows_names = ['w171828', 'w172114', 'live', 'rdot', 'nova1', 'dot1']
