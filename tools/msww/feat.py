"""Readers for pryon_extract -w files and pryon_test RESULT logs."""
import json, re
import numpy as np

CYCLE = 1280  # samples per decoder cycle (8 LFBE frames of 10 ms at 16 kHz)


def read_feat(path):
    d = open(path, 'rb').read()
    magic, ver, k, n = np.frombuffer(d[:16], '<u4')
    assert ver == 2 and k == 512 and n == 15
    rec = 1 + k + n
    a = np.frombuffer(d[16:], '<f4')
    assert len(a) % rec == 0, 'ragged: engine batches (m>1) not supported'
    a = a.reshape(-1, rec)
    return a[:, 1:1 + k].copy(), a[:, 1 + k:].copy()


def softmax(y):
    e = np.exp(y - y.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def read_results(path):
    """pryon_test stdout -> list of dicts (name, type, score, det, start, end)."""
    out, cur = [], None
    for l in open(path, errors='replace'):
        if l.startswith('RESULT'):
            m = re.search(r'keyword=(\S+) type=(\d+)', l)
            cur = {'name': m.group(1), 'type': int(m.group(2))}
        elif l.startswith('META') and cur is not None:
            j = json.loads(l[l.index('{'):])
            cur.update(score=j.get('kwClassificationScore'), thr=j.get('kwActiveThreshold'), det=j.get('kwSampleDetectionIndex'),
                       start=j.get('kwSampleStartIndex'), end=j.get('kwSampleEndIndex'))
            out.append(cur); cur = None
    return out
