#!/usr/bin/env python3
"""Pack a model set whose multi-wakeword decoder carries one edited row of kernel/Dense:20 ([15,512] int8 HybridGemm head).

Library:  pack(src_set, dst_set, class_index, row, word, spotter)  -> copies a stock msww set (e.g. computer-en-US) and
          (1) writes the row (int8[512], scale, zero point, sum, bias) into the ONNX decoder,
          (2) replaces the spotter named `replace` with a spotter `word` reading class `class_index`.
CLI:      msww_pack.py SRC DST CLASS WORD --copy-row FROM_CLASS      (row copied from another class: loading experiment)
                                                                   --zero                    (row made to output nothing)
Nothing of Amazon's is stored in this repository; SRC is a set pulled from a device into build/.
"""
import copy, json, os, shutil, sys
import numpy as np
try:
    import onnx
    from onnx import numpy_helper as nh
except ImportError:      # fitting needs scipy (system python), packing needs onnx (build/pryon-review-venv): import lazily
    onnx = nh = None

ONNX = 'multi-wakeword-wakeworddecoder/odyssey_msww.streaming.fixed.int8.onnx'
ROW = {'kernel': 'kernel/Dense:20', 's': 'kernel/Dense:20_s', 'zp': 'kernel/Dense:20_zp', 'sums': 'kernel/Dense:20_sums',
       'bias': 'bias/Dense:20'}


def quantize_row(w):
    """float weights[512] -> (int8[512], scale, zero point, sum): the same asymmetric per-row scheme the head uses."""
    w = np.asarray(w, np.float64)
    lo, hi = min(w.min(), 0.0), max(w.max(), 0.0)
    s = max((hi - lo) / 255.0, 1e-8)
    z = float(np.clip(np.rint(-128 - lo / s), -128, 127))
    q = np.clip(np.rint(w / s + z), -128, 127).astype(np.int8)
    return q, np.float32(s), np.int32(z), np.int32(q.astype(np.int64).sum())


def read_rows(path):
    m = onnx.load(path)
    ini = {i.name: nh.to_array(i) for i in m.graph.initializer}
    return {k: ini[v] for k, v in ROW.items()}


def single_word_configs(kw, op, word):
    """Keep competing network classes, but expose only this word to the daemon.

    wake_pryon forwards every accepted keyword, not only the directory's name.
    Removing only operation rules is insufficient: the other spotters still emit.
    """
    spotters = [s for s in kw['keyword-orchestra']['keyword-spotters'] if s['name'] == word]
    if len(spotters) != 1:
        raise ValueError('single-word package needs exactly one target spotter: ' + word)
    kw['keyword-orchestra']['keyword-spotters'] = spotters
    for state in op.values():
        if isinstance(state, dict) and 'rules' in state:
            state['rules'] = [r for r in state['rules'] if r['name'] == word]
    return kw, op


def pack(src, dst, class_index, row, word, replace='COMPUTER', accept=0.523, notify=0.12, decay=0.6, lock=100, kill=(), single_word=False):
    """row: dict(q=int8[512], s=float, zp=int, sums=int, bias=float). kill: other classes whose row is silenced (logit -20),
    so that they do not compete in the softmax with the new word."""
    if os.path.exists(dst):
        raise SystemExit('refusing to overwrite ' + dst)
    shutil.copytree(src, dst)
    path = os.path.join(dst, ONNX)
    m = onnx.load(path)
    by = {i.name: i for i in m.graph.initializer}
    def put(name, idx, val):
        a = nh.to_array(by[name]).copy()
        if a.ndim == 2 and name != ROW['bias']:
            a[idx] = val
        elif name == ROW['bias']:
            a[0, idx] = val
        else:
            a[idx] = val
        by[name].CopyFrom(nh.from_array(a, name))
    for k in kill:
        put(ROW['kernel'], k, np.zeros(512, np.int8)); put(ROW['s'], k, np.float32(1e-8)); put(ROW['zp'], k, np.int32(0))
        put(ROW['sums'], k, np.int32(0)); put(ROW['bias'], k, np.float32(-20))
    put(ROW['kernel'], class_index, np.asarray(row['q'], np.int8))
    put(ROW['s'], class_index, np.float32(row['s']))
    put(ROW['zp'], class_index, np.int32(row['zp']))
    put(ROW['sums'], class_index, np.int32(row['sums']))
    put(ROW['bias'], class_index, np.float32(row['bias']))
    onnx.save(m, path)
    # spotter + state machine + words
    kwp = os.path.join(dst, 'BDPGeneratedFiles/kw.cfg.json')
    kw = json.load(open(kwp))
    for sp in kw['keyword-orchestra']['keyword-spotters']:
        if sp['name'] == replace:
            sp['name'] = word
            sp['spotter']['word-component-array'][0]['scorer-index'] = class_index
            sp['classifier']['decay-rate'] = decay
            th = sp['spotter']['classification-thresholds']
            th['accept-threshold'] = accept; th['notify-threshold'] = notify
            for o in th.get('overrides', []):
                o['accept-threshold'] = accept
            sp['spotter']['lock-period'] = lock
    for f in ['BDPGeneratedFiles/op.cfg.json']:
        p = os.path.join(dst, f)
        t = open(p).read().replace('"%s"' % replace, '"%s"' % word)
        op = json.loads(t)
        if single_word:
            kw, op = single_word_configs(kw, op, word)
        json.dump(op, open(p, 'w'), indent=2)
    json.dump(kw, open(kwp, 'w'), indent=2)
    for f in ['BDPGeneratedFiles/words.shrunk.txt', 'encoder/words.shrunk.txt']:
        p = os.path.join(dst, f)
        if os.path.exists(p):
            t = open(p).read().replace(replace, word)
            open(p, 'w').write(t)


def main():
    a = sys.argv[1:]
    src, dst, cls, word = a[0], a[1], int(a[2]), a[3]
    r = read_rows(os.path.join(src, ONNX))
    if '--copy-row' in a:
        c = int(a[a.index('--copy-row') + 1])
        row = dict(q=r['kernel'][c], s=r['s'][c], zp=r['zp'][c], sums=r['sums'][c], bias=r['bias'][0, c])
    else:
        row = dict(q=np.zeros(512, np.int8), s=np.float32(1e-8), zp=np.int32(0), sums=np.int32(0), bias=np.float32(-20))
    kill = [int(x) for x in a[a.index('--kill') + 1].split(',')] if '--kill' in a else ()
    pack(src, dst, cls, row, word, kill=kill)


if __name__ == '__main__':
    main()
