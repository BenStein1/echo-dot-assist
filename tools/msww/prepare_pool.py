#!/usr/bin/env python3
"""Prepare private, phrase/model-bound full MSWW fitting/calibration pools.

Input is an array of negative feature rows: path, label=0, split=train|dev,
text (accurate transcript), speaker. Split original speakers BEFORE extraction;
never put a fitting voice into calibration. No positive or final-test input is
accepted. Outputs contain private/Amazon-derived features: do not commit them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np
from ondev_check import write_pool
import ds


def normalized(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def prepare(model, manifest, output, word):
    model, manifest, output = map(Path, (model, manifest, output))
    phrase = normalized(word)
    if not phrase or len(word.encode()) >= 128:
        raise ValueError('need a short phrase')
    rows = json.loads(manifest.read_text())
    parts = {s: [] for s in ('train', 'dev')}
    speakers = {s: set() for s in parts}
    seen = set()
    for row in rows:
        if row.get('label') != 0 or row.get('split') not in parts:
            raise ValueError('pools accept only negative train/dev recordings')
        text, speaker = row.get('text'), row.get('speaker')
        if not isinstance(text, str) or not text.strip() or not isinstance(speaker, str) or not speaker or len(speaker.encode()) >= 128:
            raise ValueError('each original recording needs a transcript and speaker')
        if f' {phrase} ' in f' {normalized(text)} ':
            raise ValueError('negative transcript contains the requested phrase')
        path = manifest.parent / row['path']
        data=path.read_bytes()
        if len(data)<16+528*4 or (len(data)-16)%(528*4) or struct.unpack('<4I',data[:16])!=(0x46595250,2,512,15):
            raise ValueError('invalid MSWW feature header/length')
        digest = hashlib.sha256(data).digest()
        if digest in seen:
            raise ValueError('duplicate original features')
        seen.add(digest)
        records=np.frombuffer(data[16:],'<f4').reshape(-1,528)
        x,y=records[:,1:513].copy(),records[:,513:].copy()
        if not len(x) or not np.isfinite(records).all() or not np.isfinite(x).all() or not np.isfinite(y).all() or np.abs(x).max() > 1e6 or np.abs(y).max() > 1e6:
            raise ValueError('empty or invalid features')
        split = row['split']
        speakers[split].add(speaker)
        indices=np.arange(len(x));keep=np.ones(len(x),bool)
        for a,b in row.get('excluded_windows',[]):
            if not np.isfinite([a,b]).all() or not 0<=a<b<=len(x)*ds.DT:
                raise ValueError('excluded window outside original recording')
            keep &= ~(((indices+1)*ds.DT>=a)&((indices+1)*ds.DT<=b))
        indices=indices[keep]
        if not len(indices):
            raise ValueError('no negative cycles remain')
        stream=ds._stream(x[keep],y[keep],12,str(path))
        stream['indices']=indices
        stream['samples']=records[keep,0]
        parts[split].append(stream)
    if not all(parts.values()) or speakers['train'] & speakers['dev']:
        raise ValueError('need disjoint fitting/calibration speakers')
    if any(sum(len(s['X']) for s in p) > 100000 for p in parts.values()):
        raise ValueError('pool exceeds 100000-cycle device budget')
    output.mkdir(mode=0o700)  # Never overwrite installed or earlier inputs.
    result = dict(version=1, word=word, source_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  fitting_speakers=sorted(speakers['train']), calibration_speakers=sorted(speakers['dev']))
    for split, name in [('train', 'fitting_pool'), ('dev', 'calibration_pool')]:
        streams = parts[split]
        seq = []
        for s in streams:
            q = np.full(len(s['X']), 2, np.uint8)
            q[0] = 1
            q[1:][(np.diff(s['indices'])!=1)|(np.diff(s['samples'])!=1280)]=1
            seq.append(q)
        path = output / f'{split}.pool'
        x = np.concatenate([s['X'] for s in streams])
        write_pool(path, x, np.concatenate([s['L'] for s in streams]), np.ones(len(x)), np.concatenate(seq))
        path.chmod(0o600)
        result[name] = dict(path=path.name, cycles=len(x), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'manifest.json').chmod(0o600)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stock_onnx')
    p.add_argument('feature_manifest')
    p.add_argument('output')
    p.add_argument('word')
    a = p.parse_args()
    print(json.dumps(prepare(a.stock_onnx, a.feature_manifest, a.output, a.word), indent=2))
