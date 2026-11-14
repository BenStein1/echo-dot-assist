"""Explicit per-word inputs for host MSWW generation (not the on-device UI).

Paths are relative to the manifest. Windows are seconds in the same padded
continuous audio passed to pryon_extract -w. Split by original recording, never
by augmented clip: this entry point deliberately accepts no augmentation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import ds
from feat import read_feat


def load(path, k=12):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    word = spec['word']
    if not isinstance(word, str) or not word or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in word):
        raise ValueError('word must be an uppercase keyword identifier')
    streams, splits, raw = {}, {s: [] for s in ('train', 'dev', 'held')}, {}
    seen = set()
    for item in spec['recordings']:
        name, split = item['name'], item['split']
        if not name or name in streams or Path(name).name != name or split not in splits:
            raise ValueError('recording names must be unique basenames; split must be train/dev/held')
        feat = (path.parent / item['features']).resolve()
        audio = (path.parent / item['raw']).resolve()
        if audio.stem != name or audio.suffix != '.raw':
            raise ValueError('raw audio must be named <recording name>.raw')
        # Catch identical recordings under different filenames across splits.
        digest = hashlib.sha256(audio.read_bytes()).digest()
        if digest in seen:
            raise ValueError('duplicate audio: split by original recording')
        seen.add(digest)
        X, Y = read_feat(feat)
        if not len(X) or not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError('empty or non-finite features')
        duration = audio.stat().st_size / 32000
        if abs(duration - len(X) * ds.DT) > ds.DT * 2:
            raise ValueError('audio/features duration mismatch')
        win = {key: item.get('windows', {}).get(key, []) for key in ('pos', 'dot', 'speech', 'reply', 'ignore')}
        for windows in win.values():
            for a, b in windows:
                if not (0 <= a < b <= duration):
                    raise ValueError('window outside recording')
        if split == 'dev' and win['pos']:
            raise ValueError('dev is the negative corpus, not positive enrollment audio')
        streams[name] = ds._stream(X, Y, k, name, win=win, kind='corpus' if split == 'dev' else 'ben')
        splits[split].append(name)
        raw[name] = str(audio)
    if len(splits['train']) < 2 or not splits['dev'] or not splits['held']:
        raise ValueError('need at least two train recordings, negative dev audio and independent held audio')
    if any(not streams[n]['win']['pos'] for n in splits['train']):
        raise ValueError('each train recording needs positive windows')
    if not any(streams[n]['win']['pos'] for n in splits['held']):
        raise ValueError('held recordings need positive windows as well as negative audio')
    if not any((~(ds.in_win(streams[n]['t'], streams[n]['win']['pos']) |
                    ds.in_win(streams[n]['t'], streams[n]['win']['ignore']))).any() for n in splits['held']):
        raise ValueError('held recordings need negative audio outside positive/ignored windows')
    return word, streams, splits, raw
