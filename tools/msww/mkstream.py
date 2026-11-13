#!/usr/bin/env python3
"""Concatenate wav/raw clips into one 16 kHz s16le stream with silence between them; write a JSON index of start/end samples.
usage: mkstream.py OUT.raw OUT.json GAP_SECONDS clip...   (a clip is label=path)"""
import sys, json, wave
import numpy as np

def load(path):
    if path.endswith('.wav'):
        with wave.open(path) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1
            return np.frombuffer(w.readframes(w.getnframes()), '<i2')
    return np.fromfile(path, '<i2')

def main():
    out, idx, gap = sys.argv[1], sys.argv[2], float(sys.argv[3])
    parts, index, pos = [np.zeros(int(gap * 16000), '<i2')], [], int(gap * 16000)
    for a in sys.argv[4:]:
        label, path = a.split('=', 1)
        x = load(path)
        index.append({'label': label, 'path': path, 'start': pos, 'end': pos + len(x)})
        parts += [x, np.zeros(int(gap * 16000), '<i2')]
        pos += len(x) + int(gap * 16000)
    np.concatenate(parts).tofile(out)
    json.dump(index, open(idx, 'w'), indent=1)

if __name__ == '__main__':
    main()
