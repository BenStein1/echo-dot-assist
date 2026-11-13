#!/usr/bin/env python3
"""Continuous replay gate: run a packaged model set through the REAL engine (pryon_test under qemu-arm, the Dot's own library) over raw
16 kHz s16le streams at several times real time, and read every RESULT of the word back.

  gate.py run SET_DIR OUT_DIR STREAM.raw...   [-j N] [-x SPEED]     -> OUT_DIR/<stream>.res (pryon_test stdout)
  python: events(res_path, word) -> [(time_s, score, type)]

Needs build/donut/pryon_test (make DEVICE=donut), qemu-arm, bwrap and the unpacked firmware (as extract.sh).  Nothing is sent to the Dot."""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from feat import read_results

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def run_one(set_dir, out_dir, raw, speed=3, timeout=3000):
    name = os.path.basename(raw)[:-4]
    out = os.path.join(out_dir, name + '.res')
    if os.path.exists(out):
        return out
    cmd = ['timeout', '-s', 'KILL', str(timeout), 'bwrap', '--ro-bind', '/', '/', '--bind', '/tmp', '/tmp', '--bind', ROOT + '/build', ROOT + '/build',
           '--dev', '/dev', '--proc', '/proc', '--unshare-pid', '--die-with-parent', '/usr/bin/qemu-arm', '-L', ROOT + '/firmware/donut/rootfs',
           '-E', 'LD_LIBRARY_PATH=/system/lib:/system/vendor/lib', ROOT + '/build/donut/pryon_test', '-m', os.path.join(set_dir, 'pryon.manifest'),
           '-x', str(speed), raw]
    with open(out + '.part', 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    os.rename(out + '.part', out)
    return out


def run(set_dir, out_dir, raws, jobs=12, speed=3):
    os.makedirs(out_dir, exist_ok=True)
    with ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(lambda r: run_one(set_dir, out_dir, r, speed), raws))


def events(res_path, word):
    """RESULT lines of `word` -> [(detection time s, score, type)]; type 2 = accept, 0 = near miss."""
    return [(r['det'] / 16000.0, r['score'], r['type']) for r in read_results(res_path) if r['name'].upper() == word.upper()]


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or a[0] != 'run' or len(a) < 4:
        sys.exit(__doc__)
    j, x = 12, 3
    args = []
    i = 1
    while i < len(a):
        if a[i] == '-j': j = int(a[i + 1]); i += 2
        elif a[i] == '-x': x = float(a[i + 1]); i += 2
        else: args.append(a[i]); i += 1
    for p in run(args[0], args[1], args[2:], j, x):
        print(p)
