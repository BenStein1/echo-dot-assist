#!/usr/bin/env python3
"""Continuous replay gate: run a packaged model set through the REAL engine (pryon_test under qemu-arm, the Dot's own library) over raw
16 kHz s16le streams at several times real time, and read every RESULT of the word back.

  gate.py run SET_DIR OUT_DIR STREAM.raw...   [-j N] [-x SPEED]     -> OUT_DIR/<stream>.res (pryon_test stdout)
  python: events(res_path, word) -> [(time_s, score, type)]

Needs build/donut/pryon_test (make DEVICE=donut), qemu-arm, bwrap and the unpacked firmware (as extract.sh).  Nothing is sent to the Dot."""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from feat import read_results
import json, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def audio_loss(err, result_path):
    """Reject actual loss, not config names, queued latency, or skipped-zero shutdowns."""
    for line in err.splitlines():
        if re.search(r'\b(?:dropped|skipped)\s+[1-9]\d*\b', line, re.I):
            return True
    with open(result_path, errors='replace') as f:
        for line in f:
            if not line.startswith('META'):
                continue
            meta = json.loads(line[line.index('{'):])
            for section, keys in (
                ('deviceMetrics', ('skipped_audio_backlog_since_last_ww_msec', 'sum_client_sample_index_gaps_since_last_ww')),
                ('audioMetadataDuringDetection', ('dropFramesCount', 'largestFrameDropedAtOnce')),
            ):
                if any(meta.get(section, {}).get(key, 0) != 0 for key in keys):
                    return True
    return False


def _fingerprint(set_dir, raw, speed):
    """What a cached result is bound to: every file of the set, the input, the speed."""
    import hashlib
    h = hashlib.sha256()
    h.update(b'replay-validation-v2')
    h.update(hashlib.sha256(open(ROOT + '/build/donut/pryon_test', 'rb').read()).digest())
    for root, _, files in sorted(os.walk(set_dir)):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, set_dir).encode()); h.update(hashlib.sha256(open(p, 'rb').read()).digest())
    st = os.stat(raw)
    h.update(('%s|%d|%d|%s' % (os.path.abspath(raw), st.st_size, st.st_mtime_ns, speed)).encode())
    return h.hexdigest()


def run_one(set_dir, out_dir, raw, speed=3, timeout=3000):
    """A result counts only if the engine exited 0 (events) or 3 (no events), and its own log says it pushed EVERY sample (no dropped audio)."""
    import re, json
    name = os.path.basename(raw)[:-4]
    out = os.path.join(out_dir, name + '.res')
    fp = _fingerprint(set_dir, raw, speed)
    meta = out + '.json'
    if os.path.exists(out) and os.path.exists(meta) and json.load(open(meta)).get('fingerprint') == fp:
        return out
    cmd = ['timeout', '-s', 'KILL', str(timeout), 'bwrap', '--ro-bind', '/', '/', '--bind', '/tmp', '/tmp', '--bind', ROOT + '/build', ROOT + '/build',
           '--dev', '/dev', '--proc', '/proc', '--unshare-pid', '--die-with-parent', '/usr/bin/qemu-arm', '-L', ROOT + '/firmware/donut/rootfs',
           '-E', 'LD_LIBRARY_PATH=/system/lib:/system/vendor/lib', ROOT + '/build/donut/pryon_test', '-m', os.path.join(set_dir, 'pryon.manifest'),
           '-x', str(speed), raw]
    with open(out + '.part', 'w') as f, open(out + '.err', 'w') as e:
        rc = subprocess.run(cmd, stdout=f, stderr=e).returncode
    err = open(out + '.err', errors='replace').read()
    m = re.search(r'pushed (\d+) samples', err)
    want = os.path.getsize(raw) // 2
    problems = []
    if rc not in (0, 3): problems.append('exit code %d' % rc)
    if not m or int(m.group(1)) != want: problems.append('pushed %s of %d samples' % (m.group(1) if m else 'unknown', want))
    if audio_loss(err, out + '.part'): problems.append('engine reported dropped/skipped audio')
    if problems:
        os.rename(out + '.part', out + '.FAILED')
        raise RuntimeError('replay gate run invalid for %s: %s (see %s.err)' % (name, '; '.join(problems), out))
    os.rename(out + '.part', out)
    json.dump(dict(fingerprint=fp, rc=rc, pushed=want), open(meta, 'w'))
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
