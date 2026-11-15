#!/usr/bin/env python3
"""Capture a private reusable wake corpus on an Echo; never train/install a model."""
import argparse
import array
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
import time
import uuid

ROOT = '/data/local/hassmic/wake-base'
CAPTURE = '/data/local/hassmic/state/capture.raw'
NEGATIVES = ['Dot', 'No', 'Go over there', 'Turn the lights on',
             'What time is it', 'Play some music',
             'Set a timer for five minutes', 'Good morning']


def quality(data, seconds):
    if len(data) % 2 or not seconds * 32000 * .8 <= len(data) <= seconds * 32000 * 1.5:
        raise ValueError('Incomplete or stalled microphone capture; keep the failed take for inspection.')
    samples = array.array('h', data)
    if sys.byteorder != 'little':
        samples.byteswap()
    peak = max(map(abs, samples), default=0)
    clipped = sum(abs(x) >= 32760 for x in samples)
    if peak < 100:
        raise ValueError('Nearly silent: speak a little closer to the Echo.')
    if clipped * 100 > len(samples):
        raise ValueError('Clipping: speak a little farther from the Echo.')
    return {'bytes': len(data), 'peak': peak, 'clipped_samples': clipped,
            'sha256': hashlib.sha256(data).hexdigest()}


class Device:
    def __init__(self, serial=None):
        self.adb = ['adb'] + (['-s', serial] if serial else [])

    def shell(self, command, data=None):
        return subprocess.run(self.adb + ['shell', command], input=data,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True, timeout=15).stdout.decode().strip()

    def preflight(self):
        pids = self.shell('pidof hassmic').split()
        running = [p for p in pids if self.shell(f'cat /proc/{int(p)}/comm') == 'hassmic']
        if len(running) != 1:
            raise RuntimeError('Expected exactly one running hassmic daemon.')
        self.pid = int(running[0])  # hassmic-log shares its executable, but is not the capture process
        self.shell('test -d /data/local/hassmic/state && test ! -L ' + CAPTURE)
        if self.opened():
            raise RuntimeError('Another capture is already open; finish it before recording this set.')

    def opened(self):
        # Do not infer signal delivery from a successful kill: wait for the actual open file.
        return self.shell(f'test -d /proc/{self.pid}/fd && for f in /proc/{self.pid}/fd/*; do '
                          f'[ "$(readlink "$f")" = {shlex.quote(CAPTURE)} ] && echo open; done; true') == 'open'

    def toggle(self, wanted):
        self.shell(f'kill -TTIN {self.pid}')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.opened() == wanted:
                return
            time.sleep(.05)
        raise RuntimeError('Microphone capture did not acknowledge the signal; do not blindly toggle it again.')

    def write_manifest(self, folder, manifest):
        self.shell(f'umask 077; cat > {folder}/manifest.tmp && mv {folder}/manifest.tmp {folder}/manifest.json',
                   (json.dumps(manifest, indent=2) + '\n').encode())


def capture(device, phrase, seconds):
    device.preflight()
    folder = ROOT + '/' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    lock = ROOT + '/.capture-lock'
    device.shell(f'umask 077; mkdir -p {ROOT} && mkdir {lock}')
    active = False
    old = False
    manifest = {'version': 1, 'phrase': phrase, 'sample_rate': 16000, 'channels': 1,
                'format': 's16le', 'source': 'hassmic post-AEC micAsr', 'complete': False,
                'split_policy': 'last two of each class held out; freeze before training', 'clips': []}
    try:
        device.shell(f'umask 077; mkdir {folder}')
        old = device.shell(f'if [ -e {CAPTURE} ]; then mv {CAPTURE} {folder}/capture-before.raw && echo moved; fi') == 'moved'
        device.write_manifest(folder, manifest)
        print('Recordings stay on the Echo at ' + folder, flush=True)
        print('No training or model change. Keep other audio quiet. Say only after GO.\n'
              'Normal wake detection remains running: avoid its currently selected wake word.\n'
              'Positives: vary pace/distance slightly; negatives must NOT contain your word.', flush=True)
        for label in (1, 0):
            for index in range(8):
                prompt = phrase if label else NEGATIVES[index]
                # Exact token exclusion, not a language-understanding mechanism.
                if not label and phrase.casefold() in prompt.casefold():
                    prompt = 'A different everyday sentence without your wake word'
                attempt = 0
                while True:
                    input(f'\n{"POSITIVE" if label else "NEGATIVE"} {index+1}/8: {prompt}\nPress Enter when ready: ')
                    for n in (3, 2, 1):
                        print(n, flush=True)
                        time.sleep(1)
                    # Mark ownership before sending: interruption during acknowledgement still stops our dump.
                    active = True
                    device.toggle(True)
                    print(f'GO — {prompt}', flush=True)
                    time.sleep(seconds)
                    device.toggle(False)
                    active = False
                    print('STOP', flush=True)
                    name = f'{"positive" if label else "negative"}-{index+1:02d}-take{attempt+1:02d}.raw'
                    device.shell(f'mv {CAPTURE} {folder}/{name}')
                    data = subprocess.run(device.adb + ['exec-out', 'cat', folder + '/' + name],
                                          capture_output=True, check=True, timeout=15).stdout
                    try:
                        metrics = quality(data, seconds)
                    except ValueError as exc:
                        print(str(exc) + ' Retrying; failed take retained on device.', flush=True)
                        attempt += 1
                        continue
                    manifest['clips'].append({'path': name, 'label': label,
                                              'split': 'test' if index >= 6 else 'train',
                                              'prompt': prompt, **metrics})
                    device.write_manifest(folder, manifest)
                    break
        manifest['complete'] = True
        device.write_manifest(folder, manifest)
        print('\nSaved 8 positives + 8 negatives. Reuse this manifest; do not record again:\n' + folder + '/manifest.json')
    finally:
        # Never blindly signal twice: SIGTTIN is a toggle, not a stop command.
        if active and device.opened():
            device.toggle(False)
        if device.opened():
            raise RuntimeError('Capture still open; keep the lock and inspect before retrying: ' + folder)
        # Preserve an interrupted take as well as the accepted takes.
        device.shell(f'if [ -f {CAPTURE} ]; then mv {CAPTURE} {folder}/interrupted.raw; fi')
        if old:
            device.shell(f'mv {folder}/capture-before.raw {CAPTURE}')
        device.shell(f'rmdir {lock}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phrase', nargs='?', default='Nova')
    parser.add_argument('--serial', help='ADB device serial (for example 10.0.0.199:5555)')
    parser.add_argument('--seconds', type=int, choices=range(2, 7), default=4)
    parser.add_argument('--check-device', action='store_true', help='check capture prerequisites without recording')
    args = parser.parse_args()
    if not args.phrase.strip() or any(ord(c) < 32 for c in args.phrase):
        parser.error('Use a nonempty wake phrase without control characters.')
    device = Device(args.serial)
    if args.check_device:
        device.preflight()
        print('Echo capture prerequisites OK; no recording started and no model changed.')
    else:
        capture(device, args.phrase.strip(), args.seconds)


if __name__ == '__main__':
    try:
        main()
    except (OSError, subprocess.SubprocessError, RuntimeError, EOFError, KeyboardInterrupt) as exc:
        print(f'Capture stopped: {exc}. Existing dataset files are retained; do not delete them.', file=sys.stderr)
        sys.exit(1)
