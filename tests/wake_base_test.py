"""Focused capture safety checks; no real microphone recording."""
import array
import copy
import contextlib
import io
import subprocess
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('wake_base', Path(__file__).parents[1] / 'tools/wake-base.py')
wake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wake)


def pcm(value, n=64000):
    samples = array.array('h', [value] * n)
    if sys.byteorder != 'little':
        samples.byteswap()
    return samples.tobytes()


class CaptureTests(unittest.TestCase):
    def test_quality_rejects_silence_clipping_and_truncation(self):
        for data in (pcm(0), pcm(32767), pcm(1000, 100), pcm(1000) + b'x'):
            with self.assertRaises(ValueError):
                wake.quality(data, 4)
        good = wake.quality(pcm(-300), 4)
        self.assertEqual(good['peak'], 300)
        self.assertEqual(good['bytes'], 128000)
        self.assertEqual(len(good['sha256']), 64)

    def test_preflight_selects_daemon_not_log_process_and_refuses_active_dump(self):
        d = wake.Device()
        def shell(cmd, data=None):
            if cmd == 'pidof hassmic':
                return '32732 32765'
            if cmd == 'cat /proc/32732/comm':
                return 'hassmic'
            if cmd == 'cat /proc/32765/comm':
                return 'hassmic-log'
            return ''
        with patch.object(d, 'shell', side_effect=shell), patch.object(d, 'opened', return_value=False):
            d.preflight()
        self.assertEqual(d.pid, 32732)
        with patch.object(d, 'shell', side_effect=shell), patch.object(d, 'opened', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'already open'):
                d.preflight()

    def test_full_set_has_fixed_holdouts_and_completes_only_after_all_takes(self):
        d = wake.Device()
        manifests = []
        toggles = []
        with patch.object(d, 'preflight'), patch.object(d, 'shell', return_value=''), \
             patch.object(d, 'opened', return_value=False), \
             patch.object(d, 'toggle', side_effect=toggles.append), \
             patch.object(d, 'write_manifest', side_effect=lambda path, m: manifests.append(copy.deepcopy(m))), \
             patch('builtins.input', return_value=''), patch.object(wake.time, 'sleep'), \
             patch.object(wake.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=pcm(300))), \
             contextlib.redirect_stdout(io.StringIO()):
            wake.capture(d, 'Nova', 4)
        self.assertEqual(toggles, [True, False] * 16)
        self.assertFalse(any(m['complete'] for m in manifests[:-1]))
        self.assertTrue(manifests[-1]['complete'])
        clips = manifests[-1]['clips']
        self.assertEqual(len(clips), 16)
        for label in (0, 1):
            selected = [c for c in clips if c['label'] == label]
            self.assertEqual([c['split'] for c in selected], ['train'] * 6 + ['test'] * 2)
        self.assertEqual(len({c['path'] for c in clips}), 16)

    def test_cancellation_preserves_previous_dump_and_incomplete_manifest(self):
        d = wake.Device()
        commands = []
        manifests = []
        def shell(cmd, data=None):
            commands.append(cmd)
            return 'moved' if 'echo moved' in cmd else ''
        with patch.object(d, 'preflight'), patch.object(d, 'shell', side_effect=shell), \
             patch.object(d, 'opened', return_value=False), \
             patch.object(d, 'write_manifest', side_effect=lambda path, m: manifests.append(m.copy())), \
             patch('builtins.input', side_effect=EOFError):
            with self.assertRaises(EOFError):
                wake.capture(d, 'Nova', 4)
        self.assertFalse(manifests[0]['complete'])
        self.assertTrue(any('capture-before.raw /data/local/hassmic/state/capture.raw' in c for c in commands))
        self.assertTrue(commands[-1].endswith('/.capture-lock'))
        self.assertFalse(any('kill ' in c for c in commands))


if __name__ == '__main__':
    unittest.main()
