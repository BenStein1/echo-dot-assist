"""Host generic MSWW pipeline tests: python3 tools/msww/test_enrollment.py."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import candidate
import enrollment


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        records = []
        for i, (name, split) in enumerate((('one', 'train'), ('two', 'train'), ('noise', 'dev'), ('fresh', 'held'))):
            (self.root / (name + '.raw')).write_bytes(struct.pack('<h', i) * 16000)
            with (self.root / (name + '.feat')).open('wb') as f:
                f.write(struct.pack('<4I', 0x46595250, 2, 512, 15))
                data = np.zeros((12, 528), '<f4'); data[:, 0] = 1
                f.write(data.tobytes())
            records.append(dict(name=name, split=split, raw=name+'.raw', features=name+'.feat',
                                windows={'pos': [] if split == 'dev' else [[0.1, 0.8]]}))
        self.spec = dict(word='JARVIS', recordings=records)
        self.path = self.root / 'enroll.json'
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.spec))

    def test_another_word_and_held_split(self):
        word, streams, splits, raw = enrollment.load(self.path)
        self.assertEqual(word, 'JARVIS')
        self.assertEqual(splits, dict(train=['one', 'two'], dev=['noise'], held=['fresh']))
        self.assertEqual(streams['fresh']['X'].shape, (12, 512))

    def test_duplicate_recording_fails(self):
        (self.root / 'fresh.raw').write_bytes((self.root / 'one.raw').read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            enrollment.load(self.path)

    def test_fit_uses_word_and_training_split_not_held(self):
        args = SimpleNamespace(dataset=str(self.path), word=None, fa_cost=3, qat=0, out=str(self.root/'model.json'))
        row = dict(q=np.zeros(512, np.int8), s=1., zp=0, sums=0, bias=0.)
        model = dict(row=row, w=np.zeros(512), b=0.)
        with patch.object(candidate.loro, 'pick', return_value=({}, .6, .8, ((0,), None, None, None, 0, 0))), \
             patch.object(candidate.T, 'fit', return_value=model) as fit:
            candidate.cmd_fit(args)
        self.assertEqual(fit.call_args.args[1:3], (['one', 'two'], ['noise']))
        result = json.loads(Path(args.out).read_text())
        self.assertEqual(result['word'], 'JARVIS')
        self.assertEqual(result['trained_on']['held_voices_untouched'], ['fresh'])

    def test_no_held_or_invalid_window_fails(self):
        self.spec['recordings'][-1]['windows']['pos'] = [[0, 4]]
        self.save()
        with self.assertRaisesRegex(ValueError, 'window outside'):
            enrollment.load(self.path)

    def test_build_orders_steps_and_only_gates_held(self):
        args = SimpleNamespace(dataset=str(self.path), model='model.json', report='gate.json')
        calls = []
        with patch.object(candidate, 'cmd_fit', side_effect=lambda a: calls.append(('fit', a.out))), \
             patch.object(candidate, 'cmd_pack', side_effect=lambda a: calls.append(('pack', a.out))), \
             patch.object(candidate, 'cmd_gate', side_effect=lambda a: calls.append(('gate', a.out, a.streams, a.strict))):
            candidate.cmd_build(args)
        self.assertEqual(calls, [('fit', 'model.json'), ('pack', 'model.json'),
                                 ('gate', 'gate.json', [str(self.root / 'fresh.raw')], True)])

    def test_strict_gate_rejects_miss_even_with_parity(self):
        args = SimpleNamespace(model='model', dst=str(self.root/'set'), streams=[str(self.root/'fresh.raw')],
                               jobs=1, speed=2, dataset=str(self.path), out=str(self.root/'report.json'), strict=True)
        m = dict(w=np.zeros(512), b=0, row={})
        with patch.object(candidate, 'load_model', return_value=(dict(word='JARVIS', k=12, decay=.6, accept=.8, notify=.2), m)), \
             patch.object(candidate.gate, 'run'), patch.object(candidate.gate, 'events', return_value=[]), \
             patch.object(candidate, 'read_results', return_value=[]), patch.object(candidate.T, 'events_of', return_value=[]):
            with self.assertRaisesRegex(SystemExit, 'GATE FAILED'):
                candidate.cmd_gate(args)


if __name__ == '__main__':
    unittest.main()
