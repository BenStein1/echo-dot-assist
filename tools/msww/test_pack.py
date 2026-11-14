"""Regression: a candidate must not silently listen for stock wake words too."""
import copy
import unittest
from msww_pack import single_word_configs


class SingleWordPackage(unittest.TestCase):
    def test_no_other_spotters_or_state_rules(self):
        target = dict(name='NOVA', classifier={'decay-rate': 0.6})
        kw = {'keyword-orchestra': {'keyword-spotters': [dict(name='AMAZON'), target, dict(name='STOP')]}}
        op = {'initial': 'sleep', 'awake': {'rules': [dict(name='NOVA', next='awake'), dict(name='STOP', next='sleep')],
                                           'timeout': dict(duration=175, next='sleep')},
              'sleep': {'rules': [dict(name='AMAZON', next='awake'), dict(name='NOVA', next='awake')]}}
        original = copy.deepcopy(op)
        kw, op = single_word_configs(kw, op, 'NOVA')
        self.assertEqual(kw['keyword-orchestra']['keyword-spotters'], [target])
        self.assertEqual(op['initial'], original['initial'])
        self.assertEqual(op['awake']['timeout'], original['awake']['timeout'])
        for state in ('sleep', 'awake'):
            self.assertEqual(op[state]['rules'], [dict(name='NOVA', next='awake')])

    def test_missing_or_duplicate_target_fails_closed(self):
        for names in (['COMPUTER'], ['NOVA', 'NOVA']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                single_word_configs({'keyword-orchestra': {'keyword-spotters': [dict(name=n) for n in names]}}, {}, 'NOVA')

    def test_another_word(self):
        kw = {'keyword-orchestra': {'keyword-spotters': [dict(name=n) for n in ('COMPUTER', 'JARVIS', 'STOP')]}}
        op = {'sleep': {'rules': [dict(name=n) for n in ('JARVIS', 'STOP')]}}
        kw, op = single_word_configs(kw, op, 'JARVIS')
        self.assertEqual(kw['keyword-orchestra']['keyword-spotters'], [dict(name='JARVIS')])
        self.assertEqual(op['sleep']['rules'], [dict(name='JARVIS')])


if __name__ == '__main__':
    unittest.main()
