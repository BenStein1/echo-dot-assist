#!/usr/bin/env python3
"""Private pool binding, raw boundaries/context lineage; no microphone needed."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PoolPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        path = Path(cls.build.name)
        (path/'helper.c').write_text(r'''
#include "enroll.c"
int main(int argc,char **argv){
 if(argc<3)return 2;
 if(!strcmp(argv[1],"window")){double a,b;if(msww_word_window(argv[2],&a,&b))return 1;printf("%.6f %.6f\n",a,b);return 0;}
 if(!strcmp(argv[1],"augment")){double offset;if(msww_augment_raw(argv[2],0,8,argv[3],&offset))return 1;printf("%.6f\n",offset);return 0;}
 int rc=msww_prepare_bundle(argv[2],argv[3],argv[4]);printf("%d\n",rc);return rc<0?1:0;
}
''')
        cls.helper = path/'helper'
        subprocess.run(['cc', '-O1', '-ffunction-sections', '-fdata-sections', '-Isrc/hassmic', str(path/'helper.c'),
                        'src/hassmic/json.c', 'src/hassmic/hash.c', '-Wl,--gc-sections', '-lpthread', '-lm', '-o', str(cls.helper)], cwd=ROOT, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.model = self.path/'model'
        self.model.write_bytes(b'bound stock decoder')
        self.meta = dict(version=1, word='Nova', source_sha256=hashlib.sha256(self.model.read_bytes()).hexdigest(),
                         fitting_speakers=['fit'], calibration_speakers=['calibration'])
        for split, key, q in [('train', 'fitting_pool', -127), ('dev', 'calibration_pool', -126)]:
            path = self.path/(split+'.pool')
            path.write_bytes(b'MSWP'+struct.pack('<Ifff4B512b', 1, 2.6, 1., .01, 1, 0, 0, 0, q, *([0]*511)))
            self.meta[key] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.manifest = self.path/'manifest.json'

    def bundle(self, word='Nova'):
        self.manifest.write_text(json.dumps(self.meta))
        target = self.path/f'out-{len(list(self.path.glob("out-*")))}'
        target.mkdir()
        return subprocess.run([self.helper, 'bundle', target, self.model, word], env=dict(os.environ, HASSMIC_ENROLL_MSWW_POOL=str(self.manifest)), text=True, capture_output=True), target

    def test_binding_and_fail_closed(self):
        result, target = self.bundle()
        self.assertEqual(result.stdout.strip(), '1')
        self.assertEqual((target/'pool-train.bin').read_bytes(), (self.path/'train.pool').read_bytes())
        result, _ = self.bundle('Aria')
        self.assertEqual(result.stdout.strip(), '0')  # Unrelated word retains legacy.
        self.model.write_bytes(b'different stock decoder')
        result, _ = self.bundle()
        self.assertEqual(result.returncode, 1)

    def test_corruption_and_speaker_overlap(self):
        self.meta['calibration_speakers'] = ['fit']
        result, _ = self.bundle()
        self.assertEqual(result.returncode, 1)
        self.meta['calibration_speakers'] = ['calibration']
        with (self.path/'train.pool').open('ab') as f:
            f.write(b'changed')
        result, _ = self.bundle()
        self.assertEqual(result.returncode, 1)

    def test_word_boundary_uses_real_noise_floor(self):
        path = self.path/'word.raw'
        frames = [[100]*160 for _ in range(400)]
        for i in range(150, 190):
            frames[i] = [5000]*160
        path.write_bytes(b''.join(struct.pack('<160h', *frame) for frame in frames))
        result = subprocess.run([self.helper, 'window', path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '3.200000 4.400000')
        path.write_bytes(bytes(128000))
        self.assertEqual(subprocess.run([self.helper, 'window', path], capture_output=True).returncode, 1)

    def test_pool_builder_keeps_calibration_speakers_and_excluded_runs_separate(self):
        import sys
        sys.path.insert(0,str(ROOT/'tools/msww'))
        from prepare_pool import prepare
        rows=[]
        for split,seed in [('train',1),('dev',2)]:
            f=self.path/(split+'.feat')
            header=struct.pack('<4I',0x46595250,2,512,15)
            f.write_bytes(header+b''.join(struct.pack('<528f',i*1280,*([seed]*512),*([0]*15)) for i in range(10)))
            rows.append(dict(path=f.name,label=0,split=split,text='Turn the lights on',speaker=split,
                             excluded_windows=[[.3,.5]] if split=='train' else []))
        spec=self.path/'features.json';spec.write_text(json.dumps(rows));out=self.path/'built'
        result=prepare(self.model,spec,out,'Nova')
        self.assertEqual(result['fitting_pool']['cycles'],7)
        self.assertEqual(result['calibration_pool']['cycles'],10)
        pool=(out/'train.pool').read_bytes()
        self.assertEqual([pool[8+i*528+12] for i in range(7)],[1,2,2,1,2,2,2])
        rows[1]['speaker']='train';spec.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'disjoint'):
            prepare(self.model,spec,self.path/'overlap','Nova')
        rows[1]['speaker']='dev';rows[0]['text']='Hello Nova';spec.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'contains'):
            prepare(self.model,spec,self.path/'word','Nova')

    def test_context_is_exact_training_prefix_body_and_padding(self):
        prefix, body = b'\x01\x00'*16000, b'\x02\x00'*32000
        (self.path/'replay-08.raw').write_bytes(bytes(64000)+prefix+bytes(64000))
        (self.path/'replay-00.raw').write_bytes(bytes(64000)+body+bytes(64000))
        out = self.path/'aug.raw'
        result = subprocess.run([self.helper, 'augment', self.path, out], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '1.500000')
        self.assertEqual(out.read_bytes(), bytes(64000)+prefix+bytes(16000)+body+bytes(96000))


if __name__ == '__main__':
    unittest.main()
