#!/usr/bin/env python3
"""Native enrollment regression, no firmware/ONNX/Python numeric dependencies.

make build/msww_train-host && python3 tests/msww_native_test.py
MSWW_NATIVE=build/donut/msww_train MSWW_QEMU=1 also exercises Android/ARM.
Synthetic separable features prove plumbing, NOT real wake-word accuracy.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import shutil

ROOT = Path(__file__).resolve().parents[1]


def var(n):
    out = bytearray()
    while n > 127:
        out.append((n & 127) | 128)
        n >>= 7
    return bytes(out + bytes([n]))


def field(k, b):
    return var(k * 8 + 2) + var(len(b)) + b


def tensor(name, dims, dtype, data, packed=False):
    return (b''.join(var(8) + var(d) for d in dims) + var(16) + var(dtype) +
            field(8, name.encode()) + field(5 if packed else 9, data))


class NativeMSWW(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        tensors = [
            tensor('kernel/Dense:20', [15, 512], 3, bytes(15 * 512)),
            tensor('kernel/Dense:20_s', [15], 1, struct.pack('<15f', *([1.] * 15))),
            tensor('kernel/Dense:20_zp', [15], 6, bytes(60)),
            tensor('kernel/Dense:20_sums', [15], 6, bytes(15), True),
            tensor('bias/Dense:20', [1, 15], 1, bytes(60)),
            # Real decoders contain unrelated packed int32 tensors longer than
            # the head. The writer must preserve them, not reject/truncate them.
            tensor('unrelated_sums', [256], 6, bytes(256), True),
        ]
        self.source = self.path / 'source.onnx'
        self.source.write_bytes(field(7, b''.join(field(5, t) for t in tensors)))
        rows = []
        for label in (1, 0):
            for i in range(8):
                name = f'features-{len(rows):02d}.bin'
                rows.append(dict(path=name, label=label, split='train' if i < 5 else 'dev' if i == 5 else 'test'))
                self.features(name, label, i)
        self.rows = rows
        self.manifest = self.path / 'features.json'
        self.manifest.write_text(json.dumps(rows))

    def features(self, name, label, seed=0):
        with (self.path / name).open('wb') as f:
            f.write(struct.pack('<4I', 0x46595250, 2, 512, 15))
            for i in range(87):
                t = (i + 1) * .08
                x = [4. if label and 2.2 < t < 2.7 else -1., (seed + 1) * .001] + [0.] * 510
                f.write(struct.pack('<528f', i * 1280., *x, *([0.] * 15)))

    def run_native(self, dest):
        binary = ROOT / os.environ.get('MSWW_NATIVE', 'build/msww_train-host')
        cmd = [str(binary), str(self.source), str(self.manifest), str(dest), '3']
        if os.environ.get('MSWW_QEMU'):
            cmd = ['bwrap', '--ro-bind', '/', '/', '--bind', '/tmp', '/tmp', '--dev', '/dev', '--proc', '/proc', '--unshare-pid', '--die-with-parent', 'qemu-arm', '-L', str(ROOT/'firmware/donut/rootfs'), '-E', 'LD_LIBRARY_PATH=/system/lib', *cmd]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    def test_fit_freeze_calibrate_and_held_miss(self):
        dest = self.path / 'candidate.onnx'
        result = self.run_native(dest)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(Path(str(dest)+'.replay.json').read_text())
        self.assertEqual(len(report['recordings']), 16)
        self.assertGreater(report['accept'], report['notify'])
        self.assertTrue(all(len(r['events']) == (1 if i < 8 else 0) for i, r in enumerate(report['recordings'])))
        self.assertIn('"test":{"tp":2,"fn":0,"fp":0,"tn":2}', result.stdout)
        frozen = dest.read_bytes()
        self.assertIn(field(5, tensor('unrelated_sums', [256], 6, bytes(256), True)), frozen)
        # Change held data only. Fit/calibration and written ONNX must not change.
        self.features('features-06.bin', 0)
        second = self.path / 'miss.onnx'
        result = self.run_native(second)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(second.read_bytes(), frozen)
        self.assertIn('"test":{"tp":1,"fn":1', result.stdout)

    def test_incompatible_features_fail_closed(self):
        (self.path / 'features-00.bin').write_bytes(struct.pack('<4I', 0x46595250, 1, 25, 100))
        dest = self.path / 'bad.onnx'
        result = self.run_native(dest)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(dest.exists())

    def test_no_calibration_leakage_into_held(self):
        # Removing a dev negative must fail rather than use held negatives.
        self.rows[13]['split'] = 'test'
        self.manifest.write_text(json.dumps(self.rows))
        dest = self.path / 'bad.onnx'
        result = self.run_native(dest)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(dest.exists())

    @unittest.skipIf(os.environ.get('MSWW_QEMU'), 'orchestration harness is a host test; native backend is tested on ARM separately')
    def test_real_enrollment_job_routes_msww_and_blocks_bad_replay(self):
        # Actual enroll.c job, native optimizer/writer, and native packager.
        # Extraction and engine events are stubs: this is integration plumbing,
        # never evidence of real audio accuracy or device rendering.
        harness = self.path / 'job.c'
        harness.write_text(r'''
#include "enroll.c"
static int committed;
int art_room(long long n,char *e,size_t z){(void)n;(void)e;(void)z;return 0;}
void art_unstage(const char *id){(void)id;}
int art_stage_dir(const char *id,char *p,size_t n){(void)id;snprintf(p,n,"%s",getenv("TEST_STAGE"));return 0;}
int art_staged(const char *id,char *e,size_t z){(void)id;(void)e;(void)z;return 0;}
int art_commit(const char *id,char *e,size_t z){(void)id;(void)e;(void)z;committed=1;return 0;}
int art_install(char *e,size_t z){snprintf(e,z,"test stops before root install");return -1;}
int main(int argc,char **argv){
 if(argc!=2)return 2;
 S.seconds=3;strcpy(S.key,"JARVIS");strcpy(S.phrase,"Jarvis");strcpy(S.id,"jarvis-en-US");strcpy(S.base,"unused");
 job(argv[1]);printf("committed=%d error=%s\n",committed,S.error);return 0;
}
''')
        binary = self.path / 'job'
        subprocess.run(['cc', '-O1', '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
                        '-I'+str(ROOT/'src/hassmic'), str(harness),
                        *[str(ROOT/'src/hassmic'/n) for n in ('enroll_pkg.c', 'statefile.c', 'json.c', 'hash.c')],
                        '-lpthread', '-lm', '-o', str(binary)], check=True, capture_output=True)
        base = self.path / 'base'
        (base/'multi-wakeword-wakeworddecoder').mkdir(parents=True)
        (base/'BDPGeneratedFiles').mkdir()
        shutil.copyfile(self.source, base/'multi-wakeword-wakeworddecoder/odyssey_msww.streaming.fixed.int8.onnx')
        (base/'pryon.manifest').write_text('{}')
        (base/'BDPGeneratedFiles/kw.cfg.json').write_text(json.dumps({'keyword-orchestra': {'keyword-spotters': [
            {'name': 'COMPUTER', 'classifier': {'decay-rate': .6}, 'spotter': {
                'word-component-array': [{'scorer-index': 3}], 'lock-period': 100,
                'classification-thresholds': {'accept-threshold': .523, 'notify-threshold': .12}}}, {'name': 'STOP'}]}}))
        (base/'BDPGeneratedFiles/op.cfg.json').write_text('{"sleep":{"rules":[{"name":"COMPUTER"},{"name":"STOP"}]}}')
        (base/'BDPGeneratedFiles/words.shrunk.txt').write_text('COMPUTER 3\nSTOP 1\n')
        extract = self.path/'extract'
        extract.write_text('#!/usr/bin/env python3\nimport sys,shutil,os\nfrom pathlib import Path\n'
                           'assert sys.argv[1]=="-w"\n'
                           'out=Path(sys.argv[-1]); shutil.copyfile(Path(os.environ["TEST_FIXTURES"])/out.name,out)\n')
        engine = self.path/'engine'
        engine.write_text(r'''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
raw=Path(sys.argv[-1]); i=int(raw.stem.split('-')[-1]); assert raw.exists()
assert Path(sys.argv[2]).exists()
r=json.loads((raw.parent/'candidate.onnx.replay.json').read_text())['recordings'][i]
for at,score in r['events']:
 print('RESULT decoder=x keyword=%s type=2 begin=0 end=0'%('ALEXA' if os.environ.get('FAIL_REPLAY') else 'JARVIS'))
 print('META header 9 bytes, '+json.dumps(dict(kwSampleDetectionIndex=at,kwClassificationScore=score)))
print('pushed %d samples'% (raw.stat().st_size//2))
sys.exit(0 if r['events'] else 3)
''')
        extract.chmod(0o755)
        engine.chmod(0o755)
        state = self.path/'state'; state.mkdir()
        for fail in (False, True):
            jobdir = self.path/('bad-job' if fail else 'good-job'); jobdir.mkdir()
            for i in range(16):
                (jobdir/f'clip-{i:02d}.raw').write_bytes(bytes(3*32000))
            env = dict(os.environ, HASSMIC_ENROLL_BASE=str(base), HASSMIC_STATE=str(state),
                       HASSMIC_PRYON_EXTRACT=str(extract), HASSMIC_MSWW_TRAIN=str(ROOT/'build/msww_train-host'),
                       HASSMIC_PRYON_TEST=str(engine), HASSMIC_ENROLL_MSWW_NEGATIVES=str(self.path/'absent.json'),
                       TEST_STAGE=str(self.path/('bad-stage' if fail else 'good-stage')), TEST_FIXTURES=str(self.path))
            if fail: env['FAIL_REPLAY']='1'
            result = subprocess.run([str(binary), str(jobdir)], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('committed=0' if fail else 'committed=1', result.stdout)
            self.assertIn('real-engine replay' if fail else 'test stops before root install', result.stdout)


if __name__ == '__main__':
    unittest.main()
