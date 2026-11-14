#!/usr/bin/env python3
"""Build a deployable candidate model set for a new word and run the continuous replay gate on the packaged engine.

  candidate.py fit  OUT.json   [--word NOVA]        choose hyper-parameters/decay/accept on inner folds, fit on ALL of Ben's recordings (words, their
                                                    augmented copies, non-word audio) + the DEV corpus voices; the HELD voices are not touched
  candidate.py pack OUT.json SRC_SET DST_SET        stock msww set (pulled from a Dot into build/) -> DST_SET with the row, spotter, words
  candidate.py gate MODEL DST_SET OUT_JSON STREAMS...  real engine over the streams; compares every accept with the python simulation
  candidate.py build DATASET MODEL SRC_SET DST_SET REPORT  generic per-word host pipeline, strict held-out replay, no installation

The row and the packed set contain Amazon-derived model files: keep DST_SET under build/ (ignored by git); nothing here touches a Dot.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ds, train as T, loro, gate, msww_pack, enrollment
from feat import read_results


def cmd_fit(a):
    cfg = dict(k=T.K, aug=True, quant=True, fixed=None, fa_cost=a.fa_cost, fit=dict(l2=3.0, pos_frac=0.05, mine_w=8.0, mine_thr=0.08, qat=a.qat),
               grid=[dict(l2=l, pos_frac=p) for l in (1.0, 3.0, 10.0) for p in (0.05, 0.15)])
    dataset = getattr(a, 'dataset', None)
    if dataset:
        word, S, splits, _ = enrollment.load(dataset, cfg['k'])
        train, dev, held = splits['train'], splits['dev'], splits['held']
        cfg['aug'] = False
    else:
        if a.word != 'NOVA':
            raise ValueError('--word only renames the Nova research row; use --dataset for another word')
        word, S = a.word, loro.load(cfg['k'], True)
        train, dev, held = list(ds.BEN), [v for v in ds.DEV if v in S], list(ds.HELD)
    hp, d, acc, info = loro.pick(S, train, dev, cfg)
    print('inner pick', hp, 'decay', d, 'accept', acc, 'cost', info[0][0], 'misses', info[4], 'FA blocks', info[5])
    fk = dict(cfg['fit']); fk.update(hp)
    m = T.fit(S, loro.with_aug(train, S, cfg['aug']), dev, **fk)
    r = m['row']
    json.dump(dict(word=word, k=T.K, decay=d, accept=acc, notify=round(acc / 4, 3), hp=hp, w=m['w'].tolist(), b=m['b'],
                   row=dict(q=r['q'].tolist(), s=float(r['s']), zp=int(r['zp']), sums=int(r['sums']), bias=float(r['bias'])),
                   trained_on=dict(recordings=train, dev_voices=dev, held_voices_untouched=held)), open(a.out, 'w'), indent=1)
    print('wrote', a.out)


def load_model(path):
    j = json.load(open(path))
    r = j['row']
    m = dict(w=np.array(j['w']), b=j['b'], row=dict(q=np.array(r['q'], np.int8), s=np.float32(r['s']), zp=np.int32(r['zp']), sums=np.int32(r['sums']),
                                                    bias=np.float32(r['bias'])))
    return j, m


def cmd_pack(a):
    j, m = load_model(a.model)
    msww_pack.pack(a.src, a.dst, j['k'], dict(q=m['row']['q'], s=m['row']['s'], zp=m['row']['zp'], sums=m['row']['sums'], bias=m['row']['bias']),
                   j['word'], accept=j['accept'], notify=j['notify'], decay=j['decay'], single_word=True)
    print('packed', a.dst)


def cmd_gate(a):
    j, m = load_model(a.model)
    raws = a.streams
    gate.run(a.dst, os.path.join(a.dst, '..', 'gate-' + os.path.basename(a.dst.rstrip('/'))), raws, jobs=a.jobs, speed=a.speed)
    outdir = os.path.join(a.dst, '..', 'gate-' + os.path.basename(a.dst.rstrip('/')))
    dataset = getattr(a, 'dataset', None)
    if dataset:
        word, S, _, raw = enrollment.load(dataset, j['k'])
        if word != j['word']:
            raise ValueError('dataset word does not match model')
        by_raw = {os.path.realpath(v): n for n, v in raw.items()}
    else:
        S = loro.load(j['k'], True)
    res = {}
    for r in raws:
        n = os.path.basename(r)[:-4]
        result_path = os.path.join(outdir, n + '.res')
        unexpected = [v['name'] for v in read_results(result_path) if v['type'] == 2 and v['name'].upper() != j['word'].upper()]
        ev = [(t, sc) for t, sc, ty in gate.events(result_path, j['word']) if ty == 2]
        s = S[by_raw[os.path.realpath(r)]] if dataset else S[n]
        sim = T.events_of(s, m, j['decay'], j['accept'], True, notify=j['notify'])
        cat = T.categorize(s, ev)
        res[n] = dict(engine=ev, sim=sim, unexpected_accepts=unexpected, cat={k: v for k, v in cat.items()})
        agree = len(ev) == len(sim) and all(abs(e[0] - q[0]) < 1 / 16000 and abs(e[1] - q[1]) < 1e-5 for e, q in zip(ev, sim))
        res[n]['agree'] = agree
        print(f"{n:16s} engine accepts {len(ev)} sim {len(sim)} agree={agree}  pos {cat['pos_hit']}/{cat['pos']} dot_fa {cat['dot_fa']} speech_fa {cat['speech_fa']} reply_fa {cat['reply_fa']} unl {cat['unl_fa']} corpus_fa {cat['corpus_fa']} ignored_accepts {cat['ign']}")
    with open(a.out, 'w') as out:
        json.dump(res, out, indent=1, default=str)
    bad = [n for n, v in res.items() if not v['agree'] or v['unexpected_accepts'] or
           (getattr(a, 'strict', False) and (v['cat']['pos_hit'] != v['cat']['pos'] or T.fa_count(v['cat']) or v['cat']['ign']))]
    if bad:
        sys.exit('GATE FAILED: keyword/parity mismatch or strict accuracy failure for ' + ', '.join(bad))


def cmd_build(a):
    """Fit, package target-only, then require independent packaged-engine replay.

    Failed candidates stay local for inspection; this command never installs.
    """
    _, _, splits, raw = enrollment.load(a.dataset)
    a.out = a.model
    a.word = None
    cmd_fit(a)
    cmd_pack(a)
    a.out = a.report
    a.streams = [raw[n] for n in splits['held']]
    a.strict = True
    cmd_gate(a)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    f = sp.add_parser('fit'); f.add_argument('out'); f.add_argument('--word', default='NOVA'); f.add_argument('--fa-cost', type=float, default=3.0); f.add_argument('--qat', type=int, default=0)
    f.add_argument('--dataset', help='per-word enrollment manifest; otherwise legacy Nova research data')
    p = sp.add_parser('pack'); p.add_argument('model'); p.add_argument('src'); p.add_argument('dst')
    g = sp.add_parser('gate'); g.add_argument('model'); g.add_argument('dst'); g.add_argument('out'); g.add_argument('streams', nargs='+'); g.add_argument('--jobs', type=int, default=8)
    g.add_argument('--speed', type=float, default=2)
    g.add_argument('--dataset'); g.add_argument('--strict', action='store_true')
    b = sp.add_parser('build', help='fit + target-only pack + strict held-recording replay; never installs')
    for arg in ('dataset', 'model', 'src', 'dst', 'report'):
        b.add_argument(arg)
    b.add_argument('--fa-cost', type=float, default=3.0); b.add_argument('--qat', type=int, default=0)
    b.add_argument('--jobs', type=int, default=8); b.add_argument('--speed', type=float, default=2)
    a = ap.parse_args()
    {'fit': cmd_fit, 'pack': cmd_pack, 'gate': cmd_gate, 'build': cmd_build}[a.cmd](a)
