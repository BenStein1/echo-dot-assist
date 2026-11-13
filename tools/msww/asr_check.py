#!/usr/bin/env python3
"""Independent label check: transcribe the raw recordings with a local speech recogniser (faster-whisper, word timestamps) and compare the
words it hears with the windows the trainer uses (ds.ben_windows).  Run with the whisperx venv (faster-whisper installed):

  ~/Projects/whisperx/.venv/bin/python tools/msww/asr_check.py [--model large-v3-turbo] OUT.json

Output (under build/, ignored): per recording the recognised words with times and, per labelled window, the words heard inside it.
ASR is not ground truth for a made-up word ("Nova" may come out as "No va", "Noah", "novel"): a window is CONFIRMED when the recogniser hears a
nova-like token inside it, 'dot' windows when it hears dot-like tokens, and anything else is reported as UNVERIFIED with what it heard.
"""
import argparse, json, os, re, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
M = os.path.join(ROOT, 'build', 'msww')
NOVA = re.compile(r'^(nova|novah|no ?va|nov+a|nover|noah|nova\'s|novo)$', re.I)
DOT = re.compile(r'^(dot|dots|dott|that|.ot)$', re.I)


def transcribe(model, raw):
    x = np.fromfile(raw, '<i2').astype(np.float32) / 32768.0
    seg, _ = model.transcribe(x, language='en', word_timestamps=True, vad_filter=False, beam_size=5, condition_on_previous_text=False,
                              initial_prompt='Nova. Dot, dot. Hello, dot. Computer. Alexa. Echo.')
    return [dict(w=w.word.strip(), t0=round(w.start, 2), t1=round(w.end, 2), p=round(w.probability, 2)) for s in seg for w in s.words]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--model', default='large-v3-turbo'); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    sys.path.insert(0, os.path.dirname(__file__))
    import ds
    m = WhisperModel(a.model, device=a.device, compute_type='float16' if a.device == 'cuda' else 'int8')
    out = {}
    for n in ds.BEN + ['nova1', 'dot1']:
        raw = os.path.join(M, 'rec', n + '.raw')
        words = transcribe(m, raw)
        res = dict(words=words, windows=[])
        try:
            W = ds.ben_windows(n)
        except Exception as e:
            W = {}; res['err'] = str(e)
        for kind in ('pos', 'dot', 'speech', 'reply', 'ignore'):
            for (t0, t1) in W.get(kind, []):
                heard = [w for w in words if w['t1'] > t0 and w['t0'] < t1]
                toks = [re.sub(r'[^a-z\' ]', '', w['w'].lower()) for w in heard]
                nov = any(NOVA.match(t) for t in toks); dot = any(DOT.match(t) for t in toks)
                status = ('CONFIRMED' if (kind == 'pos' and nov) or (kind == 'dot' and dot) else 'UNVERIFIED')
                if kind == 'pos' and not nov and dot: status = 'CONTRADICTED(heard dot)'
                res['windows'].append(dict(kind=kind, t0=round(t0, 2), t1=round(t1, 2), heard=' '.join(toks), status=status))
        out[n] = res
        print(n, len(words), 'words;', {k: sum(1 for w in res['windows'] if w['kind'] == k and w['status'] == 'CONFIRMED') for k in ('pos', 'dot')},
              'of', {k: sum(1 for w in res['windows'] if w['kind'] == k) for k in ('pos', 'dot', 'speech')}, flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
