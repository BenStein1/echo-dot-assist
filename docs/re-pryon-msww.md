# Amazon's multi-word wake word sets (msww): how five words share one engine

### Reusable host generation pipeline — 2026-11-13

The installed Nova trial uses a newly fitted row in the **15-class MSWW decoder**,
not the older Alexa binary head. Other network classes still compete with the
new word, while only the target spotter and its state rules may emit callbacks.
Ben reports that the fresh room test seemed to work. That is useful personal
validation, not a measured guarantee for every word/speaker/room.

`candidate.py build` now applies that method to explicit **per-word** data:

```sh
OPENBLAS_NUM_THREADS=1 build/pryon-review-venv/bin/python tools/msww/candidate.py build \
  build/msww/jarvis/enrollment.json build/msww/jarvis/model.json \
  build/msww/computer-en-US build/msww/jarvis/set build/msww/jarvis/replay.json
```

Manifest paths are relative to the manifest; raw names must match recording
names. Example schema (supply at least two `train`, one negative `dev`, and
independent `held` recordings, not just this single illustrative entry):

```json
{"word":"JARVIS","recordings":[
  {"name":"session1","split":"train","raw":"session1.raw",
   "features":"session1.feat","windows":{"pos":[[2.0,3.2]],"speech":[[5.0,7.0]]}}
]}
```

Use continuous 16 kHz signed-16-bit little-endian mono audio and features from
`pryon_extract -w` of **the same padded stream and stock MSWW source set**.
Windows are seconds in that stream; positive windows include the trainer's
0.3 s lead / 0.2 s tail margins. Word identity must be the uppercase identifier
used by the spotter. Use genuine examples of that word, not Nova recordings
renamed via `--word`. Choose splits by original recording before fitting;
identical audio under multiple names is rejected, but the caller must keep
related/augmented recordings in the same split. This path accepts no automatic
augmentation. `dev` is negative speech/background; include confusable words
(such as hello), replies, and real-room noise in held audio, alongside fresh
positive examples. Do not tune against held results and call them held-out.

The command chooses hyperparameters/thresholds within training/development data,
fits the final quantized row, packages only the target spotter, then replays only
the held recordings through the real engine. It fails on any missed labeled
word, false/ignored accept, non-target accept, dropped audio, or simulator/engine
count/time/score mismatch. Failed local artifacts remain for inspection; nothing
is installed or selected automatically. A successful small replay is not broad
accuracy certification. Keep Amazon-derived models and personal audio in ignored
`build/`; use the existing signed artifact path and reversible trial for delivery.

**Scope:** this integrates the host MSWW generation pipeline, **not** the settings
page's `enroll.c` pipeline. That still uses `pryon_train` and the old Alexa head.
Replacing it requires the MSWW data/pool conversion, ONNX row writer, calibration
and packaged replay to work on-device; do not silently claim UI-generated words
use this method. Existing stock words and microWakeWord are unchanged.

Checks: `python3 tools/msww/test_enrollment.py`, `test_pack.py`, `test_gate.py`.

Status: **research prototype; NOT accepted for deployment. See recovery review below.**
Source: `/data/local/hassmic/models/{alexa,amazon,computer,echo,ziggy}-en-US/` on the Dot (Amazon's downloaded model sets;
a copy of `computer-en-US` was pulled to `/tmp/amz/computer` while researching). Amazon-derived: keep out of git.

## Two architectures in the same engine

| | Old set (firmware fallback, what `enroll.c` and `pryon_train` use) | New set (DAVS download: alexa/echo/computer/ziggy/amazon) |
|---|---|---|
| Layout | one ONNX + `kw.cfg.json`, `nemort_scorer_config.json` | `encoder/`, `multi-wakeword-wakeworddecoder/`, `dvaddecoder/`, `siddecoder/`, `nttfusionconfig/`, `watermarkconfig/`, `fingerprintconfig/`, `BDPGeneratedFiles/` |
| Network | 7 HybridConv + 2 LSTM + head `fc_output_ww` [3,100] | encoder (conv only) then a separate trained decoder |
| What a word is | one int8 row of a 3-row head, per-lane, linear | one class of a 15-way softmax over a temporal conv stack, plus its own spotter config |
| Extra stages | none | DVAD, speaker ID, watermark, NTT fusion |

## The new set, measured from the ONNX files

* `encoder/primed_encoder.int8-streaming.onnx` (3.6 MB, 104 nodes: 12 `HybridGemm`, no LSTM). Input `[1,8,64]` (8 frames of the 64-bin LFBE),
  output `encoder_embeddings [1,1,100]`; published to the other stages as tensor artifact `encoder_output`.
  Shared by every word: the 100-d interface is the same width the old head sees.
* `multi-wakeword-wakeworddecoder/odyssey_msww.streaming.fixed.int8.onnx` (835 KB, 53 nodes, 12 `HybridGemm`): input `[1,1,100]`,
  six streaming conv buffers (`conv_buffer_input/Concat:*`, widths 100/384/384/100/192/192) = the temporal context. Last layers:
  `kernel/Conv1D:37 [256,192]` -> Relu -> `kernel/Dense:43 [2,256]` -> `mww/start_end`;
  and `kernel/Dense:20 [15,512]` -> Softmax -> **`msww/posteriors [1,1,15]`**.
* Per word: `<word>_spotter.cfg.json`: `scorers:["mww"]`, `scorer-index` = the class index (amazon 5, computer 7, hey_disney 3 and 11, stop 2,
  ziggy 14), classifier `exponential-decay` with `decay-rate` (0.6 / 0.6 / 0.3 / 0.0), accept 0.523 (computer), 0.576 (amazon), 0.845 (ziggy),
  0.573 (stop), 0.35 (hey_disney); notify 0.12 (computer); lock-period 100; keyword-feature-window 100;
  `mww/start_end` gives the word's start and end.
* `words.shrunk.txt`: `<eps> 0, AMAZON 1, COMPUTER 2, HEY_DISNEY 3, STOP 4, ZIGGY 5` (names, not the model's class indices).
  Unused class indices in the 15: 1, 4, 6, 8, 9, 10, 12, 13 (what they were trained for is not known).
* `BDPGeneratedFiles/` assembles the set (`decoders.cfg.json` lists scorers dvad, sid, mww; `kw.cfg.json` is the 9.5 KB orchestra).

## Why this answers "how did Amazon do five words"

One shared encoder and one multiclass decoder trained on huge audio with all words and everything else as competing classes. The decoder's
penultimate 512 values are already a temporal "which wake word, if any" embedding. A word is a row of `Dense:20` plus a spotter file.

## Hypothesis to test

Enrolling a new word = fit one row (+ scale, zero point, sum, bias) of `kernel/Dense:20` on the engine's own 512-d input, with the stock
words and speech as negatives, using the same HybridGemm per-row dynamic-quantisation math already reproduced for the old head
(`pryon_qat.c`). The features are richer (temporal, trained to reject non-words) than the old per-lane 100-d row, so few-shot enrollment
should separate near words such as "dot" from "Nova" much better.

## Plan

1. Confirm the engine loads a copy of a stock set with an edited decoder (is `checksum.txt` enforced for the new layout? do NTT/DVAD/SID gate on a
   word's identity?). Pick the class index to use (a free one, or replace `ziggy`/`hey_disney`).
2. Tap the input of the `[15,512]` HybridGemm (the library's kernel at +0x123897c, select by weight shape) in `pryon_extract`/`pryon_tap.S`, run
   under qemu on recordings.
3. Reproduce the `exponential-decay` spotter rule against the engine's own RESULT lines (as `pryon_decoder.py` does for moving-average-max-pool).
4. **Feasibility first, no new recordings**: existing captures (live stream 175530, wake-check-*, false-wake-improve/*, testdata/corpus TTS), held-out
   streams, measure Nova vs dot separation against the old head.
5. If it works: trainer (row only), packer (stock set + edited decoder + spotter + words + decoders), replay gate through the packaged engine.
6. Minimise the enrollment: accuracy against the number of positive samples and the shared negative set. Goal: a couple of minutes, not hours.

## Recovery review — 2026-11-12

Saved code through `acd6a59` implements the row-training prototype. The completed
nested leave-one-recording-out run (`build/msww/recovered-original-loro.json`,
ignored) reports **12/15 Nova detections and 2/31 dot-type false accepts**, with
0/5 other-speech false accepts. Each of four fitted rows has zero simulated
accepts on the **same** 0.78-hour, six-voice held-out corpus. This is not 3.12
unique hours, and not proof of a production false-accept rate.

These numbers remain provisional:

- `labels.py` detects acoustic bursts but assigns word identity using prompt
  timestamps/order, spectral heuristics, and manual timeline selections. It
  does **not** independently transcribe/listen to the words; interrupted prompts
  can still invalidate labels. `rdot` also uses assumed post-dot word order.
- `gate.py` ignores subprocess return codes, discards stderr, and caches only
  by stream basename. Failed, incomplete, stale, or backlog-dropping runs must
  not count as a successful gate. A zero-event run is legitimate (`pryon_test`
  returns 3) only with positive evidence of complete input processing. Capture
  stderr, compare pushed samples with input length, reject dropped audio,
  and bind cached evidence to the model, input, and runtime options.
- `candidate.py gate` prints disagreement but does not fail on it; it compares
  event times/counts, not scores, and does not enforce accuracy/false-accept
  criteria. A fitted candidate tested on its training recordings is only a
  packaging/parity check, not held-out accuracy evidence.
- `curve.py` has only 15 recorded positive windows total. Its 20/40 sample
  conditions use augmentation, not 20/40 independent spoken examples. It
  cannot yet establish a reliable enrollment-time recommendation.

Decision: **do not deploy; acceptance failed**. Correct the replay gate, verify
semantic labels independently, and obtain independent review before making
performance claims. Keep HELD speech out of further tuning; any inspection-
led changes require fresh final evaluation audio. No new recording campaign
is justified by these results yet.

Independent GPT-6 Luna review was attempted via Gateway and native Codex; both
failed with missing-bearer 401 errors. The settled worker also exhausted its
Anthropic quota and failed LiteLLM authentication. Configured fallback lists
are not evidence of a working provider. These are outstanding review/runtime
blockers, not a completed review. Nothing was deployed to the Dot.

### Replay recovery — 2026-11-13

The first gate correction was overbroad: matching `backlog|dropp|skipp`
rejected configuration keys and `skipped 0 during shutdown`. The gate now
checks nonzero skipped/dropped diagnostics and structured loss counters in
result metadata. Its cache version and test-executable digest invalidate old
results. `python3 tools/msww/test_gate.py` checks benign and nonzero cases.
Candidate parity now fails on time or score mismatch as well as counts;
timestamp tolerance is less than one sample and score tolerance is `1e-5`.

Fresh packaged-engine replay agrees with simulation on `w171828` (4 accepts),
`w172114` (4), and `live` (2), including timing and scores. These are training
recordings, not held-out accuracy evidence. Current working labels exclude
some earlier positives; these counts do not establish a new recall figure.
The `rdot` replay is rejected because `audioMetadataDuringDetection` contains
nonzero `dropFramesCount`/`largestFrameDropedAtOnce`, despite zero decoder
skipped-backlog counters. These front-end metadata counters must not be
confused with proof that the replay harness itself dropped samples. Their
origin and effect remain unresolved; the gate conservatively rejects them.
Absence of nonzero counters is not independent proof of every internal frame
being processed, especially in streams with no callbacks.

The continuation worker exhausted OpenAI authentication, Anthropic quota,
and LiteLLM authentication before producing output. No workers remained
active when the parent recovered this event. Uncommitted ASR, compact-pool,
curve, and C-trainer experiments remain separate, unreviewed work.

### Candidate recovery and metadata control — 2026-11-13

Five of the six HELD streams have validated packaged-engine evidence: zero
NOVA accepts, matching simulation, over **0.606524 h** of unique speech.
`en-gb-f-elderly` (0.175008 h) also contains zero NOVA accepts, but its replay
was rejected for nonzero front-end metadata drop counters. Do not count that
failed result as valid exposure. Evidence: ignored
`build/msww/held-recovery-verified.json`. No candidate was deployed.

The source recording must be distinguished from a decoder backlog loss.
A controlled offline copy of `rdot.raw`, with bit 0 cleared in every int16
sample (`samples & -2`, at most one PCM count of change), passes the unchanged
strict replay gate. All its callbacks now report `has_audio_lsb_metadata=0`;
the source replay reported embedded front-end drop counters. Fresh features
from that exact controlled input match all four real-engine NOVA accepts
in count, time, and score (`1e-5` tolerance): 22.96, 27.76, 32.88, 38.08 s.
The control is a **different input**, not a retroactive pass of the original
recording, nor justification to ignore front-end counters in live audio.
Evidence is under ignored `build/msww/metadata-control/`.

Importantly, the first two accepts fall inside windows that the saved ASR
check transcribed as **dot**, not Nova (22.25–23.65 and 27.00–28.40 s).
The working dataset currently sets such contradicted positive windows aside
as `ignore`. Thus reporting only `dot_fa=0` hides two unresolved accepts;
`candidate.py gate` now also prints `ignored_accepts`. ASR is imperfect and
was prompted with the target words: it is not independent semantic ground
truth. These two detections must be adjudicated, not silently excluded from
a performance claim. Training-set recall is not held-out recall.

The preserved compact/C-trainer experiments are not yet accepted. Review
found that `ondev_check.py` computes a dequantized pool (`Xdq`) but does not
use it in the scipy reference, despite claiming identical input; C mining
attenuates nonsequential samples by `(1-decay)` while Python leaves them
unsmoothed; and the C input loaders need bounds/finite-value validation.
Do not describe a host-only compilation as on-device validation.

The same bit-0-cleared control for the rejected HELD stream also passes
the unchanged gate: zero NOVA accepts over 0.175008 h, with
`has_audio_lsb_metadata=0`. This supports the metadata-origin diagnosis,
not a pass of the original source stream or additional unique exposure.

Enrollment-size sweep completed with one deterministic draw per cell:
`OPENBLAS_NUM_THREADS=1 python3 tools/msww/curve.py --draws 1 --out build/msww/curve-recovery.json`.
These are exploratory simulations with the working ASR-filtered labels,
fixed decay 0.6 / accept 0.8, not independent production estimates.

| Real examples | Shared pool: real only | Shared pool: +augmentation | +own negatives: real only | +own negatives: +augmentation |
|---|---|---|---|---|
| 2 | 6/12, 2 FA | 10/12, 53 FA | 3/12, 0 FA | 6/12, 0 FA |
| 4 | 10/12, 4 FA | 12/12, 95 FA | 4/12, 0 FA | 9/12, 0 FA |
| 8 | 12/12, 4 FA | 12/12, 100 FA | 5/12, 0 FA | 9/12, 0 FA |

Each cell shows Nova detections and **reported** Ben nonword false accepts.
Those FA counts exclude ignored windows, so zero is not a complete safety
claim. Each fitted row has zero simulated HELD-corpus accepts on the same
0.781532 h, not four independent copies of that exposure. Augmentation
uses five clips per real example, not five independent utterances. The
shared-only augmentation sharply increases false accepts; adding own
negatives improves rejection but loses recall. There is no supported
Amazon-level specificity or minutes-long enrollment recommendation yet.

### Review closure, ignored-accept adjudication, held-out result — 2026-11-13

**Review findings closed** (commit `2f8460f`; every experiment file committed, none discarded).
`ondev_check.py` now fits the scipy reference on the dequantised pool (`Xdq`) and the same float word
windows as the C trainer. `msww_row.c` uses seq codes 0 (pool cycle, not time-ordered: mined unsmoothed
like `train._smooth_pool`), 1 (start of a time-ordered run, smoothed from s=0 like `spotter.smooth`) and
2 (continues the run), and validates magic, counts, finite/bounded values, seq codes and parameters
(NaN/inf, bad seq, truncation all exit 2). Host-only check, folds `live`/`rdot`: row corr(C, scipy)
0.969/0.995 and identical Nova / Ben-FA / HELD-FA counts at accept 0.8 and 0.5. This is a host
comparison, not on-device validation.

**The two ignored `rdot` accepts (22.96 s, 27.76 s)** — `tools/msww/adjudicate.py`, listen-free. Reference
bursts are Ben's single words from the other captures (labels by prompt/timeline, not ASR): Nova n=11,
0.40±0.05 s, 52.9±1.9 dB, centroid 910±119 Hz; dot n=16, 0.32±0.07 s, 57.6±4.1 dB, 1114±119 Hz;
leave-one-out 3-NN 25/27. Both bursts are 0.40 s, 53.7 dB, centroid 924 / 738 Hz, and are far closer to
Nova than to dot (log-mel 3-NN distance 10.8 vs 25.8; 10.7 vs 23.7), exactly like the two ASR-confirmed
Nova words in the same recording (17.4 vs 26.9; 15.0 vs 26.5). The old head's live "Dot, dot. Hello, dot."
and "Hello, dot" false accepts (7.5, 26.52, 35.7 s) are dot-like (16.7/16.1/14.9 vs 23–29 to Nova), as are
three prompted dots in `rdot`. Verdict: the two accepts are **very likely true Nova, not dot false
accepts**, but this is acoustic similarity against a small single-speaker reference set, and it contradicts
the (prompted, imperfect) ASR; nothing here is a human-verified transcript. They stay in the conservative
FA column below until someone listens.

**Final numbers** (engine replay, `NOVA`, working labels, `ignored_accepts` counted; old head =
`build/msww/gate/old`, candidate = packaged candidate set / metadata-control replay):

| Held-out corpus (6 HELD voices) | accepts | exposure | FA/h |
|---|---|---|---|
| old head | 15 (de-f-elderly 1, de-m-young 5, en-gb-f-elderly 5, en-ja-m-young 1, en-us-f-young 2, en-zh-f-young 1) | 0.781532 h | 19.2 |
| candidate, 5 valid streams | 0 | 0.606524 h | 0 (rule-of-three 95% upper bound 4.9/h) |
| candidate, + en-gb-f-elderly (bit-0-cleared control, different input) | 0 | 0.781532 h | 0 (upper bound 3.8/h) |

| Ben recordings (candidate trained on all four) | old head | candidate |
|---|---|---|
| Nova windows hit (working labels) | 12/12 | 12/12 |
| dot false accepts (live) | 3 (7.5, 26.52, 35.7) | 0 |
| ignored accepts (live 72.0; rdot 22.96, 27.76) counted as FA | 3 | 2 (rdot only) |
| total conservative non-Nova accepts | 6 | 2 |

These Ben recordings are **training data for the candidate**, so that column is in-sample. The only
out-of-sample Nova figure is leave-one-recording-out at the candidate's fixed accept 0.8 / decay 0.6
(`recovered-original-loro.txt`, earlier labels, 15 windows): Nova 13/15 (w171828 3/4, w172114 4/4, live
2/3, rdot 4/4), Ben non-word FA 0, HELD-corpus FA 0 in every fold. The inner folds picked accept 0.7 for
the `live` fold, where that held-out recording gave 2 dot false accepts (26.72, 36.08 s).

**Does the candidate eliminate the "Dot, dot. Hello, dot." false wake? Not demonstrated.** It does not
fire on that phrase in the `live` recording it was trained on, and in the held-out `live` fold it gave
0 false accepts at accept 0.8 but 2 at 0.7. Evidence is one speaker, one live session and a handful of
phrases; the held corpus is clean speech with zero accepts, which does not contain that phrase pattern as
far as these labels show. The result is a strong reduction against the old head on held-out speech
(19.2/h to 0 in 0.78 h) and on Ben's own recordings, not proof of elimination. A fresh recording of the
phrase that is not in training, replayed through the packaged engine, is the missing test.

### Target-only trial packaging — 2026-11-13

Ben's real-room report that **hello still wakes the Echo** fails acceptance of the
deployed old-head tuning. The newer MSWW candidate above has not been deployed;
the two attempts must not be conflated. Live settings inspection found only
`nova-en-US` selected (no second wake word), on version `2026.11.13.003932`.

Before a controlled MSWW trial, candidate packaging now keeps only the `NOVA`
spotter and its operation-state rules. Previously it retained stock spotters:
the daemon filters non-selected words, but `STOP` is exempt and can stop a
request. Decoder classes/rows are **not** removed: they still compete in the
softmax. The low-level packer keeps its multiword default for research callers;
`candidate.py pack` requests a single-word package. Its gate also rejects any
accepted non-target keyword instead of silently discarding those events.

Runnable checks and package construction (Amazon-derived output stays ignored):

```sh
python3 tools/msww/test_pack.py
python3 tools/msww/test_gate.py
build/pryon-review-venv/bin/python tools/msww/candidate.py pack \
  build/msww/cand-nova.json build/msww/computer-en-US build/msww/nova-only-trial-set
OPENBLAS_NUM_THREADS=1 build/pryon-review-venv/bin/python tools/msww/candidate.py gate \
  build/msww/cand-nova.json build/msww/nova-only-trial-set \
  build/msww/nova-only-trial-gate.json build/msww/rec/live.raw \
  build/msww/rec/w171828.raw build/msww/rec/w172114.raw --jobs 3 --speed 2
```

The packer refuses an existing destination. These three packaged-engine replays
passed with respectively 2/2, 4/4 and 4/4 labeled Nova detections, no labeled
negative accepts, and matching simulated count/time/score. Every callback
(including near misses) named `NOVA`. These are **training recordings**, not
fresh proof of hello rejection. Independent GPT-6 Luna source review approved
the target-only config correction; actual-device loading and held-out behavior
remain unverified.

Controlled trial delivery must preserve the existing model and daemon. Use a
separate ID such as `nova-en-US-trial` (its parsed keyword is `nova`, language
`en`), not `nova-msww-trial-en-US` (language would be `msww`). The signed artifact
API is the existing installation path: begin/chunk/commit validates the set,
including device `pryon_test` where installed; install requests the root watcher
to copy the set and restart **hassmic**, not the Echo or OpenClaw Gateway. Verify
the installed artifact digest before explicitly selecting the trial. Back up
the current wake-word selection first and restore `nova-en-US` explicitly on
rollback: a failed load falls back to Alexa, **not** the previous Nova model.
Keep the prior model available until the old selection is confirmed active.

ADB is currently offline despite an accepted signed debug-unlock request.
The short-lived browser diagnostic identity was not retained after reading
settings; that approval cannot authorize a different identity. Re-establish
supported access before trial delivery, persist any newly approved diagnostic
identity under ignored local build state, and do not invent an authentication
bypass. Nothing in this section establishes that the candidate is installed or
that the wake-word problem is fixed.
