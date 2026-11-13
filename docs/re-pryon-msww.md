# Amazon's multi-word wake word sets (msww): how five words share one engine

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
