"""The multi-wakeword spotter's decision rule (exponential-decay classifier), reconstructed from the engine's own RESULTs.

Per decoder cycle t (1280 samples, one record of pryon_extract -w), with p_t the softmax posterior of the word's class:
    s_t = decay * s_{t-1} + (1 - decay) * p_t                      (s_{-1} = 0)
accept at the first unlocked cycle with s_t >= accept; detection sample 1280 * (t + 1); score = s_t; then locked.
Near-miss (type 0): peak s of a run >= notify that never reached accept (see events()).
"""
import numpy as np
from feat import CYCLE


def smooth(p, decay):
    s = np.zeros(len(p)); a = 0.0
    for t, v in enumerate(p):
        a = decay * a + (1 - decay) * v
        s[t] = a
    return s


def events(p, decay=0.6, accept=0.523, notify=0.12, lock=100, window=100, frame=160):
    """-> list of (type, score, detection_sample, cycle). lock/window in 10 ms frames; type 2 accept, 0 near-miss."""
    s = smooth(p, decay)
    out, lock_until, run = [], -1, None   # run: (peak, cycle)
    for t, v in enumerate(s):
        end_frame = 8 * (t + 1)
        locked = end_frame <= lock_until
        if not locked and v >= accept:
            out.append((2, v, CYCLE * (t + 1), t)); lock_until = end_frame + lock; run = None; continue
        if v >= notify and not locked:
            if run is None or v > run[0]:
                run = (v, t)
        elif run is not None and 8 * (run[1] + 1) + window <= end_frame:
            out.append((0, run[0], frame * (8 * (run[1] + 1) + window), run[1])); run = None
    return out
