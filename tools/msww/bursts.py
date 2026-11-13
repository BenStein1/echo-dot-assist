"""Speech bursts of a 16 kHz s16le recording by frame energy: list of (start_s, end_s, peak_dbfs)."""
import numpy as np

def bursts(raw, margin_db=14.0, min_s=0.12, merge_s=0.35, frame=160):
    x = np.fromfile(raw, '<i2').astype(np.float64) if isinstance(raw, str) else raw.astype(np.float64)
    n = len(x) // frame
    e = 10 * np.log10(np.maximum((x[:n * frame].reshape(n, frame) ** 2).mean(1), 1.0))
    floor = np.percentile(e, 20)
    on = e > floor + margin_db
    out, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or (j < n and on[j:j + int(merge_s * 100)].any())):
                j += 1
            out.append([i / 100.0, j / 100.0, float(e[i:j].max())])
            i = j
        else:
            i += 1
    return [b for b in out if b[1] - b[0] >= min_s]
