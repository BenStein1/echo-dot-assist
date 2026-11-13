#!/usr/bin/env python3
"""Crop context-preserving native PRYF windows from continuous-stream features.

Events JSON is an array of {source, start, end, label, split, text, speaker?}.
Times delimit the supervised interval in seconds of the original stream. Each
window retains at least 1.8 seconds of preceding state for the trainer's warmup.
Generated features and their source audio must remain private.
"""
import argparse
import json
import math
import pathlib
import struct

HEADER = struct.pack("<4I", 0x46595250, 1, 25, 100)
CYCLE_BYTES = 25 * 100 * 4


def read_window(source, start, end):
    if not all(math.isfinite(v) for v in (start, end)) or start < 1.8 or end <= start:
        raise ValueError("window needs finite start >=1.8s and end > start")
    # Integer arithmetic at the 60ms cycle boundary avoids a floating-point
    # ceil of 1.8000000000000003 accidentally dropping the required warmup.
    first = math.floor(start / .06 + 1e-9) - 30
    last = math.ceil(end / .06 - 1e-9)
    if not 31 <= last - first <= 180:
        raise ValueError("window plus context must fit 31..180 cycles")
    with pathlib.Path(source).open("rb") as stream:
        if stream.read(16) != HEADER:
            raise ValueError("not native PRYF features")
        stream.seek(0, 2)
        size = stream.tell()
        if (size - 16) % CYCLE_BYTES or last > (size - 16) // CYCLE_BYTES:
            raise ValueError("window exceeds complete source cycles")
        stream.seek(16 + first * CYCLE_BYTES)
        data = stream.read((last - first) * CYCLE_BYTES)
    if len(data) != (last - first) * CYCLE_BYTES:
        raise ValueError("source changed during read")
    # Native recurrent lanes are indexed by GLOBAL cycle modulo25. The trainer
    # restarts its cycle at0, so row r must become global row(r+first)%25.
    shift = (first % 25) * 100 * 4
    result = bytearray(HEADER)
    for offset in range(0, len(data), CYCLE_BYTES):
        cycle = data[offset:offset + CYCLE_BYTES]
        result.extend(cycle[shift:] + cycle[:shift])
    return result


def prepare(manifest, output):
    manifest, output = pathlib.Path(manifest), pathlib.Path(output)
    events = json.loads(manifest.read_text())
    if not isinstance(events, list) or not 1 <= len(events) <= 200:
        raise ValueError("need 1..200 explicit events")
    output.mkdir()  # Never overwrite existing features or an installed corpus.
    rows = []
    for i, event in enumerate(events):
        if type(event.get("label")) is not int or event.get("label") not in (0, 1) or event.get("split") not in ("train", "test", "diagnostic"):
            raise ValueError("every event needs explicit label and split")
        if not isinstance(event.get("text"), str) or not event["text"].strip():
            raise ValueError("every event needs an accurate transcript")
        name = f"window-{i:03d}.bin"
        data = read_window(manifest.parent / event["source"], float(event["start"]), float(event["end"]))
        (output / name).write_bytes(data)
        row = {k: event[k] for k in ("label", "split", "text", "source", "start", "end")}
        if "speaker" in event:
            row["speaker"] = event["speaker"]
        row["path"] = name
        rows.append(row)
    (output / "features.json").write_text(json.dumps(rows, indent=2) + "\n")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    print(f"Prepared {prepare(args.event_manifest, args.output)} context-preserving windows")
