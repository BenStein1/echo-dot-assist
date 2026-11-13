#!/usr/bin/env python3
"""Build a private reusable negative-feature bundle for on-device enrollment.

Input: the trainer's flat feature manifest, with a `text` transcript on each
negative row (and optionally `speaker`). Source features must have been extracted
with this stock ONNX model. Do not publish the generated native features.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import struct


def prepare(model, manifest, output, per_split=16):
    if not 1 <= per_split <= 16:
        raise ValueError("per-split must be between 1 and 16")
    model, manifest, output = map(pathlib.Path, (model, manifest, output))
    rows = json.loads(manifest.read_text())
    if not isinstance(rows, list):
        raise ValueError("feature manifest must be an array")
    selected = []
    counts = {"train": 0, "test": 0}
    texts = {"train": set(), "test": set()}
    speakers = {"train": set(), "test": set()}
    digests = set()
    for row in rows:
        if row.get("label") != 0 or row.get("split") not in counts:
            continue
        split = row["split"]
        if counts[split] >= per_split:
            continue
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 900:
            raise ValueError("every selected negative needs a short, accurate transcript")
        source = manifest.parent / row["path"]
        data = source.read_bytes()
        if not 16 + 31*10000 <= len(data) <= 16 + 180*10000 or (len(data)-16) % 10000:
            raise ValueError(f"feature clip outside 31..180-cycle budget: {source}")
        if struct.unpack("<4I", data[:16]) != (0x46595250, 1, 25, 100):
            raise ValueError(f"invalid PRYF header: {source}")
        digest = hashlib.sha256(data).digest()
        if digest in digests:
            raise ValueError("duplicate feature audio in corpus")
        digests.add(digest)
        texts[split].add(" ".join(re.findall(r"[a-z0-9]+", text.lower())))
        entry = {"path": f"negative-{len(selected):02d}.bin", "split": split, "text": text}
        if row.get("speaker"):
            if not isinstance(row["speaker"], str) or len(row["speaker"].encode()) > 100:
                raise ValueError("speaker must be a short string")
            speakers[split].add(row["speaker"])
            entry["speaker"] = row["speaker"]
        selected.append((source, entry))
        counts[split] += 1
    if not all(counts.values()):
        raise ValueError("need both training and held-out negative features")
    if texts["train"] & texts["test"] or speakers["train"] & speakers["test"]:
        raise ValueError("train/test must have disjoint transcripts and speakers")
    source_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    output.mkdir()  # Never overwrite an installed or previously generated bundle.
    for source, row in selected:
        shutil.copyfile(source, output / row["path"])
    bundle = {"source_sha256": source_sha256, "recordings": [row for _, row in selected]}
    (output / "manifest.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock_onnx")
    parser.add_argument("feature_manifest")
    parser.add_argument("output")
    parser.add_argument("--per-split", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(prepare(args.stock_onnx, args.feature_manifest, args.output, args.per_split)))
