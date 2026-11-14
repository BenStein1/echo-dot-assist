"""Runnable corpus provisioning regression; no firmware or personal audio."""
import importlib.util
import json
import pathlib
import struct
import tempfile

spec = importlib.util.spec_from_file_location("prepare_negatives", pathlib.Path(__file__).parents[2] / "tools/pryon-native/prepare_negatives.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    model = root / "stock.onnx"
    model.write_bytes(b"abc")
    manifest = root / "features.json"
    rows = []
    for i, split in enumerate(("train", "test")):
        name = f"features-{i}.bin"
        (root / name).write_bytes(struct.pack("<4I", 0x46595250, 1, 25, 100) + b"\0"*(31*10000-4) + struct.pack("<f", i+1))
        rows.append({"label": 0, "split": split, "path": name, "text": f"example {i}", "speaker": f"speaker {i}"})
    manifest.write_text(json.dumps(rows))
    assert module.prepare(model, manifest, root / "pack") == {"train": 1, "test": 1}
    bundle = json.loads((root / "pack/manifest.json").read_text())
    assert bundle["source_sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert [r["split"] for r in bundle["recordings"]] == ["train", "test"]
    for field in ("text", "speaker", "path"):
        changed = [dict(r) for r in rows]
        changed[1][field] = changed[0][field]
        manifest.write_text(json.dumps(changed))
        try:
            module.prepare(model, manifest, root / ("bad-" + field))
        except ValueError:
            pass
        else:
            raise AssertionError(f"cross-split {field} was accepted")
        assert not (root / ("bad-" + field)).exists()
    # Native MSWW enrollment keeps the exact raw streams for real-engine replay.
    for i, row in enumerate(rows):
        (root / row['path']).write_bytes(struct.pack('<4I', 0x46595250, 2, 512, 15) +
                                        b'\0' * (31 * 528 * 4 - 4) + struct.pack('<f', i+1))
        row['raw'] = f'negative-{i}.raw'
        (root / row['raw']).write_bytes(struct.pack('<h', i+1) * (31 * 1280))
    manifest.write_text(json.dumps(rows))
    module.prepare(model, manifest, root / 'msww', msww=True)
    bundle = json.loads((root / 'msww/manifest.json').read_text())
    for i, row in enumerate(bundle['recordings']):
        assert (root / 'msww' / row['raw']).read_bytes() == (root / rows[i]['raw']).read_bytes()
    (root / rows[1]['raw']).write_bytes((root / rows[0]['raw']).read_bytes())
    try:
        module.prepare(model, manifest, root / 'bad-msww', msww=True)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate MSWW raw audio accepted')
print("prepare_negatives: bounded bundles, MSWW raw replay inputs and train/test isolation passed")
