"""Lane phase and context regression; deliberately shifts by a non-multiple of25."""
import importlib.util
import pathlib
import struct
import tempfile

spec = importlib.util.spec_from_file_location("stream_windows", pathlib.Path(__file__).parents[2] / "tools/pryon-native/stream_windows.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as tmp:
    source = pathlib.Path(tmp) / "continuous.bin"
    # Every row has a recognizable global cycle/lane, not a model or recording.
    with source.open("wb") as out:
        out.write(module.HEADER)
        for cycle in range(100):
            for row in range(25):
                out.write(struct.pack("<100I", *([cycle*100+row]*100)))
    # Supervision begins at cycle47; preceding30cycles start at global17.
    cropped = module.read_window(source, 47*.06, 60*.06)
    assert len(cropped) == 16 + 43*10000
    for local_cycle in range(43):
        for local_row in range(25):
            offset = 16 + local_cycle*10000 + local_row*400
            got = struct.unpack_from("<I", cropped, offset)[0]
            assert got == (17+local_cycle)*100 + (local_row+17)%25
    for start, end in ((1, 2), (3, 2), (3, 99), (float("nan"), 4)):
        try:
            module.read_window(source, start, end)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid timing accepted")
print("stream_windows:30-cycle context and global-to-local recurrent lane phase passed")
