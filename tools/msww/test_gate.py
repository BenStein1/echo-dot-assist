"""Run with python3 tools/msww/test_gate.py."""
import json
import tempfile
from pathlib import Path
from gate import audio_loss

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'events.res'
    path.write_text('')
    assert not audio_loss('max_audio_backlog.max_backlog_limit_msec = 7000\nPerformed 1 tasks and skipped 0 during shutdown', path)
    assert audio_loss('Performed 1 tasks and skipped 2 during shutdown', path)
    assert audio_loss('dropped 160 samples', path)
    for value in (0, 80):
        path.write_text('META header 24 bytes, ' + json.dumps(dict(deviceMetrics=dict(
            max_audio_backlog_msec=100, skipped_audio_backlog_since_last_ww_msec=value))) + '\n')
        assert audio_loss('', path) == bool(value)
    path.write_text('META header 24 bytes, ' + json.dumps(dict(audioMetadataDuringDetection=dict(dropFramesCount=1))) + '\n')
    assert audio_loss('', path)
print('replay loss checks passed')
