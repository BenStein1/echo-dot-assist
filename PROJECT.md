---
name: Echo Dot Assist
status: active
summary: Amazon Echo voice satellite for Home Assistant.
next: Integrate the MSWW trainer into on-device enrollment after host pipeline review.
updated: 2026-11-13
olivetin: false
---

# Echo Dot Assist

## Current state

Nova MSWW trial is installed separately from the prior model; Ben reports the
fresh room test seemed to work. Host generation now accepts explicit per-word
manifests and requires strict held-recording packaged-engine replay. Settings
page enrollment still uses the older Alexa binary head.

## Next

Review the host pipeline, then integrate MSWW extraction, pool conversion,
quantized row writing and replay validation into the on-device enrollment path.

## Notes

Research and limitations: `docs/re-pryon-msww.md`. Proprietary models and personal
audio remain in ignored `build/`. No automatic installation in host generation.
