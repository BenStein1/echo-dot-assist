---
name: Echo Dot Assist
status: active
summary: Amazon Echo voice satellite for Home Assistant.
next: Independently review native MSWW enrollment and validate a fresh device session.
updated: 2026-11-13
olivetin: false
---

# Echo Dot Assist

## Current state

Nova MSWW trial is installed separately from the prior model; Ben reports the
fresh room test seemed to work. Host generation now accepts explicit per-word
manifests and requires strict held-recording packaged-engine replay. Settings
page enrollment now selects native MSWW when the stock source is available;
legacy-only devices retain their old backend.

## Next

Review the host and native pipeline changes independently. Native host/ARM
synthetic checks and a genuine engine load check validate plumbing; a fresh
on-device enrollment session remains unverified. Do not equate those checks
with accuracy evidence for a new word.

## Notes

Research and limitations: `docs/re-pryon-msww.md`. Proprietary models and personal
audio remain in ignored `build/`. No automatic installation in host generation.
