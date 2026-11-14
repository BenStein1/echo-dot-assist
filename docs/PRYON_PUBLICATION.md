# Pryon publication and upstream routing

This is the standing handoff for future agents updating Pryon work. Read it
before publishing changes. Ben should not have to reconstruct this split.

## Destinations

| Destination | Purpose | Existing branch |
| --- | --- | --- |
| [Upstream PR #171](https://github.com/Gamer92000/echo-dot-assist/pull/171) | Runtime/DSP reconstruction, research tools, reproducible synthetic checks and supporting evidence | `fork/pr-pryon-native` |
| [Upstream PR #172](https://github.com/Gamer92000/echo-dot-assist/pull/172) | User-facing enrollment/training, packaging, validation, installation/recovery and required research dependencies | `fork/pr-train-wake-word` |
| [Source-only research repo](https://github.com/BenStein1/pryon-research) | Portable research source, synthetic demonstrations, numerical evidence and attribution; not the complete Echo application | Inspect that repo's current instructions and source inventory before exporting |

`fork` is BenStein1/echo-dot-assist; `origin` is Gamer92000/echo-dot-assist.
The branches above were verified on 2026-11-13 at `dd56117` and `37b3c43`.
These are historical checkpoints, not pins: query current PR state, head/base
and remote refs before each update. If a PR has merged or closed, reassess the
destination instead of blindly pushing its old branch.

## Update procedure

1. Inspect working-tree state and fetch relevant remotes. Preserve unrelated
   changes. Inspect the current PR diffs and the destination repo's instructions.
2. Classify each change by the table above. A shared algorithm may belong in
   both PRs; device/UI integration belongs in #172, not automatically #171.
   Include necessary dependencies, but not unrelated fork features.
3. Update the existing scoped PR branches with reviewed commits or narrowly
   ported changes. Do **not** merge the whole development branch or fork `main`
   into them. Inspect the full resulting PR diff against its actual base, not
   just the latest commit. Preserve attribution and avoid force-pushing.
4. Separately port the applicable research source into the source-only repo,
   following its existing layout/inventory and updating provenance. Do not
   mirror this checkout or copy its Git history wholesale. Do not assume the
   standalone repo shares this application's build paths or dependencies.
5. Review the actual staged files and complete destination diffs for personal
   data and unrelated content. Exclude personal recordings/transcripts,
   captured runtime tensors, proprietary libraries, firmware, Amazon-derived
   models/weights, credentials, device-specific configuration and generated
   build artifacts. `.gitignore` alone is not an audit. Publish source,
   synthetic fixtures and appropriately summarized evidence instead.
6. Run each destination's documented build/tests, independently review changes,
   then commit and push the authorized changes to the corresponding existing
   remote branches. Verify remote hashes and PR diffs. Report each destination
   separately; pushing a development branch is not updating the upstream PRs
   or synchronizing the research repo. A PR update is not an upstream merge.

## Current MSWW follow-up

Development branch `fix/custom-wake-false-accepts` contains target-only packaging
(`b4b1d0e`), reusable per-word host generation (`893d304`) and native settings
enrollment (`578177a`). These are implementation references, **not** instructions
to cherry-pick all three wholesale into every destination.

- Route reusable MSWW research/training algorithms, packaging/gate source and
  synthetic tests to the applicable research destinations after dependency review.
- Route native enrollment/UI integration, build integration and installation
  safeguards to #172 with the required shared code.
- Keep Nova-specific personal datasets and device artifacts out of all exports.
- These development commits being pushed does not establish that either PR or
  the standalone repo contains them. Verify and record each transfer explicitly.
- Preserve the accuracy limits: Ben reported a live false wake on “greater
  project” after deployment. This is a reported regression case, not a retained
  audio fixture or measured rate. Passing synthetic/replay checks is not proof
  of reliable recognition across speakers, words or rooms.

This document records the workflow; its creation does not update either PR or
the standalone repository.
