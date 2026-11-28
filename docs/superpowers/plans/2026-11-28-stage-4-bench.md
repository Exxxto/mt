# Bench observation implementation plan

**Goal:** Observe the five reserve slots directly from recorded or live frames without influencing gameplay.

**Architecture:** An isolated reader validates frame geometry, assesses the wooden background and matches existing bench-model templates conservatively. The planner sends already captured frames to an optional diagnostics observer. A local inspection utility renders evidence and crops.

**Tech stack:** Python, OpenCV, NumPy, unittest.

**Spec:** Stage 4 user request supplied in the current session.

## Constraints and review focus

- Preserve strategy, previous action verification, and performance when disabled.
- Distinguish unknown from empty, and identity and stars from occupancy.
- Ignore captures outside 900×1600; handle missing assets and overlays.
- Keep real recordings and generated images out of Git.
- Evaluate on visually labelled frames split by match; report limitations honestly.

## Tasks

1. Inspect images, existing template usage, geometry, occlusion and separate development and evaluation matches.
2. Test and implement conservative per-slot observation, with separate evidence for occupancy and identity and unknown stars where unsupported.
3. Test and connect optional diagnostics to the initial deploy and already captured post-action and verification frames; summarize observations in the recording analyzer.
4. Add the offline inspection tool and documentation; evaluate labelled held-out frames and measure time.
5. Run the full test suite, inspect the diff, commit and publish the requested commit if all completion criteria are met.
