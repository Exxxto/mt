# Stage 3 Regression Fixes Implementation Plan

> **For agentic workers:** Execute natively in this session. Keep the user's existing untracked files untouched.

**Goal:** Make verified play resilient to uncertain purchases and stale observations while preserving evidence-based accounting.

**Architecture:** Extend the current `Observation` and `ActionVerifier` with independent transaction evidence and a bounded pending transaction. Track monotonic frame receipt times in the bot and planner, reject stale coordinates before taps, and retry decisions from fresh screenshots. Diagnostics will distinguish observed and calculated elixir and report sync restrictions.

**Tech Stack:** Python 3.12, unittest, OpenCV/NumPy already installed in `.venv`.

**Spec:** `docs/stage-3-action-verification.md` and the user-provided Stage 3 regression requirements.

## Global Constraints

- Do not change legacy strategy, character ranking, tier lists, or economy decisions.
- Do not add board/bench roster recognition, live-match testing, or user BAT edits.
- Keep unconfirmed purchases out of `RunState`; never repeat the same tap or sell blindly.
- Apply the stale-frame check to verify-mode purchases, orb collection, and other game taps.
- Keep `.venv/`, recordings, test_screen.png and the user's BAT file out of the commit.

## Review Focus

- Fully spent elixir with dim/unrecognized shop: must not become a false negative or a false positive without corroboration.
- Merge purchases: unit count can remain constant or fall, so it is corroborating evidence only when it increases.
- Fresh but conflicting delayed observations: resolve once and apply at most once by action ID.
- Stale purchase/orb coordinates: discard and re-decide from a new capture.
- Unresolved roster state: allow only decisions independent of copy counts; block duplicate/merge decisions.

## Files

- `core/action_verifier.py`: evidence combination and pending-resolution classification.
- `core/strategy.py`: pending transactions, safe mode, fresh-frame checks, balance provenance and sync status in snapshots.
- `bot.py`: monotonic timestamps, verify-mode tap guards, startup strategy diagnostic, long-pause diagnostics and CLI threshold.
- `tests/test_action_verifier.py`, `tests/test_strategy.py`, `tests/test_diagnostics.py`: focused regression tests.
- `docs/stage-3-action-verification.md`: Stage 3 rules, limits, commands and offline results.

## Tasks

- [x] Add failing verifier tests for exact full-spend plus unit increase, dim/unrecognized shop, changes without matching elixir, and contradictory evidence; implement conservative classification.
- [x] Add failing planner tests for pending UNKNOWN resolution, no duplicate accounting, limited safe choices, no blind sale, and final snapshot balance provenance; implement the pending transaction and safe mode.
- [x] Add failing frame-age tests around the configurable 3-second boundary, 35s/62s stale frames, drop-and-refresh decision flow, and verify-mode taps; implement monotonic timestamps and diagnostics.
- [x] Add diagnostics tests for strategy name, stale-action reasons, blocked-purchase reasons, pending resolution, and long pauses; update event and snapshot values.
- [x] Run existing and added tests, offline-analyze both current and previous sessions, run `git diff --check`, and update Stage 3 documentation.
- [ ] Commit as requested and push to `origin/main` without force.
