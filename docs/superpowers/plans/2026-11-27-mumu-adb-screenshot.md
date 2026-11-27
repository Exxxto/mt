# MuMuPlayer ADB Screenshot Mode Implementation Plan

**Goal:** Add an opt-in MuMu-compatible screenshot mode while retaining `exec-out` as the default.

**Architecture:** Keep the existing public screenshot method and image format. Branch internally on `ADB_SCREENSHOT_MODE`; the new `pull` path uses unique remote/local temporary files, checked ADB commands, finite timeouts, PNG validation, and best-effort cleanup.

**Tech Stack:** Python standard library, existing OpenCV and NumPy, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-27-mumu-adb-screenshot-design.md`.

## Global Constraints

- No changes to game strategy, diagnostics, dependencies, or the user BAT file.
- No automatic retry, old-frame fallback, real game actions, emulator restart, or device disconnect.
- Use `_adb_cmd()` for every ADB invocation so the selected serial remains bound.

## Review Focus

- A failed `screencap` or `pull` must raise and remove its own remote file.
- Missing, empty, or invalid local PNG data must fail instead of returning an old frame.
- Each call must use a fresh remote path, and local temporary files must be removed.
- Every `pull` command must use finite timeouts and the configured serial.
- Unset mode must preserve the existing `exec-out` path.

---

### Task 1: Screenshot mode behavior

**Files:**
- Modify: `core/adb_controller.py`
- Create: `tests/test_adb_controller.py`

- [x] Add failing tests for default mode, successful pull/decode, ADB serial binding, command failures, timeout, missing/invalid PNG, temporary cleanup, unique remote paths, and no stale-frame fallback.
- [x] Run the focused tests and confirm they fail for the missing mode.
- [x] Implement the smallest internal branch and pull helper that satisfy the tests without changing `screenshot()`'s signature or output.
- [x] Run focused tests, then the full unittest suite (81 tests pass).
- [x] Run `git diff --check`; inspect changed paths to ensure no runtime artifacts are staged.
- [ ] Commit as `fix: add compatible ADB screenshot mode for MuMuPlayer`.

### Task 2: MuMuPlayer measurement and publication

**Files:**
- No additional source files.

- [x] Measure 20 read-only frames for each mode. Direct `exec-out`: 20/20, 464.0/460.2/515.8 ms mean/median/max; direct `pull`: 20/20, 504.5/503.1/518.3 ms; implemented pull: 20/20, 498.9/495.8/529.7 ms. All 900×1600.
- [ ] Run the full unittest suite and `git diff --check` on the final commit.
- [ ] Fast-forward the verified commit into `main`, push `origin main` without force, and confirm local `main` equals `origin/main`.
