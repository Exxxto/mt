# MuMuPlayer ADB Screenshot Mode

## Intent

Keep the existing `exec-out` screenshot path as the default while adding an explicit `pull` mode for MuMuPlayer, whose ADB can save a PNG on-device and download it with `adb pull`.

## Scope and constraints

- Select the mode with `ADB_SCREENSHOT_MODE=exec-out|pull`; unset means `exec-out`.
- Keep `ADBController.screenshot()` and its returned OpenCV image unchanged.
- In `pull` mode, use unique remote and local temporary paths, check command results and timeouts, validate/decode the PNG, clean up created files, and keep using `_adb_cmd()` so `ADB_SERIAL` is honored.
- No retries, stale-frame fallback, dependency changes, strategy/diagnostic changes, gameplay taps, emulator restarts, or BAT-file edits.
- Verify with mocked tests and, when available, read-only screenshot captures from MuMuPlayer.

## Approach

Add a small mode branch in `core/adb_controller.py`; preserve the current `exec-out` implementation and route `pull` through isolated temporary files and checked ADB subprocesses. This keeps compatibility and lets the user choose the slower path only when needed.

## Files

- `core/adb_controller.py`: mode selection and pull implementation.
- `tests/test_adb_controller.py`: isolated success, failure, timeout, cleanup, serial, and no-stale-frame coverage.
