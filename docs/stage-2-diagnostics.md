# Stage 2: structured diagnostics

Diagnostics are opt-in. With the feature disabled, the bot follows its existing action path. Events are written synchronously to preserve their order; optional screenshots use a four-item bounded background queue. A full queue drops the newest frame and increments a counter.

## Start and inspect a session

Run these commands from the repository root in PowerShell:

```powershell
# Event log only
.\.venv\Scripts\python.exe .\bot.py --diagnostics

# Event log plus sampled source screenshots
.\.venv\Scripts\python.exe .\bot.py --no-search --duration 600 `
  --record-frames --diagnostics-dir .\recordings --diagnostics-max-mb 512

# Summarize a session; add --verbose for the JSON event timeline
.\.venv\Scripts\python.exe .\utils\analyze_recording.py `
  .\recordings\<session-directory> --verbose

# Run automated checks
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--record-frames` enables diagnostics automatically. The per-session size limit must be a positive integer MiB value. Plain `python bot.py` creates no recording directory. Each run gets a unique directory containing `metadata.json`, ordered `events.jsonl`, and `summary.json`; `frames/` exists only when screenshots were requested. Frames are original-size PNGs with relative paths in events. A frame first seen by identity can still be saved when a later existing code path requests its capture.

## Event schema

Schema version 1 records timezone-aware timestamps, monotonic elapsed milliseconds, sequential `event_id`, session/match/round context, phase, optional frame/action IDs, source, and a JSON `data` object. Unknown values use `null`. Event payloads are frozen and flushed when emitted.

Event families currently include:

- Session/match/phase: `session_started`, `match_boundary_assumed`, `phase`, `phase_unknown`, `match_search_attempt`, `match_search_tap_returned`, `match_end_screen`, `end_screen_action_attempt`, `end_screen_action_returned`.
- Frames: `frame_seen`, `frame_reused`, `frame_dropped`. `frame_seen.data.path` is the expected relative PNG path when frame recording is enabled; `summary.json` distinguishes saved and dropped frames.
- State and choices: `state_snapshot`, `decision`, `purchase_refused`, `purchase_limit`.
- Purchases and calculated upgrades: `buy_attempt`, `buy_tap_returned`, `buy_outcome`, `buy_simulated`, `calculated_upgrade`.
- Orbs and sale attempts: `orb_detection`, `orb_tap_attempt`, `orb_tap_returned`, `sale_attempt_started`, `sale_slot_tap_attempt`, `sale_slot_tap_returned`, `sale_slot_probe`, `sale_button_tap_attempt`, `sale_outcome`.
- Errors: `adb_error`, `screenshot_error`. Error events store the operation and exception type without raw exception text, since ADB stderr can contain a device serial.

Decision records preserve offered shop order, recognized cost/source and name inliers when available, slot index, affordability, duplicate counts, copies to the next calculated threshold, target rank, selected candidate, reason, and any tie break. A slot is resolved against the known shop centers; unknown matches remain null.

## Confidence labels and limits

- `observed` means a value came from an existing recognition/read path, such as the displayed elixir or a detected phase.
- `inferred` means it follows from existing code evidence, such as the same card still being offered after a tap.
- `assumed` marks a legacy fallback, such as subtracting a card cost when the next elixir read is unavailable, or the initial round index after `start_run()`.
- `calculated` marks strategy bookkeeping, such as a copy count reaching a star threshold.
- `unknown` means the available screen or action result cannot establish the value.

`buy_tap_returned` only says the ADB call returned; it is not confirmation from the device. A `buy_outcome` accepted by the legacy logic or `RunState` copies/purchases are strategy-accounted, not a read of the real bench. A calculated upgrade does not prove that the game merged units. Sale button attempts likewise do not establish a sale result. The bot does not infer a real match ID from gameplay state: `match_boundary_assumed` marks the local reset boundary.

When `metadata.json` or `summary.json` is missing, or an event line is malformed/truncated, the analyzer reports a partial session and keeps every valid JSONL row. It only reads these three files and never imports the bot, ADB, OpenCV, or NumPy. Summary counters include dropped events/frames, writer errors, phase/decision/action totals, file sizes, truncation, and finish reason. Diagnostic initialization/write/close errors warn once where possible and do not replace gameplay exception handling.

## Handoff and manual checks

The design decisions are captured in [the Stage 2 design spec](superpowers/specs/2026-11-27-stage-2-diagnostics-design.md) and [implementation plan](superpowers/plans/2026-11-27-stage-2-diagnostics.md). The implementation is in `core/diagnostics.py`, the existing observation/action points in `bot.py`, `core/game_logic.py`, `core/strategy.py`, and `core/board_state.py`, plus the read-only utility at `utils/analyze_recording.py`; automated coverage is in `tests/test_diagnostics.py`.

Automated checks exercise a synthetic planner trace and temporary recording files. No live device, real match, Android tap, screenshot-recognition run, or visual review of recorded gameplay has been performed. Check those manually before relying on recordings to diagnose a live match.
