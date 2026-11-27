# Stage 2 Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, bounded structured session recording and a read-only text analyzer without changing the bot's gameplay behavior.

**Architecture:** A small `core/diagnostics.py` session owns metadata, ordered JSONL events, counters, size limits, and a bounded background queue for immutable image copies. Existing loop/planner paths pass already available observations to it; disabled mode takes the original path. `utils/analyze_recording.py` independently reads partial or complete sessions.

**Tech Stack:** Python 3.12, standard library (`argparse`, `json`, `threading`, `queue`, `datetime`, `uuid`, `weakref`, `unittest`, `tempfile`), installed OpenCV/NumPy only for already available frame encoding.

**Spec:** `docs/superpowers/specs/2026-11-27-stage-2-diagnostics-design.md` plus the user's attached Stage 2 requirements.

## Global Constraints

- Diagnostics are off by default; `--record-frames` implies `--diagnostics`.
- Preserve `--no-search` and `--duration`; do not edit `Start Merge Tactics Bot.bat` or `.venv/`.
- No strategy, sale, recognition, coordinates, delay, purchase-limit, or ADB behavior changes.
- No extra screenshots, recognition passes, ADB commands, network calls, dependencies, real matches, or automatic deletion of recordings.
- Do not serialize credentials, environment variables, or device serials.
- Distinguish observed, inferred, assumed, and unknown values; never describe RunState as actual board/bench contents.
- Keep event order, unique session/event/frame/action IDs, schema version, UTF-8 JSONL, relative frame paths, size bounds, bounded image queue, and bounded shutdown.
- Publish only to `https://github.com/Exxxto/mt`, branch `stage-2-diagnostics`; no force push or merge to `main`.

## Review Focus

- **Diagnostics disabled:** no file access, snapshots, diagnostic-only computation, extra ADB calls, sleeps, or changed results. Pin with a mocked parity test comparing the exact events and outputs of `DeployPlanner.play()` and bot loop.
- **Decision ties and unknown elixir:** retain the current sort/min stability and return value while correctly labeling `None`/unaffordable cases. Pin each current branch and tie behavior in diagnostics tests.
- **Mutable/reused frames and queue saturation:** background work must own a copy, never block indefinitely, and report dropped frames. Pin immutability, repeated-frame IDs, overflow, and shutdown timeout.
- **Partial/corrupt recordings and filesystem failures:** valid JSONL prefix remains analyzable; diagnostic failures warn once and do not alter gameplay error handling. Pin truncated/malformed lines, failing writes, and cap reached.
- **Heuristic action outcomes:** a no-exception ADB tap and RunState update are not device confirmation; sell `True` is only the existing tap path. Pin provenance labels for purchase, sale, and calculated star threshold.

---

### Task 1: Session writer and bounded frame storage

**Files:**
- Create: `core/diagnostics.py`
- Modify: `.gitignore`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Produces `DiagnosticsSession.create(root, metadata, record_frames=False, max_bytes=536870912)`, `.event(event_type, *, data=None, frame_id=None, action_id=None, match_id=None, round_index=None, phase=None, source=None)`, `.record_frame(frame, *, reason) -> int | None`, `.frame_id(frame) -> int | None`, `.new_action_id() -> str`, and `.close(reason)`.
- A disabled/no-op object provides the same small surface without filesystem I/O or frame/event snapshot work.
- Frames enter a queue of at most four immutable image copies; a single worker encodes/writes them. JSONL events are synchronous and ordered. Default total-session limit is 512 MiB, configurable later by CLI.

- [x] **Step 1: Add tests for unique sessions, UTF-8 metadata, JSONL schema/order, no-op silence, bounded frame queue, copied frame immutability, size cap, writer errors, bounded close, and final summary counters/file sizes.** Assert `schema_version == 1`, sequential integer event IDs, relative frame paths, and `null` serialization for unknown values.
- [x] **Step 2: Run `\.venv\Scripts\python.exe -m unittest discover -s tests -p test_diagnostics.py -v`; confirm new API tests fail before implementation.** Initial RED was expected `ModuleNotFoundError: core.diagnostics`.
- [x] **Step 3: Implement the session, serializer, queue worker, counters, safe one-time warning, cap handling, and NoOp in `core/diagnostics.py`; add `recordings/` to `.gitignore`.** Event data is frozen/serialized at emit time; frame arrays are copied before enqueue.
- [x] **Step 4: Re-run the focused diagnostics tests and confirm all pass; verify a full queue drops a frame without blocking the producer.** Six diagnostics tests pass; existing + new full suite passed 35/35.

### Task 2: Record factual decision reasons

**Files:**
- Modify: `core/strategy.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- `DeployPlanner` accepts an optional diagnostics object while preserving existing constructor calls. `choose(cards, elixir, state)` keeps its public return contract; an optional keyword-only diagnostics context may be added only if required by tests and internal callers.
- Event payload records ordered offered cards with a slot index resolved from the existing click center against `SHOP_CENTERS` (null if unmatched), affordable/unaffordable candidates, duplicate counts and distance to next calculated threshold, target rank when present, and chosen card/reason or save reason.

- [x] **Step 1: Add tests for duplicate, target, cheapest fallback, original-order tie, insufficient elixir, and empty shop decision evidence.** Assert chosen card equals existing return and reason matches the branch actually taken. Cover unknown elixir as a separate `play()` refusal/error event; do not call `choose()` with `None`.
- [x] **Step 2: Run the focused diagnostics tests and confirm the new reason assertions fail.** RED was the expected missing `diagnostics` constructor argument.
- [x] **Step 3: Instrument the existing branches in `choose()` without moving/reordering conditions, sorts, `min()`, or returns; build diagnostic-only snapshots only when enabled.** Label RunState counts/stars as strategy-accounted/calculated.
- [x] **Step 4: Run `\.venv\Scripts\python.exe -m unittest discover -s tests -p test_strategy.py -v` and the diagnostics test file.** Strategy tests 29/29, diagnostics tests 10/10, full suite 39/39 passed.

### Task 3: Wire events through existing observations and actions

**Files:**
- Modify: `bot.py`
- Modify: `core/game_logic.py`
- Modify: `core/strategy.py`
- Modify: `core/board_state.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- A single diagnostics instance is created by CLI and passed into the bot and logic/planner; default direct construction remains compatible.
- Existing screenshot returns in `bot.run()`, `DeployPlanner.play()`, and `_sell_one()` can be registered, but diagnostics never calls screenshot itself. `frame_id(frame)` reuses the ID for the same live frame object.
- Session events cover start/reset, frame/phase/state snapshot, deploy/battle/lobby/unknown and end screens, decisions, buy attempt/outcome, calculated star-threshold change, sale search/attempt, orb detection/tap attempt, recognized errors, reused frames, and diagnostics errors/limits. State snapshots distinguish recognized phase/elixir/cards/orbs/approximate own units from strategy-accounted copies, purchase history and calculated stars. Carry existing `BoardReader.shop_cards()` tile index, cost source and identifier inliers through optional `ShopCard` fields; do not repeat recognition.

- [x] **Step 1: Add mocked integration tests asserting event order and shared action IDs across frame, state snapshot, decision, tap attempt, and next existing screenshot; cover orb calls from both sources, multi-slot sale search, screenshot failure, unknown phase/elixir/no cards, repeated frame use, and start_run/partial-match labeling.**
- [x] **Step 2: Run focused tests and confirm missing event assertions fail.** RED covered missing decision/action/sale/phase instrumentation; targeted integration tests now pass.
- [x] **Step 3: Add only guarded event calls at current data/action points.** Preserve existing `try`/`except`, ADB sequence, waits, return values and RunState mutations. ADB tap with no exception is `attempted`, not confirmed; unknown post-purchase elixir stays null alongside the strategy's assumed balance.
- [x] **Step 4: Run focused integration tests and compare mocked ADB/frame/sleep/RunState traces with diagnostics on and off.** Diagnostics and baseline strategy tests pass; full current suite 50/50. Exact on/off trace parity remains in Task 6.

### Task 4: CLI lifecycle and session metadata

**Files:**
- Modify: `bot.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Add `--diagnostics`, `--record-frames`, `--diagnostics-dir`, and `--diagnostics-max-mb` (positive integer, default 512); retain `--no-search` and `--duration`.
- Metadata records timezone-aware start, Python/OpenCV/NumPy versions when available, source image dimensions when known, safe CLI flags, strategy `legacy`, and local Git revision if obtainable without a network call.

- [x] **Step 1: Add parser/lifecycle tests for defaults, all existing flags, record-frames implication, invalid/nonpositive limit, metadata sanitization, normal return, Ctrl+C, and unexpected gameplay exception.**
- [x] **Step 2: Run focused tests and confirm new CLI assertions fail.** RED exposed missing CLI support; a test logger-name mismatch was corrected to the configured logger.
- [x] **Step 3: Initialize diagnostics only when requested and close in `finally`; if initialization fails, warn once and continue with NoOp without swallowing gameplay exceptions.**
- [x] **Step 4: Run CLI tests and verify plain `python bot.py` still creates no diagnostics directory.** CLI tests 5/5 passed.

### Task 5: Read-only recording analyzer

**Files:**
- Create: `utils/analyze_recording.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- `analyze_session(path) -> str` and CLI `python utils/analyze_recording.py SESSION_DIR [--verbose]`.
- Reader uses only `metadata.json`, `events.jsonl`, and `summary.json`; malformed/truncated JSONL rows are counted and skipped while valid rows are retained.

- [x] **Step 1: Add tests for complete and partial sessions covering summary, phase sequence, decisions/reasons, elixir changes, assumed purchases/upgrades, sale attempts, unknowns, repeated frames, malformed lines, and no file mutation.**
- [x] **Step 2: Run the analyzer tests and confirm they fail before the utility exists.** RED was `ModuleNotFoundError: utils.analyze_recording`.
- [x] **Step 3: Implement a standard-library-only formatter with concise Russian output and a verbose copyable event timeline.**
- [x] **Step 4: Run analyzer tests on a generated temporary session and confirm it performs no imports of bot/ADB/OpenCV or network access.** Analyzer tests pass; import guard found none of `bot`, `core.adb_controller`, `cv2`, or `numpy` loaded.

### Task 6: End-to-end regression, documentation, and publish

**Files:**
- Create: `docs/stage-2-diagnostics.md`
- Modify: `README.md` only if a concise launch/analyze example is useful
- Test: `tests/test_diagnostics.py`

- [x] **Step 1: Add a synthetic parity test comparing diagnostics disabled/enabled for selected cards, ADB tap order/coordinates/count, screenshot requests, sleep schedule, RunState, and return values.**
- [x] **Step 2: Run the full suite with `\.venv\Scripts\python.exe -m unittest discover -s tests -v`; confirm all 29 original tests and every added test pass.** Final fresh run: 72/72 passed.
- [x] **Step 3: Document actual event names/fields and confidence labels, limitations, PowerShell launch/analyzer/test commands, AI handoff files, and manual real-interface checks not performed in `docs/stage-2-diagnostics.md`.**
- [x] **Step 4: Run a synthetic write-through/analyzer smoke check, measure synchronous JSONL event overhead on synthetic events, run `git diff --check`, inspect the complete diff and status, and confirm no user artifacts, recordings, secrets, or temporary files are staged.** Fresh reviewer approved; user-owned local files were excluded.
- [x] **Step 5: Create branch `stage-2-diagnostics` from the verified baseline, stage only exact stage-2 source/tests/docs files, commit as `feat: add opt-in structured match diagnostics`, then push to `origin stage-2-diagnostics` without force. Do not merge to `main`.** Published as `2b45dcc7bd74e58a1babe99963fb2a4a763feab1`; verified remote `refs/heads/stage-2-diagnostics` points to the same SHA. No merge or force push.

**Expected new tests:** isolated `tests/test_diagnostics.py` tests only; do not change existing characterization expectations.
