# Stage 3: Action verification and match state

## Architecture and classification

`core/action_verifier.py` models a visible before/after observation and returns a structured action result. It keeps ADB transport status separate from the game result. A purchase is `CONFIRMED` only when the observed elixir decrease equals the selected card cost and the shop assortment or selected slot changes. A stable balance and unchanged shop is `FAILED`; missing or conflicting observations are `UNKNOWN`; a phase outside deploy is `INTERRUPTED`. These outcomes confirm the transaction only, not the troop's location or star level.

The `--verify-actions` flag enables the verifier and structured diagnostics. It captures an after-action frame, permits one bounded recheck without another tap, and ends the purchase chain when the result remains unknown. The new mode gives ADB tap and screenshot operations five-second timeouts. `RunState` is updated only after confirmation, and confirmed action IDs are applied once. Legacy behavior remains the default; legacy purchases continue to be labeled as legacy in recordings.

## State and input fixes

- `MergeTacticsBot.collect_orbs` is the sole tap owner. After any orb taps it captures a fresh frame and the main loop re-evaluates phase before making more decisions. The planner still records orb detections but does not tap them.
- The first deploy following `search_tap` or `rematch_tap` consumes that pending boundary without clearing `RunState`; a later real boundary still creates a new run.
- Unknown phase observations no longer overwrite the last stable phase. Round index advances on a stable `battle` to `deploy` transition, including when unknown observations occur between them. Diagnostics count phase observations and stable transitions separately.
- Verification mode never invokes the legacy reactive sale path. Safe automatic sales remain out of scope until bench recognition exists.

## Diagnostics and limits

Schema version remains 1. New `action_verification_result` events include action and frame IDs, visible observations, evidence, reason, ADB status, recheck count, and total duration. The analyzer reports verified outcomes separately from legacy accounting and continues to accept old schema v1 recordings. Phase observation and transition totals are included in new summaries.

The existing diagnostics byte cap still applies to both event and frame data. When exhausted, the current recorder drops further events; this stage does not add a separate event reserve. The result duration is measured with a monotonic clock. Verification events also record ADB tap duration and aggregate frame capture and recognition duration, including a recheck when used.

The five-second bound applies to each ADB operation, and a single recheck can add another screenshot delay. End-to-end latency has not been measured on MuMuPlayer in this stage. The previous approximate 500 ms frame time suggests a recheck may be noticeable, but is not a measurement of this implementation.

## Verification

The existing 81 tests plus 11 new tests pass with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use fake ADB calls and synthetic observations; no live match or game tap was run. `git diff --check` is also clean.

## Offline review of the supplied match

The following before/after pairs were reviewed from the local PNGs. Each pair shows deploy continuing, the recorded elixir delta matching the cost, and the shop changing. This supports a transaction confirmation under the stage 3 rule; it does not prove a particular board or star result.

| Action | Frames | Visible elixir | Shop evidence | Offline result |
| --- | --- | --- | --- | --- |
| barbarian (2) | 7 → 8 | 5 → 3 | slot assortment changed | Confirmable |
| giant_skeleton (3) | 8 → 9 | 3 → 0 | slot assortment changed | Confirmable |
| mini_pekka (5) | 35 → 36 | 5 → 0 | slot assortment changed | Confirmable |
| bandit (3) | 66 → 67 | 5 → 2 | slot assortment changed | Confirmable |
| knight (2) | 67 → 68 | 2 → 0 | slot assortment changed | Confirmable |

These are offline observations of the existing recording. The new mode was not run against a live game.

## Manual check

Start with the game already open and use:

```powershell
.venv\Scripts\python.exe bot.py --no-search --verify-actions --record-frames
```

Review the generated `recordings/<session>/events.jsonl` for one `buy_attempt` and one `action_verification_result` per purchase. Confirmed purchases should change inferred `RunState` once; unknown outcomes should end that phase's purchase chain. Stop with Ctrl+C. Do not use this procedure on a live match without explicitly authorizing game input.
