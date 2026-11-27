# Stage 3: action verification and state consistency

## Purchase verification

`core/action_verifier.py` compares visible observations from one frame before and one frame after a purchase. An observation groups phase, recognized elixir, recognized shop cards, and the contents of the selected slot. An incomplete shop is unknown; it is not evidence that the assortment changed or stayed the same. A selected-slot change requires the same recognized slot index with different recognized card contents.

- `CONFIRMED`: deploy continued, the visible elixir delta equals the selected card cost, and the complete shop or selected slot visibly changed.
- `FAILED`: deploy continued, the visible balance is unchanged, and both the complete shop and selected slot are visibly unchanged.
- `UNKNOWN`: observations are missing or conflict, or ADB failed without enough visual evidence to decide whether the tap took effect.
- `INTERRUPTED`: the post-action observation shows deploy ended or its phase is unknown.

ADB process status and game outcome are recorded separately. A timeout or transport exception does not prove that no tap reached the game. In verify mode an unknown/interrupted result ends the current buy chain and marks `RunState.sync_required`. Later rounds cannot use that state for purchases; it is cleared only when a new match initializes a new `RunState`. Confirmed actions update the roster once by `action_id`; unknown actions do not.

## Fresh observations and bounded work

Verify mode checks phase before a tap and after every newly captured frame while waiting for elixir or shop recognition. Shop cards and elixir used in a decision come from the same frame. A shop retry that sees another phase stops without tapping.

Each purchase and its single optional recheck share an eight-second monotonic budget. Tap, screenshot capture, and deliberate delay use the remaining budget; an expired budget prevents another capture. CPU-based image recognition cannot be forcibly interrupted, so unusually slow recognition may finish after the deadline, but it cannot trigger an additional recheck after expiration. Pull-mode screenshot capture, download, and remote cleanup share the caller's timeout. Cleanup is skipped when the budget has expired.

After orb taps, the bot requires a fresh screenshot before it makes another phase or purchase decision. If capture fails, the loop drops the old frame and retries from a new capture. `MergeTacticsBot.collect_orbs` remains the only component that taps orbs.

The legacy mode remains the default and retains its existing purchase accounting. Verify mode does not invoke the legacy sale fallback, so an uncertain purchase cannot trigger a sale. No board or bench recognition is added; synchronization recovery from an unknown result remains out of scope.

## Diagnostics and compatibility

The verifier result includes action type and ID, ADB status, frame IDs, before/after observations, evidence, reason, elapsed time, and recheck count. Existing schema version 1 recordings remain readable. The analyzer keeps verified outcomes distinct from legacy inferred purchases. The existing diagnostics storage cap still applies to both events and frames; when exhausted, recording continues to drop later data.

## Run and manually inspect

Run the automatic tests with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a manual check, start with the game already open:

```powershell
.venv\Scripts\python.exe bot.py --no-search --verify-actions --record-frames
```

Inspect each `buy_attempt` alongside its `action_verification_result`. A confirmed transaction updates `RunState` once; an unknown or interrupted result ends buying and requires a new match before the state is trusted again. This procedure sends game input. No live match or game taps were used for the automated checks in this stage.
