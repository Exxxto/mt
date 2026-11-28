# Stage 3: action verification, freshness and recovery

## Purchase evidence

`core/action_verifier.py` compares phase, visible elixir, the complete recognized shop, the selected slot, and the approximate own-unit count before and after a purchase. It does not use arbitrary image pixel differences.

- `CONFIRMED`: deploy remains visible, the observed elixir drop equals the selected card's cost, and at least one independent state change agrees: a complete shop assortment changed, the selected slot changed, or approximate own-unit count increased.
- `FAILED`: deploy remains visible, elixir did not change, and the complete shop and selected slot are both recognized and unchanged.
- `UNKNOWN`: evidence is incomplete or conflicts. Unit count is corroborating evidence only; a decrease or unchanged count does not disprove a merge. In limited mode unit count is ignored.
- `INTERRUPTED`: deploy ended or the phase became unrecognizable during verification.

An exact balance spend plus an increased unit count can confirm a purchase while zero elixir dims the cards and prevents name recognition. A changed shop or slot alone cannot confirm a purchase without the expected balance change. ADB status remains separate from game confirmation.

## Pending transactions and limited play

Confirmed purchases update `RunState` once by action ID. Unknown transactions are stored separately and do not alter copies, stars or purchase history. The same transaction is never tapped twice. Two later fresh observations get a chance to resolve it from the original balance and shop state; a confirmed resolution is applied once, while a fully visible no-effect result resolves as failed.

If those observations still conflict, the planner marks the roster unsynchronized and enters limited mode. It continues from fresh, recognized phase, balance and shop data, but ignores copy counts when choosing cards. Each new purchase still needs the expected elixir drop plus a changed slot or shop to be confirmed. Unknown purchases remain outside `RunState`; no blind sale or full roster-sync claim is made. Buying remains blocked when phase, balance, affordability or shop cards cannot be read, or while a pending transaction is still within its resolution window. Limited mode ends only with a new match reset; board and bench recognition are not part of Stage 3.

## Fresh frames and pauses

Verify mode timestamps each received frame with `time.monotonic()`. The default maximum age is 3 seconds and can be configured with `--max-frame-age SECONDS`. Before a purchase, orb, match-search or end-screen tap, a stale observation is discarded; no old coordinates are sent. The next loop obtains a new frame and repeats phase, balance, shop and action recognition. Screenshots acquired during retries receive their own new timestamp.

The threshold was compared with offline MuMuPlayer recordings. The new verify session's two post-action frame captures took 1,188 and 1,219 ms; a 3-second limit leaves about 2.5 times that measured capture duration for recognition and tapping. The previous legacy recording's frame-event spacing had a 1,407 ms median and 3,889 ms 95th percentile. It had no gaps over 30 seconds. The new verify recording had two long gaps: 62,016 ms and 34,985 ms. The limit rejects both stale action opportunities while allowing ordinary post-action capture latency. Regression tests exercise 35- and 62-second observations. Long iteration pauses of at least 5 seconds are recorded.

## Diagnostics

`session_started.strategy` reports `verify_actions` when that mode is enabled. Action diagnostics include frame age and limit, stale-action discard, purchase-block reason, pending-transaction resolution, limited-mode entry and long pauses. Final balance snapshots separate the latest observed balance, a calculated balance (null in verify mode when none was calculated), the transaction's starting balance, and the transaction status. Roster snapshots label limited and unsynchronized state explicitly.

## Offline evidence

Original frames 9–13 from `session-608c2f9560f6460db9526b242bbcd475` were inspected visually and run through the existing recognizer offline. Frame 9 reads 5 elixir, has three recognized cards, and shows one own unit. Frame 10 reads 0 elixir, has two own units, and only one of three cards is recognized. Frames 11–13 continue the deploy countdown and no card names are recognized. The average grayscale brightness of the three card tiles drops from 123/97/105 in frame 9 to 69/64/76 in frame 10 and stays lower through frame 13, consistent with the visible dimming at zero elixir. The purchase animation and dimming overlap, so the images do not isolate which one caused recognition loss; the established failure is the incomplete shop signature. The verifier does not use brightness or arbitrary pixel changes as purchase evidence. Expected 5-elixir spend plus the independent unit increase from 1 to 2 confirms this transaction.

The new session had 347 events, 2 purchase attempts, 0 confirmed purchases and 2 UNKNOWN results. Its latest prior legacy session had 335 events and 5 legacy purchase attempts. Offline analysis reads the recordings only and never taps the game.

## Run

Existing tests plus Stage 3 regressions:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Verify mode, with the game already open:

```powershell
.venv\Scripts\python.exe bot.py --no-search --verify-actions --record-frames --max-frame-age 3
```

No real match was run for this change. Strategy ranking, tier lists, economy policy, round handling, match resets, orb ownership and automatic match restart remain unchanged.
