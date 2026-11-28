# Stage 3: shop recognition and orb filtering

This update keeps the existing SIFT/RANSAC art matcher and its thresholds. It
tries the original grayscale match first, then retries a failed match with
CLAHE applied to both the cached art and the shop tile. The active images are
the season art files in `debug_images/cards/raw/`, mapped by `ART_MAP` in
`core/card_identifier.py`; `templates/units/` is not the active shop library.
Cost digits and affordability remain separate from card identity. Shop
diagnostics include matching method, correlation score, and a brightness-based
visual state (`dim`, `light`, or `unknown`).

## Offline comparison

The local session `session-a772dd2c8a8340d68ed228c924413c13` has 173 received
frames, 162 saved PNGs and 9 purchase attempts. Across 15 saved frames around
transactions, there were 45 card slots. The raw matcher named 34; the CLAHE
fallback named all 45. Eleven additional slot-frame matches were visually
checked against their PNGs and were correct (one of these is the same card
repeated in a later capture). On 24 slots with a recorded shop label, both
methods matched 24/24: 20 bright and 4 dim. This labelled subset does not show
an accuracy gain, and is too small to establish a general false-positive rate.
The 11 added matches were visually correct; 10 other raw-SIFT matches in the
45-slot sample had no independent name label. No false match was observed in
the 35 reviewed slots, but those remaining 10 prevent a full precision claim.
No threshold was lowered. Processing these 45 slots took 3.32 s with raw SIFT
and 4.75 s with the fallback enabled (about 44% more for the sample, or 32 ms
more per slot). CLAHE is only run after the raw match fails.

Four purchase attempts were `UNKNOWN` for different evidence gaps:

- `knight`, frames 7–11: inferred cost 2, recognized balance 3→3. The frame
  shows a changed shop and a second own unit, but post-action shop/unit
  recognition was incomplete. The balance digit visually reads 3, so the
  unchanged OCR value is not enough to classify the tap as failed.
- `bomber`, frames 11–14: recognized cost 2 and balance 3→1 agree, but the
  selected shop slot remained unchanged and the post-action unit count was
  unreadable.
- `barbarian`, frames 34–36: recognized cost 2 and balance 3→1 agree, but the
  shop and unit observations after the tap were incomplete.
- `dark_prince`, frames 105–107: recognized cost 5 and balance 5→0 agree, but
  all shop cards were dim/unrecognized and the post-action unit count was
  unreadable.

## Orb filtering

The old HSV component filter produced 24 tap attempts across the 17 frames with
`orb_tap_attempt` events. The local annotated set is in the ignored directory
`debug_images/stage3_orbs/` and includes boxes, click coordinates, and HSV
masks. Visual review classified all 24 attempted targets as false detections:
combat highlights, damage/coin bursts, unit art, or the tutorial banner icon.
None was visually confirmed as a collectible orb. Replaying the persistence
check against the next saved frame rejected all 24 targets; one target stayed
near the same center for one frame, but its component grew from about 50x51 to
121x129 pixels and failed the shape consistency check. No true orb was
confirmed or missed in these attempted target locations; the recording does
not provide a labelled positive orb example, so recall remains unmeasured.

Before tapping, the bot now captures a fresh frame and requires a candidate to
persist near the same center with a similar bounding-box size. Unconfirmed
candidates are recorded as rejected and are not tapped. Tap attempts are
logged as attempts; the result is explicitly `unknown` because the current
game-state reader has no reliable orb collection signal. The extra screenshot
is only taken when the HSV pass finds a candidate.

## Verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

No live match was run. The existing stale-frame guard remains active for taps.
