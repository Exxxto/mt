# Stage 4: autonomous vision data pipeline

This experiment builds an offline, resumable pipeline for the bot's own five-slot
bench. It keeps diagnostics read-only and does not connect predictions to game
state or actions. The available recordings do not support reliable field or star
labels, so those remain outside the trained model.

## Local data audit

The audit found 1,353 saved PNGs in seven recording sessions. Five sessions have
images; two contain only screenshot errors. There are 16 recorded matches. Of the
PNGs, 1,344 join to explicit `frame_seen` events with a recognized phase: 647
deployment frames and 697 battle frames. The highlighted session has 914 PNGs,
10 matches, and 912 joined game frames; two saved images lack a usable phase/frame
join and are excluded from dataset preparation. Source recordings are not changed.

The repository has 20 bench template classes and no verified labels for stars.
The existing visual holdout contains 53 frames from matches 006, 009, and 010,
with 61 occupied slots. Those three whole matches are reserved for evaluation.
They are never exported to train or validation.

## Labeling and dataset rules

`BenchReader` supplies occupancy, template identity, and model boxes. A class label
is reliable pseudo-labeling only when the occupancy is positive, the template
score is at least 0.80, and the same identity recurs in the same slot within 12
frame IDs without an intervening purchase, merge, or sale event. Stars stay null.
The pipeline excludes unknown or partial slot crops rather than treating them as
background. Adjacent unknown/overlapping slots are excluded too.
Training images are 94×170 crops for one bench slot.

Preparation hashes every joined frame and uses a DCT perceptual hash on the bench
region to select diverse representatives. The selection cache keys file size and
mtime, so unchanged frames do not need their hashes recalculated. Dataset exports
use isolated slot crops, which prevents an unlabelled character in another bench
slot from becoming a false negative. Splits are deterministic and match-scoped.

Each candidate record includes the source frame, session, match, frame and slot
IDs, bounding box, class, star value, confidence, evidence, algorithm version,
and review status. `contact` writes local labeled contact sheets with frame IDs,
coordinates, and scores for visual review.

## Environment and commands

The bot environment remains separate. On this Windows RTX 5060 machine, create the
ML environment and install the pinned packages as follows:

```powershell
py -3.12 -m venv .venv-ml
uv pip install --python .venv-ml\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -r requirements-ml-lock.txt
```

The supported GPU check runs an actual CUDA tensor operation and training refuses
to fall back to CPU. The selected model is YOLO26n, initialized from pretrained
weights and fine-tuned on the supported bench classes. Artifacts, caches, exported
images, contact sheets, logs, and checkpoints live under ignored `ml_artifacts/`.

Run all data stages and the bounded training/evaluation flow with:

```powershell
.\.venv-ml\Scripts\python.exe -m utils.ml_pipeline run-all
```

Individual commands include `audit`, `prepare`, `label`, `validate`, `contact`,
`train`, `evaluate`, `improve`, and `observe`. `observe` runs on saved recordings
and writes predictions to local diagnostics JSONL; it never changes `RunState` or
game decisions. Training checkpoints support resumption.

## Results from this checkout

The pipeline validated 321 reliable template-plus-temporal pseudo-label boxes and
rejected 297 partially labeled frames. It exported 145 train crops and 19
validation crops, with 28 train objects and 2 validation objects. Only archer
and barbarian passed the minimum class support rules. Across those selected
labels, 60 representative crops from archer, barbarian, and valkyrie were
visually reviewed; none were rejected. Many crops are near-identical, so 60 is a
review count, not 60 independent samples. The review summary is in the ignored
`ml_artifacts/visual_review.json` file.

Training used YOLO26n, 40 requested epochs (early stopping is allowed), seed
41042, 640-pixel input, batch size 2, and CUDA AMP. The run took 157.4 seconds
on an NVIDIA GeForce RTX 5060 with PyTorch 2.12.1+cu132 (CUDA 13.2), driver
621.14. The dataset provenance SHA-256 is
`2d45b18b23a73a77bfcae82d68431eecfd2417bb4617351794268c32618972f7`. Built-in
validation reported mAP50 0.995 and mAP50-95 0.8955, but those values use only
two validation objects and should not be treated as a performance estimate.

The independent fixed test covers 53 frames and 61 occupied slots from whole
matches 006, 009, and 010. It contains 60 known-name labels and one occupied
slot without a verified name. On slot occupancy, YOLO26n got 44 true positives,
0 false positives, and 17 misses (precision 1.000, recall 0.721, F1 0.838).
`BenchReader` got 59 true positives, 0 false positives, and 2 misses (precision
1.000, recall 0.967, F1 0.983). On the 60 known-name occupied slots, YOLO26n
identified 31 correctly (51.7%); `BenchReader` identified 45 (75.0%). The model
only contains archer and barbarian classes, so other units are outside its
supported identity set. The archer class had 2 correct labels and 13 false
identity predictions on this holdout.

Steady-state slot-crop inference had a 15.6 ms p95 and used about 70 MB of peak
allocated VRAM in the evaluation process. No independent test bounding boxes
exist, so test mAP is not reported. Star accuracy is also unavailable. Since the
trained detector scored below the existing reader on this held-out match test,
the capped self-training stage added no data and did not promote a second model.
The detector remains an offline experiment and is not connected to bot decisions.

Artifacts and full confusion counts are local under ignored `ml_artifacts/`.
The small committed review fixture is the fixed holdout annotation; recordings,
contact sheets, exported crops, and model weights are not committed.
