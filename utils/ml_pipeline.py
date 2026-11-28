"""Local, resumable preparation and evaluation for experimental bench detection."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = ROOT / "recordings"
ARTIFACTS = ROOT / "ml_artifacts"
HOLDOUT = ROOT / "tests" / "fixtures" / "bench_stage4_holdout.json"
ALGORITHM_VERSION = "bench-pseudo-v2"
NEAR_DUP_PHASH_DISTANCE = 1
FIXED_TEST_MATCHES = {
    "session-ee2e7d0d72b64813a96ff3cb739fe332/match-006",
    "session-ee2e7d0d72b64813a96ff3cb739fe332/match-009",
    "session-ee2e7d0d72b64813a96ff3cb739fe332/match-010",
}


def load_config(path: Path | None = None) -> dict:
    config_path = Path(path) if path is not None else ROOT / "config" / "ml_pipeline.json"
    values = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"model", "epochs", "imgsz", "batch", "device", "patience", "seed",
                "confidence", "label_score_threshold", "temporal_max_gap_frames",
                "near_duplicate_phash_distance", "max_improve_rounds",
                "min_train_objects", "min_validation_objects"}
    if set(values) != required:
        raise ValueError("ML config is missing required fields or has unknown fields")
    if (values["epochs"] < 1 or values["imgsz"] < 32 or values["batch"] < 1
            or values["device"] != "cuda:0"
            or not 0 < values["confidence"] < 1
            or not 0 < values["label_score_threshold"] <= 1
            or values["temporal_max_gap_frames"] < 1):
        raise ValueError("ML config contains invalid training or label thresholds")
    return values


def save_config(path: Path) -> dict:
    values = load_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    return values


def training_action(previous: dict | None, dataset_hash: str, epochs: int,
                    has_best: bool, has_last: bool) -> str:
    prior_fingerprint = (previous or {}).get("training_fingerprint",
                                            (previous or {}).get("dataset_sha256"))
    if previous and prior_fingerprint == dataset_hash \
            and previous.get("epochs_requested") == epochs and has_best:
        return "reuse"
    if previous and has_best:
        return "new_from_best"
    if has_last:
        return "resume"
    return "fresh"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_events(session_dir: Path) -> list[dict]:
    path = Path(session_dir) / "events.jsonl"
    if not path.is_file():
        return []
    events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def discover_frames(session_dir: Path) -> list[dict]:
    """Join saved PNGs to their explicit frame events; omit non-game/unknown phases."""
    session_dir = Path(session_dir).resolve()
    metadata_path = session_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    session_id = metadata.get("session_id") or session_dir.name.split("_session-")[-1]
    result = []
    for event in _read_events(session_dir):
        if event.get("event_type") != "frame_seen":
            continue
        data = event.get("data") or {}
        rel_path = data.get("path")
        if not isinstance(rel_path, str):
            continue
        path = (session_dir / rel_path).resolve()
        if session_dir not in path.parents or path.suffix.lower() != ".png" or not path.is_file():
            continue
        phase = event.get("phase")
        if phase not in {"deploy", "battle"}:
            continue
        match_id, frame_id = event.get("match_id"), event.get("frame_id")
        if not match_id or not isinstance(frame_id, int):
            continue
        result.append({
            "session_id": session_id,
            "session_dir": str(session_dir),
            "match_id": match_id,
            "frame_id": frame_id,
            "phase": phase,
            "reason": data.get("reason"),
            "path": str(path),
        })
    return sorted(result, key=lambda row: (row["match_id"], row["frame_id"]))


def _group_key(row: dict) -> str:
    return f"{row['session_id']}/{row['match_id']}"


def assign_splits(rows: list[dict], fixed_test_groups: set[str] | None = None,
                  seed: int | None = None, validation_percent: int = 15) -> list[dict]:
    """Assign whole matches deterministically, reserving declared test matches first."""
    fixed_test_groups = fixed_test_groups or set()
    seed = load_config()["seed"] if seed is None else seed
    groups = sorted({_group_key(row) for row in rows})
    result = []
    for row in rows:
        group = _group_key(row)
        digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % 100
        split = "test" if group in fixed_test_groups else (
            "validation" if bucket < validation_percent else "train")
        result.append({**row, "split": split})
    # Small archives still get a validation group when at least two non-test matches exist.
    available = [group for group in groups if group not in fixed_test_groups]
    assigned = {row["split"] for row in result}
    if len(available) >= 2 and "validation" not in assigned:
        chosen = min(available, key=lambda group: hashlib.sha256(
            f"validation:{seed}:{group}".encode()).digest())
        result = [{**row, "split": "validation" if _group_key(row) == chosen else row["split"]}
                  for row in result]
    if len(available) >= 2 and not any(row["split"] == "train" for row in result):
        chosen = max(available, key=lambda group: hashlib.sha256(
            f"train:{seed}:{group}".encode()).digest())
        result = [{**row, "split": "train" if _group_key(row) == chosen else row["split"]}
                  for row in result]
    return result


def is_reliable_label(label: dict, width: int, height: int) -> bool:
    box = label.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return False
    x1, y1, x2, y2 = box
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
        return False
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return False
    evidence = set(label.get("evidence") or [])
    return (
        label.get("source") == "bench_reader"
        and label.get("status") == "reliable_pseudo"
        and isinstance(label.get("class_name"), str)
        and label.get("score", 0) >= load_config()["label_score_threshold"]
        and {"occupied_slot", "same_identity_neighbor"}.issubset(evidence)
        and label.get("stars") is None
    )


def crosses_transaction_boundary(events: list[dict], match_id: str,
                                first_frame: int, last_frame: int) -> bool:
    lo, hi = sorted((first_frame, last_frame))
    for event in events:
        frame_id = event.get("frame_id")
        if not isinstance(frame_id, int) or not lo < frame_id <= hi:
            continue
        if event.get("match_id") != match_id:
            continue
        if any(word in event.get("event_type", "")
               for word in ("buy", "purchase", "merge", "sell")):
            return True
    return False


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def _phash(image) -> int:
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(gray.astype("float32"))[:8, :8]
    values = dct.flatten()[1:]
    median = float(__import__("numpy").median(values))
    bits = values > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def slot_crop(image, slot_index: int):
    if not 0 <= slot_index < 5:
        raise ValueError("slot index must be from 0 to 4")
    x0 = 160 + 94 * slot_index
    return image[1190:1360, x0:x0 + 94], (x0, 1190)


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def audit() -> dict:
    sessions = []
    rows = []
    raw_images = 0
    all_matches = set()
    events_total = collections.Counter()
    for session in sorted(path for path in RECORDINGS.iterdir() if path.is_dir()):
        metadata_path = session / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        events = _read_events(session)
        events_total.update(event.get("event_type", "unknown") for event in events)
        raw = list((session / "frames").glob("*.png")) if (session / "frames").is_dir() else []
        raw_images += len(raw)
        matches = sorted({event.get("match_id") for event in events if event.get("match_id")})
        all_matches.update((session.name, match) for match in matches)
        session_rows = discover_frames(session)
        rows.extend(session_rows)
        sessions.append({
            "session_id": metadata.get("session_id", session.name),
            "directory": session.name,
            "raw_pngs": len(raw),
            "joined_game_frames": len(session_rows),
            "matches": matches,
            "events": len(events),
            "errors": sum(event.get("event_type", "").endswith("error") for event in events),
        })
    phase_counts = collections.Counter(row["phase"] for row in rows)
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8")) if HOLDOUT.exists() else {"frames": []}
    stars = collections.Counter(slot.get("stars") for frame in holdout["frames"]
                                for slot in frame.get("slots", []) if slot.get("stars") is not None)
    classes = sorted(path.stem.split("__", 1)[0] for path in (ROOT / "templates/bench").glob("*.png"))
    result = {
        "algorithm_version": ALGORITHM_VERSION,
        "raw_pngs": raw_images,
        "joined_game_frames": len(rows),
        "deploy_frames": phase_counts["deploy"],
        "battle_frames": phase_counts["battle"],
        "matches": len(all_matches),
        "sessions": sessions,
        "bench_template_classes": sorted(set(classes)),
        "bench_template_class_count": len(set(classes)),
        "visually_labeled_holdout_frames": len(holdout.get("frames", [])),
        "visually_labeled_holdout_occupied_slots": sum(
            slot.get("occupancy") == "occupied" for frame in holdout.get("frames", [])
            for slot in frame.get("slots", [])),
        "visually_labeled_star_levels": sorted(stars),
        "star_labels_available": sum(stars.values()),
        "event_types": dict(events_total),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def prepare(seed: int | None = None) -> list[dict]:
    source_rows = []
    for session in sorted(path for path in RECORDINGS.iterdir() if path.is_dir()):
        source_rows.extend(discover_frames(session))
    rows = assign_splits(source_rows, FIXED_TEST_MATCHES, seed=seed)
    cache_path = ARTIFACTS / "prepare_cache.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache = {}
    hashes = {}
    exact_seen = set()
    phash_seen = []
    duplicate_exact = duplicate_near = 0
    for row in sorted(rows, key=lambda item: (item["split"] != "test", item["path"])):
        path = Path(row["path"])
        stat = path.stat()
        key = str(path)
        cached = cache.get(key, {})
        if cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            digest, perceptual = cached["sha256"], int(cached["phash"])
        else:
            image = cv2.imread(key)
            if image is None:
                continue
            # Bench-only hashing avoids treating unrelated arena animation as diversity.
            digest = sha256_file(path)
            perceptual = _phash(image[1190:1360, 130:670])
        hashes[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "sha256": digest, "phash": str(perceptual)}
        row.update({"sha256": digest, "phash": str(perceptual), "selected": True})
        if digest in exact_seen:
            duplicate_exact += 1
            row["selected"] = False
            continue
        exact_seen.add(digest)
        if any(_hamming(perceptual, value) <= NEAR_DUP_PHASH_DISTANCE for value in phash_seen):
            duplicate_near += 1
            row["selected"] = False
        else:
            phash_seen.append(perceptual)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(hashes, separators=(",", ":")), encoding="utf-8")
    _write_jsonl(ARTIFACTS / "manifest.jsonl", rows)
    counts = collections.Counter(row["split"] for row in rows)
    selected = collections.Counter(row["split"] for row in rows if row["selected"])
    report = {"frames": len(rows), "selected": sum(selected.values()),
              "selected_by_split": dict(selected), "by_split": dict(counts),
              "exact_duplicates": duplicate_exact, "near_duplicates": duplicate_near,
              "fixed_test_matches": sorted(FIXED_TEST_MATCHES)}
    (ARTIFACTS / "prepare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return rows


def label() -> list[dict]:
    from core.bench_reader import BenchReader

    rows = _load_jsonl(ARTIFACTS / "manifest.jsonl")
    if not rows:
        rows = prepare()
    reader = BenchReader()
    cached_frames = {(item.get("path"), item.get("sha256")): item
                     for item in _load_jsonl(ARTIFACTS / "frames_labeled.jsonl")
                     if item.get("algorithm_version") == ALGORITHM_VERSION}
    raw = []
    event_cache = {}
    for row in rows:
        if row["phase"] != "deploy":
            continue
        cached = cached_frames.get((row["path"], row.get("sha256")))
        if cached:
            raw.append({**cached, "split": row["split"], "selected": row["selected"]})
            continue
        image = cv2.imread(row["path"])
        if image is None:
            continue
        observation = reader.read(image, row["frame_id"])
        event_cache.setdefault(row["session_id"], _read_events(Path(row["session_dir"])))
        raw.append({**row, "algorithm_version": ALGORITHM_VERSION,
                    "image_width": image.shape[1], "image_height": image.shape[0],
                    "slots": [{
                        "slot": slot.index, "occupancy": slot.occupancy,
                        "class_name": slot.unit_name, "stars": None,
                        "recognition_score": slot.recognition_score,
                        "occupancy_score": slot.occupancy_score,
                        "bbox": ([slot.unit_box[0], slot.unit_box[1],
                                  slot.unit_box[0] + slot.unit_box[2],
                                  slot.unit_box[1] + slot.unit_box[3]]
                                 if slot.unit_box else None),
                    } for slot in observation.slots]})
    # A name is trusted only when the same slot/class recurs nearby and no buy/merge/sell intervenes.
    for item in raw:
        for slot in item["slots"]:
            name, score = slot["class_name"], slot["recognition_score"]
            if not name or score is None or score < 0.80 or slot["occupancy"] != "occupied":
                slot["status"], slot["evidence"] = "preliminary", ["occupied_slot"]
                continue
            neighbors = [other for other in raw if other is not item
                         and other["session_id"] == item["session_id"]
                         and other["match_id"] == item["match_id"]
                         and other["frame_id"] != item["frame_id"]
                         and abs(other["frame_id"] - item["frame_id"])
                         <= load_config()["temporal_max_gap_frames"]
                         and other["slots"][slot["slot"]]["class_name"] == name
                         and other["slots"][slot["slot"]]["recognition_score"] is not None
                         and other["slots"][slot["slot"]]["recognition_score"] >= 0.80]
            nearest = min(neighbors, key=lambda row: abs(row["frame_id"] - item["frame_id"])) if neighbors else None
            boundary = crosses_transaction_boundary(
                event_cache[item["session_id"]], item["match_id"],
                item["frame_id"], nearest["frame_id"]) if nearest else True
            if neighbors and not boundary:
                slot["status"] = "reliable_pseudo"
                slot["evidence"] = ["occupied_slot", "template_correlation", "same_identity_neighbor"]
            else:
                slot["status"] = "preliminary"
                slot["evidence"] = ["occupied_slot", "template_correlation"]
    # Evidence fields retain source and algorithm version for every candidate, including unknowns.
    labels = []
    for item in raw:
        for slot in item["slots"]:
            label_row = {
                "session_id": item["session_id"], "match_id": item["match_id"],
                "frame_id": item["frame_id"], "split": item["split"],
                "image": item["path"], "slot": slot["slot"], "bbox": slot["bbox"],
                "class_name": slot["class_name"], "stars": None,
                "source": "bench_reader", "status": slot["status"],
                "score": slot["recognition_score"], "occupancy": slot["occupancy"],
                "evidence": slot["evidence"], "algorithm_version": ALGORITHM_VERSION,
                "occupancy_score": slot["occupancy_score"],
            }
            labels.append(label_row)
    _write_jsonl(ARTIFACTS / "labels.jsonl", labels)
    _write_jsonl(ARTIFACTS / "frames_labeled.jsonl", raw)
    counts = collections.Counter(row["status"] for row in labels)
    class_counts = collections.Counter(row["class_name"] for row in labels
                                       if row["status"] == "reliable_pseudo")
    summary = {"labeled_frames": len(raw), "slot_candidates": len(labels),
               "status_counts": dict(counts), "reliable_by_class": dict(class_counts),
               "reliable_labels": sum(class_counts.values()), "verified_labels": 0,
               "stars_labeled": 0}
    (ARTIFACTS / "label.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return labels


def validate() -> dict:
    labels = _load_jsonl(ARTIFACTS / "labels.jsonl")
    frames = _load_jsonl(ARTIFACTS / "frames_labeled.jsonl")
    split_groups = collections.defaultdict(set)
    for row in frames:
        split_groups[_group_key(row)].add(row["split"])
    leaks = {key: sorted(value) for key, value in split_groups.items() if len(value) != 1}
    invalid = [row for row in labels if row["status"] == "reliable_pseudo"
               and not is_reliable_label(row, 900, 1600)]
    partial = sum(
        0 < sum(slot["occupancy"] == "occupied" for slot in row["slots"])
        and any(slot["occupancy"] == "unknown"
                or (slot["occupancy"] == "occupied" and slot["status"] != "reliable_pseudo")
                for slot in row["slots"])
        for row in frames)
    result = {"valid": not leaks and not invalid, "match_split_leaks": leaks,
              "invalid_reliable_labels": len(invalid),
              "frames_rejected_for_partial_labels": partial,
              "valid_boxes": sum(is_reliable_label(row, 900, 1600) for row in labels)}
    (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise RuntimeError("Dataset validation failed; refusing export/training")
    return result


def contact_sheets(page_size: int = 20) -> int:
    """Render local crops with candidate, source frame and pixel coordinates for visual QA."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    output = ARTIFACTS / "contact_sheets"
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.jpg"):
        old.unlink()
    grouped = collections.defaultdict(list)
    for row in _load_jsonl(ARTIFACTS / "labels.jsonl"):
        if row["status"] == "reliable_pseudo" and row.get("bbox"):
            grouped[row["class_name"]].append(row)
    tile_w, tile_h, caption_h, columns = 240, 210, 42, 5
    written = 0
    for name, items in sorted(grouped.items()):
        items.sort(key=lambda row: (row["split"], row["session_id"], row["frame_id"], row["slot"]))
        for page, start in enumerate(range(0, len(items), page_size), 1):
            page_items = items[start:start + page_size]
            rows = math.ceil(len(page_items) / columns)
            sheet = np.full((rows * (tile_h + caption_h), columns * tile_w, 3), 245, dtype=np.uint8)
            for index, item in enumerate(page_items):
                image = cv2.imread(item["image"])
                x1, y1, x2, y2 = map(int, item["bbox"])
                crop = image[max(0, y1 - 14):min(image.shape[0], y2 + 14),
                             max(0, x1 - 14):min(image.shape[1], x2 + 14)]
                if crop.size == 0:
                    continue
                scale = min(tile_w / crop.shape[1], tile_h / crop.shape[0])
                resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                            max(1, int(crop.shape[0] * scale))))
                col, row = index % columns, index // columns
                left = col * tile_w + (tile_w - resized.shape[1]) // 2
                top = row * (tile_h + caption_h) + (tile_h - resized.shape[0]) // 2
                sheet[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
                caption = f"{item['split']} {item['match_id']} f{item['frame_id']} s{item['slot']}"
                cv2.putText(sheet, caption, (col * tile_w + 4, row * (tile_h + caption_h) + tile_h + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, (15, 15, 15), 1, cv2.LINE_AA)
                coord = f"xyxy={x1},{y1},{x2},{y2} score={item['score']:.2f}"
                cv2.putText(sheet, coord, (col * tile_w + 4, row * (tile_h + caption_h) + tile_h + 34),
                            cv2.FONT_HERSHEY_SIMPLEX, .34, (15, 15, 15), 1, cv2.LINE_AA)
            path = output / f"{name}_{page:02d}.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            written += 1
    print(f"Wrote {written} contact sheets to {output}")
    return written


def export_dataset(min_examples: int = 12) -> dict:
    validate()
    frames = _load_jsonl(ARTIFACTS / "frames_labeled.jsonl")
    supported = collections.Counter()
    groups = collections.defaultdict(set)
    for frame in frames:
        if frame["split"] != "train" or not frame["selected"]:
            continue
        for slot in frame["slots"]:
            if slot["status"] == "reliable_pseudo":
                supported[slot["class_name"]] += 1
                groups[slot["class_name"]].add(_group_key(frame))
    classes = sorted(name for name, count in supported.items()
                     if count >= min_examples and len(groups[name]) >= 2)
    if not classes:
        raise RuntimeError("No class has enough reliable training examples from multiple matches")
    dataset = ARTIFACTS / "dataset"
    for split in ("train", "validation"):
        for folder in (dataset / "images" / split, dataset / "labels" / split):
            folder.mkdir(parents=True, exist_ok=True)
            for old_file in folder.iterdir():
                if old_file.is_file() and old_file.suffix.lower() in {".jpg", ".txt"}:
                    old_file.unlink()
    class_to_id = {name: index for index, name in enumerate(classes)}
    exported = collections.Counter()
    skipped = collections.Counter()
    rows_out = []
    crop_hashes = collections.defaultdict(list)
    for frame in frames:
        split = frame["split"]
        if split not in {"train", "validation"} or not frame["selected"]:
            continue
        image = cv2.imread(frame["path"])
        if image is None:
            skipped["missing_image"] += 1
            continue
        for slot in frame["slots"]:
            if slot["occupancy"] == "unknown":
                skipped["unknown_occupancy"] += 1
                continue
            is_positive = slot["occupancy"] == "occupied"
            if is_positive and (slot["status"] != "reliable_pseudo"
                                or slot["class_name"] not in class_to_id or not slot["bbox"]):
                skipped["unlabeled_occupied"] += 1
                continue
            x0 = 160 + 94 * slot["slot"]
            x1 = x0 + 94
            # Never turn a neighboring unknown/partially visible unit into background.
            contaminated = False
            for other in frame["slots"]:
                if other["slot"] == slot["slot"]:
                    continue
                if (not is_positive and other["occupancy"] == "unknown"
                        and abs(other["slot"] - slot["slot"]) == 1):
                    contaminated = True
                    break
                if other["occupancy"] == "occupied" and other["bbox"]:
                    bx1, _, bx2, _ = other["bbox"]
                    if bx1 < x1 and bx2 > x0:
                        contaminated = True
                        break
                elif other["occupancy"] == "occupied" and abs(other["slot"] - slot["slot"]) == 1:
                    contaminated = True
                    break
            if contaminated:
                skipped["neighbor_overlap_or_unknown"] += 1
                continue
            crop, _ = slot_crop(image, slot["slot"])
            identity = slot["class_name"] if is_positive else "empty"
            dedupe_key = (split, _group_key(frame), slot["slot"], identity)
            perceptual = _phash(crop)
            if any(_hamming(perceptual, prior) == 0 for prior in crop_hashes[dedupe_key]):
                skipped["near_duplicate_slot_crop"] += 1
                continue
            crop_hashes[dedupe_key].append(perceptual)
            stem = f"{frame['session_id']}_{frame['match_id']}_{frame['frame_id']:06d}_s{slot['slot']}"
            image_path = dataset / "images" / split / f"{stem}.jpg"
            label_path = dataset / "labels" / split / f"{stem}.txt"
            line = ""
            if is_positive:
                bx1, by1, bx2, by2 = slot["bbox"]
                if bx1 < x0 or bx2 > x1 or by1 < 1190 or by2 > 1360:
                    skipped["bbox_outside_slot_crop"] += 1
                    continue
                cx, cy = ((bx1 + bx2) / 2 - x0) / 94, ((by1 + by2) / 2 - 1190) / 170
                width, height = (bx2 - bx1) / 94, (by2 - by1) / 170
                line = f"{class_to_id[slot['class_name']]} {cx:.7f} {cy:.7f} {width:.7f} {height:.7f}"
            cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            label_path.write_text(line, encoding="utf-8")
            exported[split] += 1
            rows_out.append({"image": str(image_path), "labels": str(label_path),
                             "source": frame["path"], "session_id": frame["session_id"],
                             "match_id": frame["match_id"], "frame_id": frame["frame_id"],
                             "slot": slot["slot"], "split": split,
                             "objects": 1 if line else 0,
                             "class_name": slot["class_name"] if line else None})
    yaml = f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/validation\nnames:\n"
    yaml += "".join(f"  {index}: {name}\n" for index, name in enumerate(classes))
    (dataset / "data.yaml").write_text(yaml, encoding="utf-8")
    _write_jsonl(dataset / "provenance.jsonl", rows_out)
    catalog = []
    for name in sorted(set(supported) | set(classes)):
        catalog.append({"id": name, "name": name.replace("_", " ").title(),
                        "bench_reference_count": sum(
                            path.stem.split("__", 1)[0] == name
                            for path in (ROOT / "templates/bench").glob("*.png")),
                        "reliable_train_examples": supported[name],
                        "train_match_count": len(groups[name]),
                        "supported_by_model": name in class_to_id,
                        "stars_supported": False, "field_supported": False})
    (ARTIFACTS / "class_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    summary = {"classes": classes, "class_count": len(classes), "images": dict(exported),
               "train_objects": sum(row["objects"] for row in rows_out if row["split"] == "train"),
               "validation_objects": sum(row["objects"] for row in rows_out if row["split"] == "validation"),
               "skipped_frames": dict(skipped), "dataset": str(dataset)}
    (ARTIFACTS / "export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def _torch_gpu_smoke() -> tuple:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the pinned packages in .venv-ml first") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to train on CPU")
    torch.cuda.reset_peak_memory_stats()
    values = torch.arange(1_000_000, dtype=torch.float32, device="cuda")
    result = (values * values).sum()
    torch.cuda.synchronize()
    if not torch.isfinite(result):
        raise RuntimeError("CUDA smoke computation returned a non-finite value")
    return torch, {"device": torch.cuda.get_device_name(0),
                   "capability": torch.cuda.get_device_capability(0),
                   "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "smoke_sum": float(result.item()),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def train(epochs: int | None = None, resume: bool = True) -> dict:
    config = load_config()
    epochs = config["epochs"] if epochs is None else epochs
    export = json.loads((ARTIFACTS / "export.json").read_text(encoding="utf-8"))
    if (export["train_objects"] < config["min_train_objects"]
            or export["validation_objects"] < config["min_validation_objects"]):
        raise RuntimeError("Too few isolated train/validation slot crops for a detector experiment")
    torch, gpu = _torch_gpu_smoke()
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the pinned ML requirements in .venv-ml first") from exc
    provenance = ARTIFACTS / "dataset" / "provenance.jsonl"
    dataset_hash = sha256_file(provenance)
    summary_path = ARTIFACTS / "training.json"
    pending_path = ARTIFACTS / "training_pending.json"
    previous = None
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
    pending = json.loads(pending_path.read_text(encoding="utf-8")) if pending_path.exists() else None
    training_fingerprint = hashlib.sha256(
        (dataset_hash + json.dumps({**config, "epochs": epochs}, sort_keys=True)).encode()
    ).hexdigest()
    if pending and pending.get("training_fingerprint") == training_fingerprint and resume:
        version = pending["iteration"]
        run_name = pending["run_name"]
        run_dir = ARTIFACTS / "runs" / run_name
        last = run_dir / "weights" / "last.pt"
        can_resume = last.exists()
        starting_weights = previous.get("best_weights") if previous else None
    else:
        version = int(previous.get("iteration", 1)) + 1 if previous else 1
        run_name = "bench-yolo26n" if version == 1 else f"bench-yolo26n-v{version}"
        run_dir = ARTIFACTS / "runs" / run_name
        last = run_dir / "weights" / "last.pt"
        action = training_action(previous, training_fingerprint, epochs,
                                bool(previous and Path(previous.get("best_weights", "")).exists()),
                                last.exists())
        if action == "reuse":
            print("Training already completed for this dataset/configuration; reusing its checkpoint.")
            return previous
        can_resume = action == "resume" and resume and last.exists()
        starting_weights = previous.get("best_weights") if previous else None
        pending = {"iteration": version, "run_name": run_name,
                   "dataset_sha256": dataset_hash, "training_fingerprint": training_fingerprint,
                   "epochs_requested": epochs}
        pending_path.write_text(json.dumps(pending, indent=2), encoding="utf-8")
    model = YOLO(str(last) if can_resume else (starting_weights or "yolo26n.pt"))
    started = time.perf_counter()
    results = model.train(
        data=str(ARTIFACTS / "dataset" / "data.yaml"), epochs=epochs,
        imgsz=config["imgsz"], batch=config["batch"],
        device=config["device"].split(":", 1)[1], workers=0,
        patience=config["patience"], seed=config["seed"],
        amp=True, cache=False, fliplr=0.0, scale=0.10, translate=0.05,
        hsv_h=0.01, hsv_s=0.15, hsv_v=0.15, mosaic=0.0,
        project=str(ARTIFACTS / "runs"),
        name=run_name, exist_ok=True, resume=can_resume,
        plots=True, save=True, verbose=True,
    )
    summary = {"iteration": version, "model": "YOLO26n", "ultralytics": ultralytics.__version__,
               "epochs_requested": epochs, "elapsed_seconds": time.perf_counter() - started,
               "dataset_sha256": dataset_hash, "training_fingerprint": training_fingerprint,
               "gpu": gpu, "best_weights": str(run_dir / "weights" / "best.pt"),
               "last_weights": str(run_dir / "weights" / "last.pt"),
               "metrics": getattr(results, "results_dict", {})}
    history_path = ARTIFACTS / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    history.append(summary)
    history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pending_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, default=str))
    return summary


def evaluate(model_path: Path | None = None) -> dict:
    import torch
    from ultralytics import YOLO
    from core.bench_reader import BenchReader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing the requested GPU evaluation")
    if model_path is None and (ARTIFACTS / "training.json").exists():
        model_path = Path(json.loads((ARTIFACTS / "training.json").read_text(encoding="utf-8"))["best_weights"])
    model_path = Path(model_path or ARTIFACTS / "runs/bench-yolo26n/weights/best.pt")
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    fixture = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    session_suffix = fixture["session_id"].removeprefix("session-")
    session = next(path for path in RECORDINGS.iterdir()
                   if path.name.endswith(f"session-{session_suffix}"))
    model = YOLO(str(model_path))
    bench = BenchReader()
    counts = {"tp": 0, "fp": 0, "fn": 0, "name_correct": 0,
              "name_wrong": 0, "name_missed": 0, "named_truth": 0}
    baseline = {"tp": 0, "fp": 0, "fn": 0, "name_correct": 0,
                "name_wrong": 0, "name_missed": 0, "named_truth": 0}
    class_names = [model.names[index] for index in range(len(model.names))]
    confusion = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)
    timings = []
    warmup_frame = fixture["frames"][0]
    warmup_image = cv2.imread(str(session / "frames" / f"{warmup_frame['frame_id']:06d}.png"))
    warmup_crop, _ = slot_crop(warmup_image, 0)
    config = load_config()
    model.predict(warmup_crop, device=0, conf=config["confidence"],
                  imgsz=config["imgsz"], verbose=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for frame in fixture["frames"]:
        image_path = session / "frames" / f"{frame['frame_id']:06d}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        observation = bench.read(image, frame["frame_id"])
        for index, truth in enumerate(frame["slots"]):
            crop, _ = slot_crop(image, index)
            start = time.perf_counter()
            config = load_config()
            prediction = model.predict(crop, device=0, conf=config["confidence"],
                                       imgsz=config["imgsz"], verbose=False)[0]
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
            positive = truth["occupancy"] == "occupied"
            best = max(prediction.boxes, key=lambda box: float(box.conf[0]), default=None)
            pred = model.names[int(best.cls[0])] if best is not None else None
            counts["tp"] += positive and pred is not None
            counts["fp"] += not positive and pred is not None
            counts["fn"] += positive and pred is None
            if positive and truth.get("unit_name"):
                counts["named_truth"] += 1
                counts["name_correct"] += pred == truth["unit_name"]
                counts["name_wrong"] += pred is not None and pred != truth["unit_name"]
                counts["name_missed"] += pred is None
            true_name = truth.get("unit_name") if positive else None
            true_row = (true_name if true_name in class_names else
                        "out_of_scope" if true_name else
                        "unlabeled_truth" if positive else "empty")
            predicted_column = pred if pred in class_names else "missed"
            confusion[(true_row, predicted_column)] += 1
            for class_name in class_names:
                if positive and true_name == class_name:
                    if pred == class_name:
                        per_class[class_name]["tp"] += 1
                    else:
                        per_class[class_name]["fn"] += 1
                elif pred == class_name:
                    per_class[class_name]["fp"] += 1
            old = observation.slots[index]
            old_positive = old.occupancy == "occupied"
            baseline["tp"] += positive and old_positive
            baseline["fp"] += not positive and old_positive
            baseline["fn"] += positive and not old_positive
            if positive and truth.get("unit_name"):
                baseline["named_truth"] += 1
                baseline["name_correct"] += old.unit_name == truth["unit_name"]
                baseline["name_wrong"] += old.unit_name is not None and old.unit_name != truth["unit_name"]
                baseline["name_missed"] += old.unit_name is None
    def metrics(values):
        precision = values["tp"] / max(1, values["tp"] + values["fp"])
        recall = values["tp"] / max(1, values["tp"] + values["fn"])
        return {**values, "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(1e-12, precision + recall),
                "name_accuracy_on_all_known_occupied":
                    values["name_correct"] / max(1, values["named_truth"])}
    report = {
        "holdout": fixture["session_id"], "holdout_matches": sorted({f["match_id"] for f in fixture["frames"]}),
        "frames": len(fixture["frames"]), "model": str(model_path),
        "yolo26n_slot_metrics": metrics(counts), "bench_reader_slot_metrics": metrics(baseline),
        "class_names": class_names,
        "confusion_matrix": {"rows_and_columns": class_names + ["missed", "out_of_scope", "unlabeled_truth", "empty"],
                             "counts": {f"{row}->{column}": count
                                        for (row, column), count in sorted(confusion.items())}},
        "per_class": {name: {
            **dict(values),
            "precision": values["tp"] / max(1, values["tp"] + values["fp"]),
            "recall": values["tp"] / max(1, values["tp"] + values["fn"]),
        } for name, values in per_class.items()},
        "mean_inference_ms": statistics.mean(timings),
        "p95_inference_ms": sorted(timings)[int((len(timings) - 1) * .95)],
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "mAP": None,
        "mAP_note": "No independent per-object test boxes exist; mAP is intentionally not reported.",
        "stars_accuracy": None, "stars_note": "No visually verified star labels are available.",
    }
    (ARTIFACTS / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def improve(max_rounds: int = 1) -> dict:
    if max_rounds < 0 or max_rounds > 1:
        raise ValueError("Self-training is capped at one conservative iteration")
    export_path = ARTIFACTS / "export.json"
    export = json.loads(export_path.read_text(encoding="utf-8")) if export_path.exists() else {}
    evaluation_path = ARTIFACTS / "evaluation.json"
    evaluation = (json.loads(evaluation_path.read_text(encoding="utf-8"))
                  if evaluation_path.exists() else {})
    yolo_metrics = evaluation.get("yolo26n_slot_metrics", {})
    baseline_metrics = evaluation.get("bench_reader_slot_metrics", {})
    did_not_beat_baseline = (
        evaluation and (
            yolo_metrics.get("f1", 0) < baseline_metrics.get("f1", 0)
            or yolo_metrics.get("name_accuracy_on_all_known_occupied", 0)
            < baseline_metrics.get("name_accuracy_on_all_known_occupied", 0)
        )
    )
    reason = ("The detector scored below BenchReader on the held-out match benchmark; "
              "no self-training round was promoted." if did_not_beat_baseline else
              "No additional training round: the labeled export is too small to support a safe self-training promotion.")
    result = {"iterations": 0, "added_examples": 0,
              "reason": reason,
              "heldout_comparison": {
                  "yolo_f1": yolo_metrics.get("f1"),
                  "bench_reader_f1": baseline_metrics.get("f1"),
                  "yolo_known_identity_accuracy": yolo_metrics.get("name_accuracy_on_all_known_occupied"),
                  "bench_reader_known_identity_accuracy": baseline_metrics.get("name_accuracy_on_all_known_occupied"),
              },
              "train_objects": export.get("train_objects", 0),
              "validation_objects": export.get("validation_objects", 0),
              "max_rounds": max_rounds}
    (ARTIFACTS / "improvement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def observe(model_path: Path | None = None) -> int:
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU inference")
    model_path = Path(model_path or ARTIFACTS / "runs/bench-yolo26n/weights/best.pt")
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = YOLO(str(model_path))
    records = []
    for row in _load_jsonl(ARTIFACTS / "manifest.jsonl"):
        image = cv2.imread(row["path"])
        if image is None:
            continue
        predictions = []
        for index in range(5):
            crop, (x0, y0) = slot_crop(image, index)
            config = load_config()
            result = model.predict(crop, device=0, conf=config["confidence"],
                                   imgsz=config["imgsz"], verbose=False)[0]
            for box in result.boxes:
                coords = [float(value) for value in box.xyxy[0].tolist()]
                coords = [coords[0] + x0, coords[1] + y0, coords[2] + x0, coords[3] + y0]
                predictions.append({"slot": index, "bbox": [round(value, 2) for value in coords],
                                    "class_name": model.names[int(box.cls[0])],
                                    "confidence": float(box.conf[0])})
        records.append({**{key: row[key] for key in ("session_id", "match_id", "frame_id", "phase")},
                        "predictions": predictions})
    _write_jsonl(ARTIFACTS / "observations.jsonl", records)
    print(f"Wrote {len(records)} offline observations to {ARTIFACTS / 'observations.jsonl'}")
    return len(records)


def run_all(skip_train: bool = False, epochs: int = 40) -> None:
    audit()
    prepare()
    label()
    contact_sheets()
    validate()
    export_dataset()
    if skip_train:
        return
    train(epochs=epochs, resume=True)
    evaluate()
    improve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    commands.add_parser("prepare")
    commands.add_parser("label")
    commands.add_parser("validate")
    commands.add_parser("contact")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=40)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--weights", type=Path)
    improve_parser = commands.add_parser("improve")
    improve_parser.add_argument("--max-rounds", type=int, default=1)
    observe_parser = commands.add_parser("observe")
    observe_parser.add_argument("--weights", type=Path)
    all_parser = commands.add_parser("run-all")
    all_parser.add_argument("--skip-train", action="store_true")
    all_parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args(argv)
    if args.command == "audit": audit()
    elif args.command == "prepare": prepare()
    elif args.command == "label": label()
    elif args.command == "validate": validate()
    elif args.command == "contact": contact_sheets()
    elif args.command == "train": train(args.epochs)
    elif args.command == "evaluate": evaluate(args.weights)
    elif args.command == "improve": improve(args.max_rounds)
    elif args.command == "observe": observe(args.weights)
    elif args.command == "run-all": run_all(args.skip_train, args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
