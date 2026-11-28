import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from utils.ml_pipeline import (
    assign_splits,
    crosses_transaction_boundary,
    discover_frames,
    is_reliable_label,
    sha256_file,
    slot_crop,
    load_config,
    save_config,
    training_action,
)


class MLPipelineTests(unittest.TestCase):
    def test_discovery_joins_recorded_frames_to_match_and_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session-demo"
            (session / "frames").mkdir(parents=True)
            (session / "metadata.json").write_text(json.dumps({"session_id": "demo"}), encoding="utf-8")
            (session / "frames" / "000007.png").write_bytes(b"png")
            events = [
                {"event_type": "frame_seen", "frame_id": 7, "match_id": "match-002",
                 "phase": "deploy", "data": {"path": "frames/000007.png", "reason": "phase_deploy"}},
                {"event_type": "frame_seen", "frame_id": 8, "match_id": "match-002",
                 "phase": "battle", "data": {"path": "frames/000008.png", "reason": "phase_battle"}},
            ]
            (session / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events), encoding="utf-8")

            frames = discover_frames(session)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["match_id"], "match-002")
        self.assertEqual(frames[0]["frame_id"], 7)
        self.assertEqual(frames[0]["phase"], "deploy")

    def test_match_split_keeps_groups_together_and_honors_fixed_test(self):
        rows = [
            {"session_id": "s1", "match_id": "match-001", "frame_id": n}
            for n in range(20)
        ] + [
            {"session_id": "s1", "match_id": f"match-{i:03}", "frame_id": 1}
            for i in range(2, 8)
        ]

        first = assign_splits(rows, {"s1/match-001"}, seed=42)
        second = assign_splits(rows, {"s1/match-001"}, seed=42)

        self.assertEqual(first, second)
        self.assertEqual({r["split"] for r in first if r["match_id"] == "match-001"}, {"test"})
        by_match = {}
        for row in first:
            by_match.setdefault((row["session_id"], row["match_id"]), set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in by_match.values()))
        self.assertIn("validation", {r["split"] for r in first})
        self.assertIn("train", {r["split"] for r in first})

    def test_only_temporally_supported_valid_boxes_are_reliable(self):
        label = {
            "bbox": [10, 20, 30, 60], "class_name": "knight", "stars": None,
            "source": "bench_reader", "status": "reliable_pseudo",
            "score": 0.91, "evidence": ["occupied_slot", "same_identity_neighbor"],
        }

        self.assertTrue(is_reliable_label(label, 900, 1600))
        self.assertFalse(is_reliable_label({**label, "bbox": [10, 20, 10, 60]}, 900, 1600))
        self.assertFalse(is_reliable_label({**label, "evidence": ["same_identity_neighbor"]}, 900, 1600))
        self.assertFalse(is_reliable_label({**label, "score": 0.74}, 900, 1600))

    def test_sha256_file_is_stable_and_distinguishes_content(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first.png"
            second = Path(temp) / "second.png"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            digest = sha256_file(first)
            self.assertEqual(digest, sha256_file(second))
            second.write_bytes(b"different")
            self.assertNotEqual(digest, sha256_file(second))

    def test_boundary_scan_ignores_events_without_frame_ids(self):
        events = [
            {"event_type": "match_boundary_assumed", "frame_id": None, "match_id": "match-002"},
            {"event_type": "buy_outcome", "frame_id": 12, "match_id": "match-002"},
        ]

        self.assertFalse(crosses_transaction_boundary(events[:1], "match-002", 10, 14))
        self.assertTrue(crosses_transaction_boundary(events, "match-002", 10, 14))

    def test_slot_crop_keeps_fixed_bench_geometry(self):
        import numpy as np

        crop, offset = slot_crop(np.zeros((1600, 900, 3), dtype=np.uint8), 2)

        self.assertEqual(crop.shape, (170, 94, 3))
        self.assertEqual(offset, (348, 1190))

    def test_pipeline_config_round_trips_as_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pipeline.json"
            written = save_config(path)
            loaded = load_config(path)

        self.assertEqual(loaded, written)
        self.assertEqual(loaded["epochs"], 40)
        self.assertEqual(loaded["device"], "cuda:0")

    def test_training_resume_uses_cache_checkpoint_and_fresh_data_paths(self):
        same = {"dataset_sha256": "abc", "epochs_requested": 40}
        changed = {"dataset_sha256": "old", "epochs_requested": 40}

        self.assertEqual(training_action(same, "abc", 40, has_best=True, has_last=True), "reuse")
        self.assertEqual(training_action(None, "abc", 40, has_best=False, has_last=True), "resume")
        self.assertEqual(training_action(changed, "abc", 40, has_best=True, has_last=True), "new_from_best")
        self.assertEqual(training_action(None, "abc", 40, has_best=False, has_last=False), "fresh")

    def test_pipeline_import_does_not_load_or_mutate_game_bot(self):
        script = "import sys; import utils.ml_pipeline; assert 'bot' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
