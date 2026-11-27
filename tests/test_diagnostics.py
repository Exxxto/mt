import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.diagnostics import DiagnosticsSession, NoOpDiagnostics
from core.board_state import BoardReader, ShopCard
from core.strategy import DeployPlanner, RunState
from config.board_layout import SHOP_CENTERS
from core.game_logic import GameLogic
from config.board_layout import BENCH_CELLS, END_BUTTONS, SEARCH_BUTTON
import bot
from bot import MergeTacticsBot


class DiagnosticsSessionTests(unittest.TestCase):
    def test_sessions_are_unique_and_events_have_ordered_jsonl_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            first = DiagnosticsSession.create(temp, {"strategy": "legacy", "unknown": None})
            second = DiagnosticsSession.create(temp, {"strategy": "legacy"})
            first.event("phase", data={"phase": "deploy", "card": "Воин"})
            first.event("error", data={"message": "test"})
            first.event("decision", data={"reason": "duplicate", "decision_duration_ms": 12})
            first.close("test")
            second.close("test")

            self.assertNotEqual(first.session_id, second.session_id)
            self.assertNotEqual(first.session_dir, second.session_dir)
            metadata = json.loads((first.session_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 1)
            self.assertIsNone(metadata["unknown"])
            self.assertEqual(metadata["strategy"], "legacy")
            events = [json.loads(line) for line in (first.session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event_id"] for event in events], [1, 2, 3])
            self.assertEqual([event["event_type"] for event in events], ["phase", "error", "decision"])
            self.assertTrue(all(event["session_id"] == first.session_id for event in events))
            self.assertEqual(events[0]["data"]["card"], "Воин")
            summary = json.loads((first.session_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counters"]["decisions"], 1)
            self.assertEqual(summary["average_decision_duration_ms"], 12)
            self.assertEqual(summary["decision_reasons"], {"duplicate": 1})

    def test_noop_diagnostics_create_no_files(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = NoOpDiagnostics()
            self.assertIsNone(diag.event("phase", data={"phase": "deploy"}))
            self.assertIsNone(diag.record_frame(np.zeros((2, 2, 3), dtype=np.uint8), reason="test"))
            self.assertIsNone(diag.frame_id(None))
            self.assertIsNone(diag.new_action_id())
            diag.close("test")
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_first_observed_frame_size_is_saved_to_metadata_without_frame_recording(self):
        with tempfile.TemporaryDirectory() as temp:
            session = DiagnosticsSession.create(temp, {"strategy": "legacy"})
            frame_id = session.frame_id(np.zeros((24, 48, 3), dtype=np.uint8))
            session.close("normal")

            metadata = json.loads(session.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(frame_id, 1)
            self.assertEqual(metadata["frame_size"], {"width": 48, "height": 24})
            self.assertFalse((session.session_dir / "frames").exists())

    def test_original_frame_is_written_at_full_size_and_summary_counts_it(self):
        with tempfile.TemporaryDirectory() as temp:
            image = np.zeros((9, 13, 3), dtype=np.uint8)
            image[2, 4] = (11, 27, 43)
            diag = DiagnosticsSession.create(temp, {}, record_frames=True)
            frame_id = diag.record_frame(image, reason="decision")
            image[2, 4] = (0, 0, 0)
            diag.event("decision", frame_id=frame_id, action_id="action-001", data={"selected": "knight"})
            diag.close("test")

            frame_path = diag.session_dir / "frames" / f"{frame_id:06d}.png"
            saved = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            self.assertEqual(saved.shape, (9, 13, 3))
            self.assertEqual(saved[2, 4].tolist(), [11, 27, 43])
            events = [json.loads(line) for line in (diag.session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            frame_seen = next(event for event in events if event["event_type"] == "frame_seen")
            decision = next(event for event in events if event["event_type"] == "decision")
            self.assertEqual(frame_seen["data"]["path"], f"frames/{frame_id:06d}.png")
            self.assertEqual(decision["frame_id"], frame_id)
            self.assertEqual(decision["action_id"], "action-001")
            summary = json.loads((diag.session_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counters"]["frames_saved"], 1)
            self.assertGreater(summary["total_bytes"], 0)

    def test_frame_write_errors_are_counted_without_escaping_close(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = DiagnosticsSession.create(temp, {}, record_frames=True)
            real_write_bytes = Path.write_bytes

            def fail_png_write(path, data):
                if path.name.endswith(".png.tmp"):
                    raise OSError("disk unavailable")
                return real_write_bytes(path, data)

            with patch.object(Path, "write_bytes", fail_png_write):
                with self.assertLogs("core.diagnostics", level="WARNING") as warnings:
                    diag.record_frame(np.zeros((2, 2, 3), dtype=np.uint8), reason="decision")
                    diag.close("test")
            summary = json.loads((diag.session_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(warnings.output), 1)
            self.assertEqual(summary["counters"]["diagnostic_errors"], 1)
            self.assertEqual(summary["counters"]["frames_saved"], 0)

    def test_summary_write_error_does_not_escape_close(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = DiagnosticsSession.create(temp, {})
            real_write_bytes = Path.write_bytes

            def fail_summary_write(path, data):
                if path.name == "summary.json":
                    raise OSError("disk unavailable")
                return real_write_bytes(path, data)

            with patch.object(Path, "write_bytes", fail_summary_write):
                with self.assertLogs("core.diagnostics", level="WARNING") as warnings:
                    diag.close("test")
            self.assertEqual(len(warnings.output), 1)

    def test_close_has_a_bounded_wait_for_a_stalled_frame_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            entered = threading.Event()
            release = threading.Event()
            real_imencode = cv2.imencode

            def stalled_encode(extension, frame):
                entered.set()
                release.wait(2)
                return real_imencode(extension, frame)

            diag = DiagnosticsSession.create(temp, {}, record_frames=True)
            with patch("core.diagnostics.cv2.imencode", side_effect=stalled_encode), \
                    patch("core.diagnostics.CLOSE_TIMEOUT_SECONDS", 0.01):
                diag.record_frame(np.zeros((2, 2, 3), dtype=np.uint8), reason="test")
                self.assertTrue(entered.wait(1))
                started = time.perf_counter()
                with self.assertLogs("core.diagnostics", level="WARNING") as warnings:
                    diag.close("test")
                self.assertLess(time.perf_counter() - started, 0.5)
                self.assertEqual(len(warnings.output), 1)
                release.set()
                diag._worker.join(1)
            summary = json.loads((diag.session_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counters"]["frames_dropped"], 1)
            self.assertEqual(list((diag.session_dir / "frames").iterdir()), [])

    def test_close_does_not_wait_for_a_stalled_png_replace(self):
        with tempfile.TemporaryDirectory() as temp:
            entered = threading.Event()
            release = threading.Event()
            real_replace = Path.replace

            def delayed_replace(path, target):
                if path.name.endswith(".png.tmp"):
                    entered.set()
                    release.wait(2)
                return real_replace(path, target)

            diag = DiagnosticsSession.create(temp, {}, record_frames=True)
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            diag.frame_id(frame)  # finish the metadata replace before stalling PNG commit
            with patch.object(Path, "replace", delayed_replace), \
                    patch("core.diagnostics.CLOSE_TIMEOUT_SECONDS", 0.01):
                diag.record_frame(frame, reason="test")
                self.assertTrue(entered.wait(1))
                started = time.perf_counter()
                with self.assertLogs("core.diagnostics", level="WARNING"):
                    diag.close("test")
                self.assertLess(time.perf_counter() - started, 0.5)
                release.set()
                diag._worker.join(1)
            summary = json.loads(diag.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["counters"]["frames_dropped"], 1)
            self.assertFalse(list((diag.session_dir / "frames").glob("*.png")))

    def test_frame_worker_never_flushes_synchronous_jsonl_events(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = DiagnosticsSession.create(temp, {}, record_frames=True)
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            worker_flush = threading.Event()
            real_flush = diag._events.flush

            def track_flush():
                if threading.current_thread() is diag._worker:
                    worker_flush.set()
                    raise AssertionError("frame worker attempted synchronous JSONL flush")
                return real_flush()

            with patch.object(diag._events, "flush", side_effect=track_flush):
                diag.record_frame(frame, reason="test")
                diag.close("test")
            self.assertFalse(worker_flush.is_set())
            self.assertEqual(diag.counters["frames_saved"], 1)

    def test_inflight_png_reservation_keeps_concurrent_events_under_total_cap(self):
        class Frame:
            shape = (10, 10, 3)

            def copy(self):
                return self

        with tempfile.TemporaryDirectory() as temp:
            entered = threading.Event()
            release = threading.Event()
            real_replace = Path.replace

            def delayed_replace(path, target):
                if path.name.endswith(".png.tmp"):
                    entered.set()
                    release.wait(2)
                return real_replace(path, target)

            diag = DiagnosticsSession.create(temp, {}, record_frames=True, max_bytes=240000)
            frame = Frame()
            diag.frame_id(frame)
            fake_encoded = np.zeros(200000, dtype=np.uint8)
            with patch("core.diagnostics.cv2.imencode", return_value=(True, fake_encoded)), \
                    patch.object(Path, "replace", delayed_replace):
                diag.record_frame(frame, reason="budget_test")
                self.assertTrue(entered.wait(1))
                index = 0
                while diag.event("fill", data={"index": index, "padding": "x" * 900}):
                    index += 1
                release.set()
                diag.close("test")
            total = sum(path.stat().st_size for path in diag.session_dir.rglob("*") if path.is_file())
            summary = json.loads(diag.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["counters"]["frames_saved"], 1)
            self.assertLessEqual(total, 240000)
            self.assertLessEqual(summary["total_bytes"], 240000)

    def test_frame_rejected_after_event_budget_exhaustion_is_counted_as_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            entered = threading.Event()
            release = threading.Event()
            encoded = np.zeros(1000, dtype=np.uint8)

            def delayed_encode(_extension, _frame):
                entered.set()
                release.wait(2)
                return True, encoded

            diag = DiagnosticsSession.create(temp, {}, record_frames=True, max_bytes=8192)
            with patch("core.diagnostics.cv2.imencode", side_effect=delayed_encode):
                diag.record_frame(np.zeros((2, 2, 3), dtype=np.uint8), reason="budget_test")
                self.assertTrue(entered.wait(1))
                self.assertFalse(diag.event("fill", data={"padding": "x" * 5000}))
                release.set()
                diag.close("test")
            summary = json.loads(diag.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["counters"]["frames_seen"], 1)
            self.assertEqual(summary["counters"]["frames_saved"], 0)
            self.assertEqual(summary["counters"]["frames_dropped"], 1)

    def test_boundary_dedup_marker_is_consumed_when_rematch_first_deploy_has_no_gap(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        diag = DiagnosticsSession.create(temp.name, {})
        logic = GameLogic.__new__(GameLogic)
        logic.diagnostics = diag
        logic.run_state = RunState()
        logic.start_run(boundary_source="rematch_tap")
        match_before_deploy = diag._match_id
        diag.observe_phase("deploy")
        bot = MergeTacticsBot.__new__(MergeTacticsBot)
        bot.diagnostics = diag
        logic.deploy_phase = lambda _screen: {"elixir": 5}
        bot.game_logic = logic
        bot._last_deploy_ts = time.time() - 10
        bot.last_elixir_value = 0
        bot.handle_deploy_phase(FakeFrame(5))
        diag.observe_phase("battle")
        diag.observe_phase("deploy")
        bot._last_deploy_ts = time.time() - 100
        bot.handle_deploy_phase(FakeFrame(5))
        diag.close("test")

        events = [json.loads(line) for line in diag.events_path.read_text(encoding="utf-8").splitlines()]
        boundaries = [event for event in events if event["event_type"] == "match_boundary_assumed"]
        self.assertEqual(len(boundaries), 2)
        self.assertEqual(boundaries[0]["match_id"], match_before_deploy)
        self.assertNotEqual(boundaries[1]["match_id"], boundaries[0]["match_id"])
        self.assertFalse(boundaries[1]["data"]["deduplicated"])
        self.assertEqual(boundaries[1]["round_index"], 1)

    def test_unknown_phase_keeps_last_stable_phase_for_round_count(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = DiagnosticsSession.create(temp, {})
            diag.new_match_id("search_tap")
            diag.observe_phase("deploy", frame_id=1)
            diag.observe_phase("battle", frame_id=2)
            diag.observe_phase(None, frame_id=3)
            diag.observe_phase("deploy", frame_id=4)
            diag.close("test")

            events = [json.loads(line) for line in diag.events_path.read_text(encoding="utf-8").splitlines()]
            phases = [event for event in events if event["event_type"] == "phase"]
            self.assertEqual([event["round_index"] for event in phases], [1, 1, 2])
            summary = json.loads(diag.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["counters"]["phase_observations"], 3)
            self.assertEqual(summary["counters"]["phase_transitions"], 3)

    def test_full_frame_queue_drops_new_frames_and_counts_them(self):
        with tempfile.TemporaryDirectory() as temp:
            entered = threading.Event()
            release = threading.Event()
            real_imencode = cv2.imencode

            def delayed_encode(extension, frame):
                entered.set()
                release.wait(2)
                return real_imencode(extension, frame)

            diag = DiagnosticsSession.create(temp, {}, record_frames=True)
            with patch("core.diagnostics.cv2.imencode", side_effect=delayed_encode):
                first_id = diag.record_frame(np.zeros((2, 2, 3), dtype=np.uint8), reason="test")
                self.assertTrue(entered.wait(1))
                for _ in range(4):
                    diag.record_frame(np.zeros((2, 2, 3), dtype=np.uint8), reason="test")
                dropped_id = diag.record_frame(np.zeros((2, 2, 3), dtype=np.uint8), reason="test")
                self.assertIsNotNone(first_id)
                self.assertIsNotNone(dropped_id)
                self.assertEqual(diag.counters["frames_dropped"], 1)
                release.set()
                diag.close("test")
            self.assertEqual(diag.counters["frames_saved"], 5)

    def test_size_limit_stops_events_and_summary_reports_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = DiagnosticsSession.create(temp, {}, max_bytes=8192)
            self.assertTrue(diag.event("first", data={"value": "a" * 3000}))
            self.assertFalse(diag.event("over_limit", data={"value": "b" * 5000}))
            self.assertFalse(diag.event("smaller_after_limit", data={"value": "x"}))
            diag.close("test")
            summary = json.loads((diag.session_dir / "summary.json").read_text(encoding="utf-8"))
            events = [json.loads(line) for line in (diag.session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event_id"] for event in events], [1])
            self.assertTrue(summary["truncated"])

    def test_size_limit_prevents_copying_later_frames(self):
        class CountedFrame:
            shape = (2, 2, 3)

            def __init__(self):
                self.copy_count = 0

            def copy(self):
                self.copy_count += 1
                return self

        with tempfile.TemporaryDirectory() as temp:
            diag = DiagnosticsSession.create(temp, {}, record_frames=True, max_bytes=8192)
            diag.event("first", data={"value": "a" * 3000})
            diag.event("over_limit", data={"value": "b" * 5000})
            frame = CountedFrame()
            diag.record_frame(frame, reason="after_limit")
            diag.close("test")
            self.assertEqual(frame.copy_count, 0)
            summary = json.loads(diag.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["counters"]["frames_dropped"], 1)

    def test_file_byte_accounting_matches_disk_and_respects_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            session = DiagnosticsSession.create(temp, {"strategy": "legacy"}, max_bytes=8192)
            for index in range(100):
                session.event("synthetic", data={"index": index, "name": "Синтетика"})
            session.close("normal")

            total = sum(path.stat().st_size for path in session.session_dir.rglob("*") if path.is_file())
            summary = json.loads(session.summary_path.read_text(encoding="utf-8"))
            self.assertLessEqual(total, 8192)
            self.assertEqual(session.events_path.stat().st_size,
                             session._bytes_written - session.metadata_path.stat().st_size)
            self.assertTrue(summary["truncated"])
            self.assertLessEqual(summary["total_bytes"], 8192)


class DecisionDiagnosticsTests(unittest.TestCase):
    def _decision(self, cards, elixir, state):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        diag = DiagnosticsSession.create(temp.name, {})
        try:
            planner = DeployPlanner(None, None, diagnostics=diag)
            selected = planner.choose(cards, elixir, state, action_id="action-000009", frame_id=12)
        finally:
            diag.close("test")
        events = [json.loads(line) for line in (diag.session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        return selected, next(event for event in events if event["event_type"] == "decision")

    def test_duplicate_decision_records_counts_and_existing_shop_slot(self):
        duplicate = ShopCard("knight", 2, SHOP_CENTERS[1])
        state = RunState(copies={"knight": 1}, purchases=["knight"])

        selected, event = self._decision([duplicate], 5, state)

        self.assertIs(selected, duplicate)
        self.assertEqual(event["action_id"], "action-000009")
        self.assertEqual(event["frame_id"], 12)
        self.assertEqual(event["data"]["reason"], "duplicate")
        self.assertEqual(event["data"]["selected"]["slot_index"], 1)
        self.assertEqual(event["data"]["selected"]["copies_before"]["value"], 1)
        self.assertEqual(event["data"]["selected"]["copies_before"]["certainty"], "inferred")
        self.assertEqual(event["data"]["selected"]["copies_to_next_star"]["value"], 1)

    def test_target_decision_records_target_rank_without_changing_selection(self):
        target = ShopCard("knight", 2, SHOP_CENTERS[2])
        other = ShopCard("goblin_cage", 1, SHOP_CENTERS[0])

        selected, event = self._decision([other, target], 5, RunState())

        self.assertIs(selected, target)
        self.assertEqual(event["data"]["reason"], "target_tier_priority")
        self.assertIsInstance(event["data"]["selected"]["target_rank"], int)
        self.assertEqual(event["data"]["unaffordable_cards"], [])

    def test_cheapest_tie_records_original_shop_order(self):
        first = ShopCard("dark_prince", 2, SHOP_CENTERS[0])
        second = ShopCard("goblin_cage", 2, SHOP_CENTERS[2])

        selected, event = self._decision([first, second], 5, RunState())

        self.assertIs(selected, first)
        self.assertEqual(event["data"]["reason"], "cheapest_fallback")
        self.assertEqual(event["data"]["selection_tiebreak"], "original_order")

    def test_empty_and_unaffordable_shops_record_distinct_reasons(self):
        _, empty = self._decision([], 5, RunState())
        _, unaffordable = self._decision([ShopCard("knight", 2, SHOP_CENTERS[0])], 1, RunState())

        self.assertEqual(empty["data"]["reason"], "no_cards")
        self.assertEqual(unaffordable["data"]["reason"], "insufficient_elixir")
        self.assertEqual(unaffordable["data"]["unaffordable_cards"][0]["name"], "knight")

    def test_shop_reader_preserves_existing_slot_cost_source_and_inliers(self):
        reader = BoardReader.__new__(BoardReader)
        reader.include_diagnostics = True
        names = iter([("knight", 11), ("archer", 7), ("goblin", 4)])
        reader.identifier = type("Identifier", (), {"identify": lambda _self, _tile: next(names)})()
        costs = iter([3, None, None])
        reader._read_cost = lambda _screen, _index: next(costs)
        screen = np.zeros((1600, 900, 3), dtype=np.uint8)

        with patch("core.board_state.unit_costs", {"knight": 2, "archer": 1, "goblin": 1}):
            cards = reader.shop_cards(screen)

        self.assertEqual(cards[0].slot_index, 0)
        self.assertEqual(cards[0].cost_source, "recognized_digit")
        self.assertEqual(cards[0].name_inliers, 11)
        self.assertEqual(cards[1].slot_index, 1)
        self.assertEqual(cards[1].cost_source, "fallback_unit_costs")
        self.assertEqual(cards[1].name_inliers, 7)


class FakeFrame:
    def __init__(self, elixir):
        self.elixir = elixir

    def copy(self):
        return FakeFrame(self.elixir)


class FakeImageProcessor:
    def __init__(self, sale_positions=None):
        self.sale_positions = iter(sale_positions or [])

    def read_elixir(self, screen, _rect, force=False):
        return None if screen is None else screen.elixir

    def find_template(self, _screen, _path, _threshold):
        return next(self.sale_positions, None)


class FakeReader:
    def __init__(self, cards, orbs=None, sale_positions=None):
        self.cards = cards
        self.orb_list = orbs or []
        self.image_processor = FakeImageProcessor(sale_positions)

    def orbs(self, _screen):
        return list(self.orb_list)

    def own_units(self, _screen):
        return 2

    def shop_cards(self, _screen):
        return list(self.cards)


class FakeADB:
    def __init__(self, screens=None):
        self.screens = iter(screens or [])
        self.taps = []
        self.screenshots = 0

    def screenshot(self):
        self.screenshots += 1
        return next(self.screens, None)

    def tap(self, x, y):
        self.taps.append((x, y))


class ActionDiagnosticsTests(unittest.TestCase):
    def _session(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        diag = DiagnosticsSession.create(temp.name, {})
        self.addCleanup(diag.close, "test")
        return temp, diag

    @staticmethod
    def _events(diag):
        return [json.loads(line) for line in (diag.session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_purchase_decision_attempt_and_next_frame_share_action_id(self):
        temp, diag = self._session()
        screen = FakeFrame(2)
        after = FakeFrame(0)
        card = ShopCard("dark_prince", 2, SHOP_CENTERS[0])
        reader = FakeReader([card])
        adb = FakeADB([after])
        planner = DeployPlanner(reader, adb, diagnostics=diag)
        diag.record_frame(screen, reason="deploy")

        with patch("core.strategy.time.sleep"):
            summary = planner.play(screen, RunState())
        diag.close("test")

        events = self._events(diag)
        decision = next(event for event in events if event["event_type"] == "decision")
        attempt = next(event for event in events if event["event_type"] == "buy_attempt")
        outcome = next(event for event in events if event["event_type"] == "buy_outcome")
        self.assertEqual(summary["bought"], ["dark_prince"])
        self.assertEqual(attempt["action_id"], decision["action_id"])
        self.assertEqual(outcome["action_id"], decision["action_id"])
        self.assertEqual(attempt["frame_id"], decision["frame_id"])
        self.assertNotEqual(outcome["frame_id"], attempt["frame_id"])
        self.assertEqual(attempt["data"]["tap_status"], "attempted")
        self.assertEqual(outcome["data"]["confirmation"], "inferred")
        self.assertEqual(adb.taps, [card.center])

    def test_unknown_elixir_and_empty_shop_are_recorded_as_refusals(self):
        temp, diag = self._session()
        planner = DeployPlanner(FakeReader([ShopCard("knight", 2, SHOP_CENTERS[0])]),
                                FakeADB([None] * 10), diagnostics=diag)
        with patch("core.strategy.time.sleep"):
            result = planner.play(FakeFrame(None), RunState())
        self.assertEqual(result["reason"], "no_elixir")
        self.assertEqual(result["bought"], [])
        diag.close("test")
        refusal = next(event for event in self._events(diag) if event["event_type"] == "purchase_refused")
        self.assertEqual(refusal["data"]["reason"], "unknown_elixir")
        self.assertIsNone(refusal["data"]["recognized_elixir"])

        _, diag2 = self._session()
        planner = DeployPlanner(FakeReader([]), FakeADB([FakeFrame(2)] * 5), diagnostics=diag2)
        with patch("core.strategy.time.sleep"):
            result = planner.play(FakeFrame(2), RunState())
        diag2.close("test")
        no_cards = next(event for event in self._events(diag2) if event["event_type"] == "purchase_refused")
        self.assertEqual(result["reason"], "no_cards")
        self.assertEqual(no_cards["data"]["reason"], "no_cards")

    def test_sale_probe_logs_slots_and_unconfirmed_button_tap(self):
        temp, diag = self._session()
        reader = FakeReader([], sale_positions=[None, (15, 25)])
        adb = FakeADB([FakeFrame(2), FakeFrame(2)])
        planner = DeployPlanner(reader, adb, diagnostics=diag)

        result = planner._sell_one(FakeFrame(2), action_id="action-000004", reason="suspected_no_effect")
        diag.close("test")

        events = self._events(diag)
        probes = [event for event in events if event["event_type"] == "sale_slot_probe"]
        outcome = next(event for event in events if event["event_type"] == "sale_outcome")
        self.assertTrue(result)
        self.assertEqual(len(probes), 2)
        self.assertEqual([event["data"]["slot_center"] for event in probes], [list(p) for p in BENCH_CELLS[:2]])
        self.assertTrue(probes[0]["data"]["sell_button_found"] is False)
        self.assertTrue(probes[1]["data"]["sell_button_found"] is True)
        self.assertEqual(outcome["data"]["confirmation"], "unknown")
        self.assertEqual(adb.taps, [BENCH_CELLS[0], BENCH_CELLS[1], (15, 25)])

    def test_run_reset_and_observed_phases_keep_boundary_assumed(self):
        temp, diag = self._session()
        logic = GameLogic.__new__(GameLogic)
        logic.run_state = RunState(copies={"knight": 2})
        logic.diagnostics = diag

        logic.start_run()
        diag.observe_phase("deploy", frame_id=1)
        diag.observe_phase("battle", frame_id=2)
        diag.observe_phase("deploy", frame_id=3)
        diag.observe_phase(None, frame_id=4)
        diag.close("test")

        events = self._events(diag)
        boundary = next(event for event in events if event["event_type"] == "match_boundary_assumed")
        phases = [event for event in events if event["event_type"] == "phase"]
        self.assertEqual(boundary["data"]["certainty"], "assumed")
        self.assertEqual([event["data"]["phase"] for event in phases], ["deploy", "battle", "deploy"])
        self.assertEqual(phases[-1]["round_index"], 2)
        unknown = next(event for event in events if event["event_type"] == "phase_unknown")
        self.assertIsNone(unknown["phase"])
        self.assertEqual(logic.run_state.copies, {})

    def test_lobby_frame_is_saved_and_first_deploy_deduplicates_search_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            diag = DiagnosticsSession.create(temp, {}, record_frames=True)
            lobby = np.zeros((3, 4, 3), dtype=np.uint8)
            deploy = np.zeros((3, 4, 3), dtype=np.uint8)
            logic = GameLogic.__new__(GameLogic)
            logic.diagnostics = diag
            logic.run_state = RunState(copies={"knight": 1})
            bot = MergeTacticsBot.__new__(MergeTacticsBot)
            bot.diagnostics = diag
            bot.image_processor = SimpleNamespace(find_template=lambda *_: (10, 20))
            bot.adb = FakeADB()
            bot.game_logic = logic
            bot._deploy_played = False
            bot._last_deploy_ts = 0.0

            lobby_id = diag.frame_id(lobby)
            self.assertTrue(bot.search_match(lobby))
            logic.run_state.buy("knight")
            deploy_id = diag.frame_id(deploy)
            diag.observe_phase("deploy", frame_id=deploy_id)
            logic.start_run(boundary_source="deploy_gap")
            self.assertEqual(logic.run_state.copies, {"knight": 1})
            diag.record_frame(deploy, reason="phase_deploy")
            logic.start_run(boundary_source="deploy_gap")  # later gap is a new partial match
            diag.close("test")

            events = [json.loads(line) for line in diag.events_path.read_text(encoding="utf-8").splitlines()]
            boundaries = [event for event in events if event["event_type"] == "match_boundary_assumed"]
            deployment_phase = next(event for event in events if event["event_type"] == "phase")
            summary = json.loads(diag.summary_path.read_text(encoding="utf-8"))
            saved_ids = {int(path.stem) for path in (diag.session_dir / "frames").glob("*.png")}
            self.assertEqual(len(boundaries), 2)
            prevented = [event for event in events
                         if event["event_type"] == "match_boundary_prevented_reset"]
            self.assertEqual(len(prevented), 1)
            self.assertEqual(deployment_phase["data"]["phase"], "deploy")
            self.assertEqual(logic.run_state.copies, {})
            self.assertEqual(saved_ids, {lobby_id, deploy_id})
            later_boundaries = [event for event in self._events(diag)
                                if event["event_type"] == "match_boundary_assumed"]
            self.assertNotEqual(later_boundaries[-1]["match_id"], later_boundaries[0]["match_id"])
            self.assertFalse(later_boundaries[-1]["data"]["deduplicated"])
            self.assertEqual(summary["counters"]["matches_detected"], 2)

    def test_no_search_startup_keeps_observed_deploy_phase_on_partial_boundary(self):
        _, diag = self._session()
        logic = GameLogic.__new__(GameLogic)
        logic.diagnostics = diag
        logic.run_state = RunState()
        diag.observe_phase("deploy", frame_id=1)
        logic.start_run(boundary_source="deploy_gap")
        diag.close("test")
        boundary = next(event for event in self._events(diag)
                        if event["event_type"] == "match_boundary_assumed")
        self.assertEqual(boundary["phase"], "deploy")
        self.assertEqual(boundary["match_id"], "match-001")
        self.assertFalse(boundary["data"]["deduplicated"])

    def test_orbs_have_one_tap_owner_for_each_observed_frame(self):
        _, diag = self._session()
        screen = FakeFrame(0)
        card = ShopCard("dark_prince", 2, SHOP_CENTERS[0])
        reader = FakeReader([card], orbs=[(100, 200)])
        adb = FakeADB([FakeFrame(0)])
        diag.record_frame(screen, reason="deploy")
        bot = MergeTacticsBot.__new__(MergeTacticsBot)
        bot.diagnostics = diag
        bot.game_logic = type("Logic", (), {"reader": reader})()
        bot.adb = adb
        planner = DeployPlanner(reader, adb, diagnostics=diag)

        with patch("bot.time.sleep"), patch("core.strategy.time.sleep"):
            bot.collect_orbs(screen)
            planner.play(screen, RunState())
        diag.close("test")

        events = self._events(diag)
        taps = [event for event in events if event["event_type"] == "orb_tap_attempt"]
        self.assertEqual(len(taps), 1)
        self.assertEqual({event["frame_id"] for event in taps}, {1})
        self.assertEqual({event["source"] for event in taps},
                         {"MergeTacticsBot.collect_orbs"})
        self.assertTrue(any(event["event_type"] == "frame_reused" for event in events))
        self.assertEqual(adb.taps, [(100, 200)])
        self.assertEqual(adb.screenshots, 1)

    def test_existing_main_loop_screenshot_error_is_structured(self):
        _, diag = self._session()
        bot = MergeTacticsBot.__new__(MergeTacticsBot)
        bot.diagnostics = diag
        bot.adb = FakeADB()
        bot.adb.screenshot = lambda: (_ for _ in ()).throw(RuntimeError("adb down"))
        bot.game_logic = type("Logic", (), {"phase": lambda *_: None})()
        bot.handle_end_screen = lambda _screen: False
        bot.search_match = lambda _screen: False

        fake_time = SimpleNamespace(time=unittest.mock.Mock(side_effect=[0, 0, 2]), sleep=lambda _: None)
        with patch("bot.time", fake_time):
            bot.run(duration=1)
        diag.close("test")

        error = next(event for event in self._events(diag) if event["event_type"] == "screenshot_error")
        self.assertEqual(error["data"]["certainty"], "observed")
        self.assertEqual(error["data"]["error_type"], "RuntimeError")
        self.assertNotIn("message", error["data"])

    def test_search_and_end_screen_actions_are_linked_to_detected_frames(self):
        _, diag = self._session()
        screen = FakeFrame(2)
        calls = []

        def find_template(_screen, path, _threshold):
            calls.append(path)
            if path in (SEARCH_BUTTON, END_BUTTONS["rematch"]):
                return (31, 47)
            return None

        bot = MergeTacticsBot.__new__(MergeTacticsBot)
        bot.diagnostics = diag
        bot.image_processor = SimpleNamespace(find_template=find_template)
        bot.adb = FakeADB()
        bot.game_logic = type("Logic", (), {"start_run": lambda _self, **_kwargs: None})()
        bot._deploy_played = True
        bot._last_deploy_ts = 3.0

        with patch("bot.time.sleep"):
            self.assertTrue(bot.search_match(screen))
            self.assertTrue(bot.handle_end_screen(screen))
        diag.close("test")

        events = self._events(diag)
        search = next(event for event in events if event["event_type"] == "match_search_attempt")
        end_screen = next(event for event in events if event["event_type"] == "match_end_screen")
        end_attempt = next(event for event in events if event["event_type"] == "end_screen_action_attempt")
        self.assertEqual(search["action_id"], next(e for e in events if e["event_type"] == "match_search_tap_returned")["action_id"])
        self.assertEqual(end_screen["action_id"], end_attempt["action_id"])
        self.assertEqual(bot.adb.taps, [(31, 47), (31, 47)])
        self.assertEqual(calls[0], SEARCH_BUTTON)
        self.assertEqual(calls[1], END_BUTTONS["rematch"])

    def test_adb_tap_error_is_recorded_and_reraised_unchanged(self):
        _, diag = self._session()

        class FailingADB(FakeADB):
            def tap(self, _x, _y):
                raise RuntimeError("device 'SERIAL-1234' not found")

        planner = DeployPlanner(FakeReader([ShopCard("dark_prince", 2, SHOP_CENTERS[0])]),
                                FailingADB(), diagnostics=diag)
        with patch("core.strategy.time.sleep"), self.assertRaisesRegex(RuntimeError, "SERIAL-1234"):
            planner.play(FakeFrame(2), RunState())
        diag.close("test")

        errors = [event for event in self._events(diag) if event["event_type"] == "adb_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["data"]["operation"], "buy_tap")
        self.assertEqual(errors[0]["data"]["error_type"], "RuntimeError")
        self.assertNotIn("SERIAL-1234", json.dumps(self._events(diag)))

    def test_planner_orb_and_sale_tap_errors_are_recorded_without_exception_text(self):
        secret = "SERIAL-5678"

        class FailingADB(FakeADB):
            def tap(self, _x, _y):
                raise RuntimeError(f"device '{secret}' not found")

        cases = (
            ("sale_slot_tap", lambda reader, adb, diag: DeployPlanner(reader, adb, diagnostics=diag)._sell_one(
                FakeFrame(2))),
        )
        for operation, invoke in cases:
            with self.subTest(operation=operation):
                _, diag = self._session()
                reader = FakeReader([])
                with patch("core.strategy.time.sleep"), self.assertRaisesRegex(RuntimeError, secret):
                    invoke(reader, FailingADB(), diag)
                diag.close("test")
                events = self._events(diag)
                error = next(event for event in events if event["event_type"] == "adb_error")
                self.assertEqual(error["data"]["operation"], operation)
                self.assertNotIn(secret, json.dumps(events))

        _, diag = self._session()

        class SaleButtonFailingADB(FakeADB):
            def tap(self, x, y):
                if self.taps:
                    raise RuntimeError(f"device '{secret}' not found")
                self.taps.append((x, y))

        reader = FakeReader([], sale_positions=[(5, 6)])
        with patch("core.strategy.time.sleep"), self.assertRaisesRegex(RuntimeError, secret):
            DeployPlanner(reader, SaleButtonFailingADB([FakeFrame(2)]), diagnostics=diag)._sell_one(FakeFrame(2))
        diag.close("test")
        events = self._events(diag)
        error = next(event for event in events if event["event_type"] == "adb_error")
        self.assertEqual(error["data"]["operation"], "sale_button_tap")
        self.assertNotIn(secret, json.dumps(events))


class DiagnosticsCLITests(unittest.TestCase):
    def test_plain_launch_keeps_existing_constructor_and_defaults(self):
        with patch.object(sys, "argv", ["bot.py"]), \
                patch("bot.DiagnosticsSession.create") as create_session, \
                patch("bot.MergeTacticsBot") as bot_class:
            bot.main()

        create_session.assert_not_called()
        bot_class.assert_called_once_with()
        bot_class.return_value.run.assert_called_once_with(auto_search=True, duration=None)

    def test_record_frames_implies_session_and_metadata_has_only_safe_flags(self):
        diag = unittest.mock.Mock(enabled=True)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(sys, "argv", ["bot.py", "--no-search", "--duration", "45",
                                            "--record-frames", "--diagnostics-dir", temp,
                                            "--diagnostics-max-mb", "2"]), \
                patch("bot.DiagnosticsSession.create", return_value=diag) as create_session, \
                patch("bot.MergeTacticsBot") as bot_class:
            bot.main()

        kwargs = create_session.call_args.kwargs
        self.assertEqual(kwargs["root"], temp)
        self.assertTrue(kwargs["record_frames"])
        self.assertEqual(kwargs["max_bytes"], 2 * 1024 * 1024)
        metadata = kwargs["metadata"]
        self.assertTrue(metadata["parameters"]["diagnostics"])
        self.assertTrue(metadata["parameters"]["record_frames"])
        self.assertFalse(metadata["parameters"]["auto_search"])
        self.assertEqual(metadata["parameters"]["duration_seconds"], 45)
        self.assertNotIn("diagnostics_dir", metadata["parameters"])
        self.assertEqual(metadata["strategy"], "legacy")
        self.assertNotIn("ADB_SERIAL", json.dumps(metadata))
        bot_class.assert_called_once_with(diagnostics=diag)
        bot_class.return_value.run.assert_called_once_with(auto_search=False, duration=45.0)
        diag.close.assert_called_once_with("normal")

    def test_diagnostics_init_error_warns_and_keeps_bot_running(self):
        with patch.object(sys, "argv", ["bot.py", "--diagnostics"]), \
                patch("bot.DiagnosticsSession.create", side_effect=OSError("disk unavailable")), \
                patch("bot.MergeTacticsBot") as bot_class, \
                self.assertLogs("config.logging_config", level="WARNING") as warnings:
            bot.main()

        self.assertEqual(len(warnings.output), 1)
        bot_class.assert_called_once_with()
        bot_class.return_value.run.assert_called_once_with(auto_search=True, duration=None)

    def test_nonpositive_size_limit_is_rejected_before_bot_start(self):
        with patch.object(sys, "argv", ["bot.py", "--diagnostics", "--diagnostics-max-mb", "0"]), \
                patch("bot.MergeTacticsBot") as bot_class:
            with self.assertRaises(SystemExit):
                bot.main()
        bot_class.assert_not_called()

    def test_interrupt_closes_session_without_swallowing_other_exceptions(self):
        diag = unittest.mock.Mock(enabled=True)
        with patch.object(sys, "argv", ["bot.py", "--diagnostics"]), \
                patch("bot.DiagnosticsSession.create", return_value=diag), \
                patch("bot.MergeTacticsBot") as bot_class:
            bot_class.return_value.run.side_effect = KeyboardInterrupt
            bot.main()
        diag.close.assert_called_once_with("keyboard_interrupt")

        diag = unittest.mock.Mock(enabled=True)
        with patch.object(sys, "argv", ["bot.py", "--diagnostics"]), \
                patch("bot.DiagnosticsSession.create", return_value=diag), \
                patch("bot.MergeTacticsBot") as bot_class:
            bot_class.return_value.run.side_effect = RuntimeError("game error")
            with self.assertRaisesRegex(RuntimeError, "game error"):
                bot.main()
        diag.close.assert_called_once_with("error")

    def test_constructor_error_still_closes_diagnostics_and_is_propagated(self):
        diag = unittest.mock.Mock(enabled=True)
        with patch.object(sys, "argv", ["bot.py", "--diagnostics"]), \
                patch("bot.DiagnosticsSession.create", return_value=diag), \
                patch("bot.MergeTacticsBot", side_effect=RuntimeError("adb init failed")):
            with self.assertRaisesRegex(RuntimeError, "adb init failed"):
                bot.main()
        diag.close.assert_called_once_with("error")


class RecordingAnalyzerTests(unittest.TestCase):
    def _event(self, event_id, event_type, data=None, frame_id=None):
        return {
            "schema_version": 1, "event_id": event_id, "event_type": event_type,
            "timestamp": "2026-11-27T12:00:00+03:00", "elapsed_ms": event_id * 10,
            "session_id": "session-test", "match_id": "match-001", "round_index": 1,
            "phase": None, "frame_id": frame_id, "action_id": None,
            "source": "test", "data": data or {},
        }

    def test_complete_recording_reports_summary_decisions_state_and_timeline_without_mutation(self):
        from utils.analyze_recording import analyze_session

        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            (session / "metadata.json").write_text(
                json.dumps({"strategy": "legacy", "started_at": "2026-11-27T12:00:00+03:00"}),
                encoding="utf-8",
            )
            events = [
                self._event(1, "phase", {"phase": "deploy", "round_index_certainty": "assumed"}),
                self._event(2, "decision", {"reason": "target_tier_priority", "chosen": "knight",
                    "elixir": {"value": 5, "certainty": "observed"}} , frame_id=4),
                self._event(3, "state_snapshot", {"elixir": {"value": 3, "certainty": "observed"}}),
                self._event(4, "buy_outcome", {"legacy_classification": "accepted_by_legacy_logic",
                    "confirmation": "assumed", "card": "knight"}),
                self._event(5, "calculated_upgrade", {"threshold_reached": True, "copies": 2,
                    "certainty": "calculated"}),
                self._event(6, "sale_outcome", {"outcome": "button_tap_attempted", "confirmation": "unknown"}),
                self._event(7, "phase_unknown", {"phase": None, "certainty": "unknown"}),
                self._event(8, "frame_reused", {"reason": "same object"}, frame_id=4),
            ]
            (session / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            (session / "summary.json").write_text(json.dumps({"finish_reason": "normal",
                "duration_ms": 120, "counters": {"events_written": 8, "frames_reused": 1},
                "truncated": False}), encoding="utf-8")
            before = {p.name: p.read_bytes() for p in session.iterdir()}

            output = analyze_session(session, verbose=True)

            self.assertIn("Диагностика", output)
            self.assertIn("normal", output)
            self.assertIn("deploy", output)
            self.assertIn("target_tier_priority", output)
            self.assertIn("elixir", output.lower())
            self.assertIn("assumed", output)
            self.assertIn("sale_outcome", output)
            self.assertIn("unknown", output)
            self.assertIn("frame_reused", output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in session.iterdir()})

    def test_partial_session_skips_bad_lines_and_reports_missing_files_and_unknowns(self):
        from utils.analyze_recording import analyze_session

        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            (session / "events.jsonl").write_bytes(
                (json.dumps(self._event(1, "phase_unknown", {"phase": None})) + "\n"
                 + "{truncated json\n"
                 + json.dumps(self._event(2, "purchase_refused", {"reason": "unknown_elixir",
                     "elixir": {"value": None, "certainty": "unknown"}})) + "\n").encode("utf-8")
                + b'{"event_type":"bad","data":"\xff"}\n'
            )

            output = analyze_session(session)

            self.assertIn("неполная", output.lower())
            self.assertIn("повреждённых строк: 2", output.lower())
            self.assertIn("unknown_elixir", output)
            self.assertIn("отсутствует", output.lower())

    def test_analyzer_cli_accepts_verbose(self):
        from utils.analyze_recording import main

        with tempfile.TemporaryDirectory() as temp, patch("sys.argv", ["analyze_recording.py", temp, "--verbose"]):
            (Path(temp) / "events.jsonl").write_text("", encoding="utf-8")
            with patch("builtins.print") as output:
                main()
        self.assertIn("Диагностика", output.call_args.args[0])

    def test_summary_truncation_marks_analyzer_report_partial(self):
        from utils.analyze_recording import analyze_session

        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            (session / "metadata.json").write_text("{}", encoding="utf-8")
            (session / "events.jsonl").write_text("", encoding="utf-8")
            (session / "summary.json").write_text(json.dumps({"truncated": True}), encoding="utf-8")
            report = analyze_session(session)
        self.assertIn("неполная", report.lower())


class DiagnosticsParityTests(unittest.TestCase):
    def test_diagnostics_preserves_planner_decisions_actions_frames_sleeps_and_state(self):
        class Frame:
            pass

        class FakeADB:
            def __init__(self, frames):
                self.frames = list(frames)
                self.taps = []
                self.screenshot_requests = 0

            def tap(self, x, y):
                self.taps.append((x, y))

            def screenshot(self):
                self.screenshot_requests += 1
                return self.frames.pop(0) if self.frames else Frame()

        class FakeImageProcessor:
            def __init__(self, elixir_by_frame):
                self.elixir_by_frame = elixir_by_frame

            def read_elixir(self, frame, _rect, force=False):
                return self.elixir_by_frame.get(frame)

        class FakeReader:
            def __init__(self, initial, card, image_processor):
                self.initial = initial
                self.card = card
                self.image_processor = image_processor

            def orbs(self, _frame):
                return []

            def own_units(self, _frame):
                return 2

            def shop_cards(self, frame):
                return [self.card] if frame is self.initial else []

        def run_case(enabled):
            initial = Frame()
            updated = Frame()
            retries = [Frame() for _ in range(5)]
            choice = ShopCard("knight", 2, (123, 456))
            image_processor = FakeImageProcessor({initial: 3, updated: 0})
            reader = FakeReader(initial, choice, image_processor)
            adb = FakeADB([updated, *retries])
            diag = None
            temp = tempfile.TemporaryDirectory()
            if enabled:
                diag = DiagnosticsSession.create(temp.name, {"strategy": "legacy"})
            planner = DeployPlanner(reader, adb, tap_delay=0.25, diagnostics=diag)
            state = RunState()
            sleeps = []
            with patch("core.strategy.time.sleep", side_effect=sleeps.append):
                result = planner.play(initial, state)
            events = []
            if diag:
                diag.close("normal")
                events = [json.loads(line) for line in diag.events_path.read_text(encoding="utf-8").splitlines()]
            snapshot = (result, dict(state.copies), list(state.purchases), adb.taps,
                        adb.screenshot_requests, sleeps)
            temp.cleanup()
            return snapshot, events

        disabled, _ = run_case(False)
        enabled, events = run_case(True)

        self.assertEqual(disabled, enabled)
        self.assertEqual(enabled[0]["bought"], ["knight"])
        self.assertEqual(enabled[3], [(123, 456)])
        self.assertEqual(enabled[4], 6)
        self.assertEqual(enabled[5], [0.25, 0.6, 0.6, 0.6, 0.6, 0.6])
        decision = next(event for event in events if event["event_type"] == "decision")
        self.assertEqual(decision["data"]["selected"]["name"], "knight")


if __name__ == "__main__":
    unittest.main()
