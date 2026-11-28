import unittest

import cv2
import numpy as np

from core.bench_reader import BenchReader, BenchObserver
from unittest.mock import Mock, patch
import sys
import tempfile
from pathlib import Path
import json
import bot
from utils.analyze_recording import analyze_session
from utils.inspect_bench import inspect_image


class BenchReaderTests(unittest.TestCase):
    def setUp(self):
        self.background = cv2.imread('templates/empty_bench.png')
        self.model = cv2.imread('templates/bench/archer.png')
        self.reader = BenchReader()

    def frame(self, occupied=()):
        frame = np.zeros((1600, 900, 3), dtype=np.uint8)
        frame[1234:1346, 160:646] = self.background
        for index in occupied:
            x = 207 + index * 94 - self.model.shape[1] // 2
            cv2.ellipse(frame, (207 + index * 94, 1290), (30, 17), 0, 0, 360, (25, 30, 35), -1)
            frame[1230:1230+self.model.shape[0], x:x+self.model.shape[1]] = self.model
        return frame

    def test_empty_bench(self):
        obs = self.reader.read(self.frame(), frame_id=12)
        self.assertEqual((obs.empty_count, obs.occupied_count, obs.unknown_count), (5, 0, 0))
        self.assertEqual(obs.frame_id, 12)

    def test_dimmed_empty_bench_is_not_occupied(self):
        image = (self.frame().astype(np.float32) * 0.65).astype(np.uint8)
        self.assertEqual(self.reader.read(image).empty_count, 5)

    def test_partially_and_fully_occupied(self):
        for occupied in ((0, 2), (0, 1, 2, 3, 4)):
            obs = self.reader.read(self.frame(occupied))
            self.assertEqual([s.occupancy for s in obs.slots],
                             ['occupied' if i in occupied else 'empty' for i in range(5)])

    def test_neighbor_overlap_does_not_claim_another_unit(self):
        image = self.frame((0,))
        cv2.rectangle(image, (266, 1252), (284, 1313), (20, 20, 20), -1)
        self.assertNotEqual(self.reader.read(image).slots[1].occupancy, 'occupied')

    def test_bright_merge_effect_is_not_reported_as_empty(self):
        image = self.frame()
        cv2.circle(image, (207, 1280), 8, (255, 255, 255), -1)
        self.assertEqual(self.reader.read(image).slots[0].occupancy, 'unknown')

    def test_absent_invalid_and_obscured_images_are_unknown(self):
        for image in (None, np.zeros((5, 5, 3), np.uint8), np.zeros((1600, 900, 3), np.uint8)):
            self.assertEqual(self.reader.read(image).unknown_count, 5)

    def test_missing_background_does_not_claim_empty(self):
        self.assertEqual(BenchReader(background_path='missing.png').read(self.frame()).unknown_count, 5)

    def test_unknown_identity_and_stars_are_independent(self):
        reader = BenchReader(model_dir='missing')
        slot = reader.read(self.frame((0,))).slots[0]
        self.assertEqual(slot.occupancy, 'occupied')
        self.assertIsNone(slot.unit_name)
        self.assertIsNone(slot.stars)

    def test_recognized_identity_does_not_imply_stars(self):
        slot = self.reader.read(self.frame((2,))).slots[2]
        self.assertEqual(slot.unit_name, 'archer')
        self.assertIsNone(slot.stars)

    def test_alternate_visual_reference_keeps_character_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            variant = cv2.flip(self.model, 1)
            cv2.imwrite(str(Path(folder) / 'archer__turn.png'), variant)
            frame = self.frame((0,))
            x = 207 - variant.shape[1] // 2
            frame[1230:1230 + variant.shape[0], x:x + variant.shape[1]] = variant
            slot = BenchReader(model_dir=folder).read(frame).slots[0]
            self.assertEqual(slot.unit_name, 'archer')

    def test_overlapping_references_do_not_identify_one_model_twice(self):
        rng = np.random.default_rng(23)
        unit = rng.integers(0, 256, (62, 170, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            cv2.imwrite(str(Path(folder) / 'unit.png'), unit)
            cv2.imwrite(str(Path(folder) / 'unit__detail.png'), unit[:, -25:])
            image = self.frame()
            image[1252:1314, 122:292] = unit
            image[1252:1314, 292:325] = 20
            slots = BenchReader(model_dir=folder).read(image).slots
            self.assertEqual(sum(slot.unit_name == 'unit' for slot in slots[:2]), 1)

    def test_two_separate_matching_models_can_share_a_name(self):
        slots = self.reader.read(self.frame((0, 1))).slots
        self.assertEqual([slot.unit_name for slot in slots[:2]], ['archer', 'archer'])

    def test_partially_overlapping_references_do_not_identify_one_model_twice(self):
        unit = np.random.default_rng(37).integers(0, 256, (62, 204, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            cv2.imwrite(str(Path(folder) / 'unit.png'), unit[:, :110])
            cv2.imwrite(str(Path(folder) / 'unit__detail.png'), unit[:, -110:])
            image = self.frame()
            image[1252:1314, 152:356] = unit
            slots = BenchReader(model_dir=folder).read(image).slots
            self.assertEqual(sum(slot.unit_name == 'unit' for slot in slots[:2]), 1)

    def test_overlapping_references_with_different_names_do_not_identify_twice(self):
        unit = np.random.default_rng(37).integers(0, 256, (62, 204, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            cv2.imwrite(str(Path(folder) / 'unit.png'), unit[:, :110])
            cv2.imwrite(str(Path(folder) / 'other.png'), unit[:, -110:])
            image = self.frame()
            image[1252:1314, 152:356] = unit
            slots = BenchReader(model_dir=folder).read(image).slots
            self.assertEqual(sum(slot.unit_name is not None for slot in slots[:2]), 1)


class BenchIntegrationTests(unittest.TestCase):
    def test_observer_records_frame_and_changes(self):
        diagnostics = Mock(enabled=True)
        diagnostics.record_frame.return_value = 7
        observer = BenchObserver(diagnostics, BenchReader(model_dir='missing'))
        image = np.zeros((1600, 900, 3), np.uint8)
        observer.observe(image, reason='test')
        event = diagnostics.event.call_args
        self.assertEqual(event.args[0], 'bench_observation')
        self.assertEqual(event.kwargs['frame_id'], 7)
        self.assertEqual(event.kwargs['data']['unknown_count'], 5)
        self.assertIn('changes', event.kwargs['data'])

    def test_diagnostic_error_does_not_interrupt_game(self):
        diagnostics = Mock(enabled=True)
        diagnostics.record_frame.side_effect = RuntimeError('disk problem')
        BenchObserver(diagnostics).observe(np.zeros((1600, 900, 3), np.uint8), 'test')

    def test_inspector_saves_annotation_and_five_crops(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.zeros((1600, 900, 3), np.uint8)
            source = Path(folder) / 'input.png'
            cv2.imwrite(str(source), image)
            output = Path(folder) / 'output'
            inspect_image(source, output, BenchReader(model_dir='missing'))
            self.assertTrue((output / 'input_annotated.png').is_file())
            self.assertEqual(len(list(output.glob('input_slot_*.png'))), 5)

    def test_analyzer_reports_bench_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'events.jsonl').write_text(json.dumps({'event_type': 'bench_observation',
                'data': {'occupied_count': 2, 'empty_count': 2, 'unknown_count': 1}}) + '\n')
            report = analyze_session(path)
        self.assertIn('\u0421\u043a\u0430\u043c\u0435\u0439\u043a\u0430: \u043d\u0430\u0431\u043b\u044e\u0434\u0435\u043d\u0438\u0439=1', report)

    def test_cli_enables_diagnostics_and_observer(self):
        diagnostics = Mock(enabled=True)
        with patch.object(sys, 'argv', ['bot.py', '--observe-bench']), \
             patch('bot.DiagnosticsSession.create', return_value=diagnostics), \
             patch('bot.MergeTacticsBot') as bot_class:
            bot.main()
        bot_class.assert_called_once_with(diagnostics=diagnostics, observe_bench=True)


if __name__ == '__main__':
    unittest.main()
