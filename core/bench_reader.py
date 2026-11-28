"""Conservative visual observations of the local reserve on 900x1600 frames."""
from dataclasses import asdict, dataclass, replace
import logging
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class BenchSlot:
    index: int
    center: tuple
    bounds: tuple
    occupancy: str = 'unknown'
    unit_name: str | None = None
    stars: int | None = None
    occupancy_certainty: str = 'unknown'
    identity_certainty: str = 'unknown'
    stars_certainty: str = 'unknown'
    recognition_method: str | None = None
    recognition_score: float | None = None
    occupancy_score: float | None = None
    unit_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class BenchObservation:
    slots: list
    occupied_count: int
    empty_count: int
    unknown_count: int
    frame_id: int | None
    duration_ms: float


class BenchReader:
    def __init__(self, background_path='templates/empty_bench.png', model_dir='templates/bench'):
        self.background = cv2.imread(str(background_path)) if Path(background_path).is_file() else None
        self.models = []
        for path in Path(model_dir).glob('*.png'):
            image = cv2.imread(str(path))
            if image is not None:
                self.models.append((path.stem.split('__', 1)[0], image))

    def read(self, screen, frame_id=None):
        started = time.perf_counter()
        slots = []
        matches = []
        valid = (isinstance(screen, np.ndarray) and screen.shape == (1600, 900, 3)
                 and screen.dtype == np.uint8 and self.background is not None
                 and self.background.shape == (112, 486, 3))
        if valid:
            bench = screen[1234:1346, 160:646]
            # A corner remains visible even when the first or last slot is occupied.
            anchors = [(slice(1, 12), slice(1, 30)), (slice(1, 12), slice(450, 480))]
            ratios = []
            for a in anchors:
                ref = self.background[a].astype(np.float32)
                observed = bench[a].astype(np.float32)
                ratio = float(np.median((observed.mean(axis=2) + 1) / (ref.mean(axis=2) + 1)))
                error = np.mean(np.abs(observed - ref * ratio))
                if .5 <= ratio <= 1.3 and error < 30:
                    ratios.append(ratio)
            valid = bool(ratios)
            lighting = max(ratios) if valid else 1.0
        for index in range(5):
            center = (207 + 94 * index, 1280)
            bounds = (160 + index * 94, 1234, 160 + (index + 1) * 94, 1346)
            if not valid:
                slots.append(BenchSlot(index, center, bounds))
                matches.append(None)
                continue
            x = 12 + index * 94
            reference = self.background[18:80, x:x+75]
            crop = bench[18:80, x:x+75]
            hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
            # Exclude the shadow that is baked into the first template slot.
            mask = (hsv[:, :, 2] > 100) & (hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 40)
            difference = np.abs(reference.astype(np.float32) * lighting - crop.astype(np.float32)).mean(axis=2)
            score = float(np.mean(difference[mask] > 35)) if mask.any() else None
            occupancy = ('empty' if score is not None and score < .15 else
                         'occupied' if score is not None and score > .35 else 'unknown')
            if occupancy == 'empty':
                observed_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                bright = (observed_hsv[:, :, 2] > 220) & (observed_hsv[:, :, 1] < 100)
                if np.mean(bright) > .04:
                    occupancy = 'unknown'
            name, match_score, match_box = None, None, None
            if occupancy == 'occupied':
                candidates = []
                for unit, template in self.models:
                    h, w = template.shape[:2]
                    x0 = max(0, center[0] - w // 2 - 24)
                    x1 = min(900, center[0] - w // 2 + 24 + w)
                    region = screen[1190:min(1600, 1280+h), x0:x1]
                    if region.shape[0] >= h and region.shape[1] >= w:
                        _, value, _, location = cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED))
                        candidates.append((value, unit, (x0 + location[0], 1190 + location[1], w, h)))
                if candidates:
                    match_score, match_name, match_box = max(candidates)
                    if match_score >= .75:
                        name = match_name
            slots.append(BenchSlot(index, center, bounds, occupancy, name, None,
                                   'observed' if occupancy != 'unknown' else 'unknown',
                                   'observed' if name else 'unknown', 'unknown',
                                   'template_correlation' if match_score is not None else None,
                                   match_score, score, match_box if name else None))
            matches.append(match_box if name else None)
        for index in range(1, len(slots)):
            first, second = matches[index - 1:index + 1]
            if not first or not second:
                continue
            ax, ay, aw, ah = first
            bx, by, bw, bh = second
            overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
            if overlap:
                loser = index if slots[index - 1].recognition_score >= slots[index].recognition_score else index - 1
                slots[loser] = replace(slots[loser], unit_name=None, identity_certainty='unknown')
                matches[loser] = None
        return BenchObservation(slots, sum(s.occupancy == 'occupied' for s in slots),
                                sum(s.occupancy == 'empty' for s in slots),
                                sum(s.occupancy == 'unknown' for s in slots), frame_id,
                                (time.perf_counter() - started) * 1000)


class BenchObserver:
    """Write visual observations without feeding them into strategy state."""
    def __init__(self, diagnostics, reader=None):
        self.diagnostics = diagnostics
        self.reader = reader or BenchReader()
        self.previous = None

    def observe(self, screen, reason):
        if screen is None:
            return
        frame_id = None
        try:
            frame_id = self.diagnostics.record_frame(screen, reason=reason)
            result = self.reader.read(screen, frame_id)
            slots = [asdict(slot) for slot in result.slots]
            keys = ('occupancy', 'unit_name', 'stars')
            changes = []
            if self.previous is not None:
                for old, new in zip(self.previous, slots):
                    for key in keys:
                        if old[key] != new[key]:
                            changes.append({'index': new['index'], 'field': key,
                                            'from': old[key], 'to': new[key]})
            data = asdict(result)
            data.update({'reason': reason, 'changes': changes})
            self.diagnostics.event('bench_observation', frame_id=frame_id,
                                   source='BenchObserver', data=data)
            if self.previous is None or changes:
                logging.getLogger(__name__).info('[BENCH] %s', ' '.join(
                    f"{s['index']}:{s['occupancy']}:{s['unit_name'] or '?'}:{s['stars'] or '?'}" for s in slots))
            self.previous = slots
        except Exception as exc:
            try:
                self.diagnostics.event('bench_recognition_error', frame_id=frame_id,
                                       source='BenchObserver', data={'error_type': type(exc).__name__})
            except Exception:
                logging.getLogger(__name__).warning('[BENCH] diagnostics failed')
