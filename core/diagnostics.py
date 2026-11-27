"""Optional local recording for observations already available to the bot."""

import json
import logging
import queue
import threading
import time
import uuid
import weakref
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2


logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
FRAME_QUEUE_SIZE = 4
SUMMARY_RESERVE_BYTES = 4096
CLOSE_TIMEOUT_SECONDS = 1.0
_UNSET = object()


class NoOpDiagnostics:
    """Disabled diagnostics; methods do no disk or observation work."""

    enabled = False
    session_id = None

    def event(self, event_type, *, data=None, frame_id=None, action_id=None,
              match_id=None, round_index=None, phase=None, source=None):
        return None

    def record_frame(self, frame, *, reason):
        return None

    def frame_id(self, frame):
        return None

    def new_action_id(self):
        return None

    def new_match_id(self, boundary_source=None):
        return None

    def observe_phase(self, phase, *, frame_id=None):
        return None

    def consume_pending_first_deploy_boundary(self):
        return None

    def close(self, reason):
        return None


class DiagnosticsSession:
    """Owns one session's JSONL stream, counters, and optional PNG worker."""

    enabled = True

    @classmethod
    def create(cls, root, metadata, record_frames=False, max_bytes=536870912):
        if not isinstance(max_bytes, int) or max_bytes < SUMMARY_RESERVE_BYTES * 2:
            raise ValueError(f"max_bytes must be at least {SUMMARY_RESERVE_BYTES * 2}")

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        started = datetime.now().astimezone()
        while True:
            session_id = f"session-{uuid.uuid4().hex}"
            folder = f"{started.strftime('%Y-%m-%d_%H-%M-%S')}_{session_id}"
            session_dir = root / folder
            try:
                session_dir.mkdir()
                break
            except FileExistsError:
                continue

        return cls(session_dir, session_id, started, metadata,
                   record_frames=record_frames, max_bytes=max_bytes)

    def __init__(self, session_dir, session_id, started, metadata,
                 record_frames=False, max_bytes=536870912):
        self.session_dir = Path(session_dir)
        self.session_id = session_id
        self.started_at = started
        self._started_mono = time.monotonic()
        self._max_bytes = max_bytes
        self._record_frames = bool(record_frames)
        self._lock = threading.RLock()
        self._next_event_id = 1
        self._next_frame_id = 1
        self._next_action_id = 1
        self._next_match_id = 1
        self._frame_refs = {}
        self._size_limited = False
        self._counters = Counter({
            "events_written": 0,
            "events_dropped": 0,
            "frames_seen": 0,
            "frames_saved": 0,
            "frames_dropped": 0,
            "frames_reused": 0,
            "diagnostic_errors": 0,
            "matches_detected": 0,
            "deploy_phases": 0,
            "battle_phases": 0,
            "phase_observations": 0,
            "phase_transitions": 0,
            "decisions": 0,
            "decision_duration_ms_total": 0,
            "buy_attempts": 0,
            "strategy_accounted_purchases": 0,
            "assumed_successful_purchases": 0,
            "suspected_failed_purchases": 0,
            "sale_attempts": 0,
            "sale_button_taps": 0,
            "calculated_upgrades": 0,
            "orb_taps": 0,
            "confirmed_actions": 0,
            "failed_actions": 0,
            "unknown_actions": 0,
            "interrupted_actions": 0,
            "elixir_errors": 0,
            "no_card_observations": 0,
        })
        self._event_types = Counter()
        self._decision_reasons = Counter()
        self._truncated = False
        self._closing = False
        self._closed = False
        self._warned = False
        self._reserved_bytes = 0
        self._frame_size = None
        self._match_id = None
        self._round_index = None
        self._phase = None
        self._boundary_source = None
        self._last_boundary_deduplicated = False
        self._pending_first_deploy_boundary = False

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.session_dir / "metadata.json"
        self.events_path = self.session_dir / "events.jsonl"
        self.summary_path = self.session_dir / "summary.json"
        if self._record_frames:
            (self.session_dir / "frames").mkdir()

        metadata_record = dict(metadata or {})
        metadata_record.update({
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": started.isoformat(timespec="milliseconds"),
            "diagnostics_mode": "frames" if self._record_frames else "events",
        })
        metadata_record.setdefault("frame_size", None)
        self._metadata_record = metadata_record
        metadata_bytes = self._json_bytes(metadata_record)
        if len(metadata_bytes) + SUMMARY_RESERVE_BYTES > self._max_bytes:
            raise ValueError("diagnostics metadata exceeds the configured session size limit")
        self.metadata_path.write_bytes(metadata_bytes)
        self._bytes_written = len(metadata_bytes)
        # Keep byte accounting platform independent: text-mode newline
        # translation would otherwise add CR bytes on Windows.
        self._events = self.events_path.open("a", encoding="utf-8", newline="\n", buffering=1)

        self._frame_queue = None
        self._worker = None
        self._frame_active = False
        if self._record_frames:
            self._frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
            self._worker = threading.Thread(
                target=self._write_frames, name="diagnostics-frame-writer", daemon=True
            )
            self._worker.start()

    @property
    def counters(self):
        """A snapshot for tests and callers that need current drop counts."""
        with self._lock:
            return dict(self._counters)

    def new_action_id(self):
        with self._lock:
            action_id = f"action-{self._next_action_id:06d}"
            self._next_action_id += 1
            return action_id

    def new_match_id(self, boundary_source=None):
        with self._lock:
            if (boundary_source == "deploy_gap" and self._pending_first_deploy_boundary
                    and self._phase == "deploy" and self._match_id is not None):
                self._last_boundary_deduplicated = True
                self._pending_first_deploy_boundary = False
                self._boundary_source = "deploy_gap"
                return self._match_id
            match_id = f"match-{self._next_match_id:03d}"
            self._next_match_id += 1
            self._match_id = match_id
            self._round_index = 1
            preserve_deploy = boundary_source == "deploy_gap" and self._phase == "deploy"
            if not preserve_deploy:
                self._phase = None
            self._boundary_source = boundary_source
            self._last_boundary_deduplicated = False
            self._pending_first_deploy_boundary = boundary_source in ("search_tap", "rematch_tap")
            return match_id

    def consume_pending_first_deploy_boundary(self):
        with self._lock:
            self._pending_first_deploy_boundary = False

    def observe_phase(self, phase, *, frame_id=None):
        with self._lock:
            previous = self._phase
            if phase == "deploy" and previous == "battle" and self._match_id is not None:
                self._round_index = (self._round_index or 1) + 1
            if phase in ("deploy", "battle"):
                self._counters["phase_observations"] += 1
                if phase != previous:
                    self._counters["phase_transitions"] += 1
            round_index = self._round_index
            match_id = self._match_id
        if phase in ("deploy", "battle"):
            self.event("phase", frame_id=frame_id, match_id=match_id,
                       round_index=round_index, phase=phase, source="BoardReader.phase",
                       data={"phase": phase, "transition_from": previous,
                             "round_index_certainty": "assumed" if match_id else "unknown",
                             "round_index_basis": "start_run_then_observed_transitions" if match_id else None})
        else:
            self.event("phase_unknown", frame_id=frame_id, match_id=match_id,
                       round_index=round_index, phase=None, source="BoardReader.phase",
                       data={"phase": None, "certainty": "unknown"}, inherit_context=False)

    def frame_id(self, frame):
        if frame is None:
            return None
        self._observe_frame_size(frame)
        frame_id, _ = self._identify_frame(frame)
        return frame_id

    def record_frame(self, frame, *, reason):
        if frame is None:
            return None
        self._observe_frame_size(frame)
        frame_id, _ = self._identify_frame(frame)
        if not self._claim_frame_record(frame, frame_id):
            with self._lock:
                self._counters["frames_reused"] += 1
            self.event("frame_reused", frame_id=frame_id,
                       data={"reason": reason, "certainty": "observed"}, source="diagnostics")
            return frame_id

        expected_path = Path("frames") / f"{frame_id:06d}.png" if self._record_frames else None
        self.event("frame_seen", frame_id=frame_id,
                   data={"reason": reason, "record_requested": self._record_frames,
                         "path": expected_path.as_posix() if expected_path is not None else None},
                   source="diagnostics")
        if not self._record_frames:
            return frame_id

        with self._lock:
            if self._size_limited or self._closed or self._closing:
                self._counters["frames_dropped"] += 1
                self._truncated = True
                return frame_id

        try:
            frozen = frame.copy()
            self._frame_queue.put_nowait((frame_id, frozen, reason))
        except queue.Full:
            with self._lock:
                self._counters["frames_dropped"] += 1
                self._truncated = True
            self.event("frame_dropped", frame_id=frame_id,
                       data={"reason": "queue_full", "source_reason": reason}, source="diagnostics")
        except Exception as exc:
            self._record_error("frame_copy", exc)
        return frame_id

    def _observe_frame_size(self, frame):
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            return
        try:
            size = {"width": int(shape[1]), "height": int(shape[0])}
        except (TypeError, ValueError, IndexError):
            return
        with self._lock:
            if self._frame_size is not None:
                return
            temporary = None
            try:
                old_size = self.metadata_path.stat().st_size
                self._metadata_record["frame_size"] = size
                encoded = self._json_bytes(self._metadata_record)
                delta = len(encoded) - old_size
                if self._bytes_written + self._reserved_bytes + delta + SUMMARY_RESERVE_BYTES > self._max_bytes:
                    self._metadata_record["frame_size"] = None
                    self._truncated = True
                    return
                temporary = self.metadata_path.with_name(".metadata.json.tmp")
                temporary.write_bytes(encoded)
                temporary.replace(self.metadata_path)
                self._bytes_written += delta
                self._frame_size = size
            except Exception as exc:
                self._metadata_record["frame_size"] = None
                self._record_error_locked("metadata_update", exc)
            finally:
                if temporary is not None and temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass

    def event(self, event_type, *, data=None, frame_id=None, action_id=None,
              match_id=None, round_index=None, phase=_UNSET, source=None,
              inherit_context=True):
        with self._lock:
            if self._closed:
                return False
            if inherit_context:
                if match_id is not None:
                    self._match_id = match_id
                if round_index is not None:
                    self._round_index = round_index
                if phase is not _UNSET:
                    self._phase = phase
                record_match_id = match_id if match_id is not None else self._match_id
                record_round_index = round_index if round_index is not None else self._round_index
                record_phase = self._phase if phase is _UNSET else phase
            else:
                record_match_id = match_id
                record_round_index = round_index
                record_phase = None if phase is _UNSET else phase

            record = {
                "schema_version": SCHEMA_VERSION,
                "event_id": self._next_event_id,
                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "elapsed_ms": int((time.monotonic() - self._started_mono) * 1000),
                "session_id": self.session_id,
                "match_id": record_match_id,
                "round_index": record_round_index,
                "phase": record_phase,
                "event_type": event_type,
                "frame_id": frame_id,
                "action_id": action_id,
                "source": source,
                "data": data or {},
            }
            try:
                if self._size_limited:
                    self._counters["events_dropped"] += 1
                    return False
                line = self._json_bytes(record) + b"\n"
                if self._bytes_written + self._reserved_bytes + len(line) + SUMMARY_RESERVE_BYTES > self._max_bytes:
                    self._truncated = True
                    self._size_limited = True
                    self._counters["events_dropped"] += 1
                    return False
                self._events.write(line.decode("utf-8"))
                self._events.flush()
                self._bytes_written += len(line)
                self._next_event_id += 1
                self._counters["events_written"] += 1
                self._event_types[event_type] += 1
                self._count_event(event_type, data or {})
                if event_type == "decision" and (data or {}).get("reason"):
                    self._decision_reasons[(data or {})["reason"]] += 1
                return True
            except Exception as exc:
                self._record_error_locked("event_write", exc)
                return False

    def close(self, reason):
        if self._closed:
            return
        with self._lock:
            self._closing = True
        if self._worker is not None:
            self._worker.join(timeout=CLOSE_TIMEOUT_SECONDS)
            if self._worker.is_alive():
                with self._frame_queue.mutex:
                    pending_frames = self._frame_queue.unfinished_tasks
                with self._lock:
                    self._counters["frames_dropped"] += pending_frames
                    self._counters["diagnostic_errors"] += 1
                    self._truncated = True
                    self._warn_once("frame writer did not stop before the shutdown timeout")
        with self._lock:
            self._closed = True
            try:
                self._events.flush()
                self._events.close()
            except Exception as exc:
                self._record_error_locked("event_close", exc)
            try:
                self._write_summary(reason)
            except Exception as exc:
                self._record_error_locked("summary_write", exc)

    def _identify_frame(self, frame):
        key = id(frame)
        with self._lock:
            old = self._frame_refs.get(key)
            if old is not None and old[0]() is frame:
                return old[1], False
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            try:
                ref = weakref.ref(frame, lambda ref, key=key: self._forget_frame(key, ref))
                self._frame_refs[key] = (ref, frame_id, False)
            except TypeError:
                pass
            self._counters["frames_seen"] += 1
            return frame_id, True

    def _claim_frame_record(self, frame, frame_id):
        key = id(frame)
        with self._lock:
            current = self._frame_refs.get(key)
            if current is None or current[0]() is not frame or current[1] != frame_id:
                return True
            if current[2]:
                return False
            self._frame_refs[key] = (current[0], current[1], True)
            return True

    def _forget_frame(self, key, ref):
        with self._lock:
            current = self._frame_refs.get(key)
            if current is not None and current[0] is ref:
                self._frame_refs.pop(key, None)

    def _write_frames(self):
        while True:
            try:
                item = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                with self._lock:
                    if self._closing:
                        return
                continue
            frame_id, frame, reason = item
            with self._lock:
                if self._closed or self._size_limited:
                    if self._size_limited:
                        self._counters["frames_dropped"] += 1
                        self._truncated = True
                    self._frame_queue.task_done()
                    continue
                self._frame_active = True
            temporary = None
            reservation = 0
            reservation_held = False
            try:
                ok, encoded = cv2.imencode(".png", frame)
                if not ok:
                    raise OSError("OpenCV could not encode frame")
                payload = encoded.tobytes()
                path = Path("frames") / f"{frame_id:06d}.png"
                reservation = len(payload) + 512
                with self._lock:
                    if self._closed:
                        continue
                    if self._size_limited:
                        self._counters["frames_dropped"] += 1
                        self._truncated = True
                        continue
                    if (self._bytes_written + self._reserved_bytes + reservation
                            + SUMMARY_RESERVE_BYTES > self._max_bytes):
                        self._counters["frames_dropped"] += 1
                        self._truncated = True
                        self._size_limited = True
                        continue
                    self._reserved_bytes += reservation
                    reservation_held = True
                target = self.session_dir / path
                temporary = target.with_name(f".{target.name}.tmp")
                temporary.write_bytes(payload)
                with self._lock:
                    if self._closed:
                        self._reserved_bytes -= reservation
                        reservation_held = False
                        continue
                temporary.replace(target)
                with self._lock:
                    self._reserved_bytes -= reservation
                    reservation_held = False
                    if self._closed:
                        discard = True
                    else:
                        discard = False
                        self._bytes_written += len(payload)
                        self._counters["frames_saved"] += 1
                if discard:
                    try:
                        target.unlink()
                    except OSError:
                        pass
            except Exception as exc:
                with self._lock:
                    if reservation_held:
                        self._reserved_bytes -= reservation
                    self._counters["frames_dropped"] += 1
                    self._record_error_locked("frame_write", exc)
            finally:
                if temporary is not None and temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
                with self._lock:
                    self._frame_active = False
                self._frame_queue.task_done()

    def _record_error(self, operation, exc):
        with self._lock:
            self._record_error_locked(operation, exc)

    def _record_error_locked(self, operation, exc):
        self._counters["diagnostic_errors"] += 1
        self._warn_once(f"{operation} failed: {exc}")

    def _warn_once(self, message):
        if not self._warned:
            self._warned = True
            logger.warning("[DIAGNOSTICS] %s", message)

    def _count_event(self, event_type, data):
        if event_type == "match_boundary_assumed" and not data.get("deduplicated"):
            self._counters["matches_detected"] += 1
        elif event_type == "phase":
            if data.get("phase") == "deploy":
                self._counters["deploy_phases"] += 1
            elif data.get("phase") == "battle":
                self._counters["battle_phases"] += 1
        elif event_type == "decision":
            self._counters["decisions"] += 1
            self._counters["decision_duration_ms_total"] += int(data.get("decision_duration_ms", 0) or 0)
        elif event_type == "buy_attempt":
            self._counters["buy_attempts"] += 1
        elif event_type == "buy_outcome":
            classification = data.get("legacy_classification")
            if classification == "accepted_by_legacy_logic":
                self._counters["strategy_accounted_purchases"] += 1
                if data.get("confirmation") == "assumed":
                    self._counters["assumed_successful_purchases"] += 1
            elif classification == "no_effect_suspected":
                self._counters["suspected_failed_purchases"] += 1
        elif event_type == "sale_attempt_started":
            self._counters["sale_attempts"] += 1
        elif event_type == "sale_button_tap_attempt":
            self._counters["sale_button_taps"] += 1
        elif event_type == "calculated_upgrade" and data.get("threshold_reached"):
            self._counters["calculated_upgrades"] += 1
        elif event_type == "orb_tap_attempt":
            self._counters["orb_taps"] += 1
        elif event_type == "action_verification_result":
            status = data.get("status")
            counter = {"CONFIRMED": "confirmed_actions", "FAILED": "failed_actions",
                       "UNKNOWN": "unknown_actions", "INTERRUPTED": "interrupted_actions"}.get(status)
            if counter:
                self._counters[counter] += 1
        elif event_type == "purchase_refused":
            if data.get("reason") == "unknown_elixir":
                self._counters["elixir_errors"] += 1
            elif data.get("reason") == "no_cards":
                self._counters["no_card_observations"] += 1

    def _write_summary(self, reason):
        file_sizes = {}
        for path in self.session_dir.rglob("*"):
            if path.is_file() and path != self.summary_path and not path.name.endswith(".tmp"):
                file_sizes[path.relative_to(self.session_dir).as_posix()] = path.stat().st_size
        total_before = sum(file_sizes.values())
        summary = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "duration_ms": int((time.monotonic() - self._started_mono) * 1000),
            "finish_reason": reason,
            "counters": dict(self._counters),
            "event_type_counts": dict(self._event_types),
            "decision_reasons": dict(self._decision_reasons),
            "average_decision_duration_ms": (
                self._counters["decision_duration_ms_total"] / self._counters["decisions"]
                if self._counters["decisions"] else None
            ),
            "file_sizes": file_sizes,
            "truncated": self._truncated,
        }
        for _ in range(5):
            raw = self._json_bytes(summary)
            summary["file_sizes"]["summary.json"] = len(raw)
            summary["total_bytes"] = total_before + len(raw)
            if len(self._json_bytes(summary)) == len(raw):
                break
        raw = self._json_bytes(summary)
        if total_before + len(raw) > self._max_bytes:
            summary = {"schema_version": SCHEMA_VERSION, "session_id": self.session_id,
                       "finish_reason": reason, "truncated": True,
                       "counters": dict(self._counters), "total_bytes": total_before}
            for _ in range(3):
                raw = self._json_bytes(summary)
                summary["total_bytes"] = total_before + len(raw)
                if len(self._json_bytes(summary)) == len(raw):
                    break
            raw = self._json_bytes(summary)
            self._truncated = True
        self.summary_path.write_bytes(raw)

    @staticmethod
    def _json_bytes(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
