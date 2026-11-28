"""
Headless Merge Tactics bot.

Run it on the game lobby: it presses "Batalla" to search for a match, plays every
round (pops orbs, buys/merges toward the season target comp) and rematches on the
results screen, forever. No GUI.

    python bot.py                 # search for a match and play non-stop
    python bot.py --no-search     # already in a match: just play it
    python bot.py --duration 1800 # stop after 30 minutes
"""

import argparse
import math
import platform
import subprocess
import time

import cv2
import numpy as np

from core.adb_controller import ADBController
from core.image_processing import ImageProcessor
from core.game_logic import GameLogic
from core.diagnostics import DiagnosticsSession
from config.paths import ADB_PATH
from config.board_layout import (
    END_BUTTONS,
    END_BUTTON_THRESHOLD,
    SEARCH_BUTTON,
    SEARCH_THRESHOLD,
)
from config.constants import ERROR_DELAY, BATTLE_WAIT
from config.target_comp import TARGET_COMP, SEASON
from config.logging_config import setup_logging

logger = setup_logging()

# Re-press "Batalla" at most this often while no match is running
SEARCH_RETRY_SECONDS = 8.0


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_seconds(value):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be positive seconds") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive seconds")
    return parsed


def _git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=2, check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _diagnostics_metadata(args):
    return {
        "strategy": "verify_actions" if getattr(args, "verify_actions", False) else "legacy",
        "git_revision": _git_revision(),
        "runtime": {
            "python": platform.python_version(),
            "opencv": getattr(cv2, "__version__", None),
            "numpy": getattr(np, "__version__", None),
        },
        "frame_size": None,
        "parameters": {
            "diagnostics": bool(args.diagnostics or args.record_frames or getattr(args, "observe_bench", False)),
            "observe_bench": getattr(args, "observe_bench", False),
            "record_frames": bool(args.record_frames),
            "auto_search": not args.no_search,
            "duration_seconds": args.duration,
            "diagnostics_max_mb": args.diagnostics_max_mb,
            "verify_actions": getattr(args, "verify_actions", False),
            "max_frame_age_seconds": getattr(args, "max_frame_age", 3.0),
        },
    }


def diagnostics_enabled(bot):
    diagnostics = getattr(bot, "diagnostics", None)
    return bool(diagnostics and diagnostics.enabled)


class MergeTacticsBot:
    def __init__(self, diagnostics=None, verify_actions=False, max_frame_age_seconds=3.0, observe_bench=False):
        self.diagnostics = diagnostics
        self.adb = ADBController(ADB_PATH)
        self.image_processor = ImageProcessor()
        self.game_logic = GameLogic(self.adb, self.image_processor, diagnostics=diagnostics,
                                    verify_actions=verify_actions,
                                    max_frame_age_seconds=max_frame_age_seconds)
        if observe_bench and diagnostics and diagnostics.enabled:
            from core.bench_reader import BenchObserver
            self.game_logic.planner.bench_observer = BenchObserver(diagnostics)
        self.verify_actions = verify_actions
        self.max_frame_age_seconds = max_frame_age_seconds
        self._current_frame_captured_at = None
        self.last_elixir_value = 0
        self._deploy_played = False
        self._last_deploy_ts = 0.0
        logger.info(f"[BOT] season {SEASON} | target comp: {', '.join(TARGET_COMP[:8])} ...")

    def search_match(self, screen):
        """Press the lobby 'Batalla' button to start a match"""
        pos = self.image_processor.find_template(screen, SEARCH_BUTTON, SEARCH_THRESHOLD)
        if not pos:
            return False
        if self._record_action_age("match_search", screen):
            self._record_stale_tap("match_search", screen)
            return False
        logger.info(f"[SEARCH] pressing Batalla at {pos}")
        action_id = None
        if diagnostics_enabled(self):
            frame_id = self.diagnostics.record_frame(screen, reason="match_search")
            action_id = self.diagnostics.new_action_id()
            self.diagnostics.event("match_search_attempt", frame_id=frame_id, action_id=action_id,
                                   source="MergeTacticsBot.search_match",
                                   data={"button": "search", "coordinates": list(pos), "status": "attempted"})
        if diagnostics_enabled(self):
            try:
                self.adb.tap(*pos)
            except Exception as exc:
                self.diagnostics.event("adb_error", frame_id=frame_id, action_id=action_id,
                                       source="MergeTacticsBot.search_match",
                                       data={"operation": "search_tap", "error_type": type(exc).__name__,
                                             "certainty": "observed"})
                raise
        else:
            self.adb.tap(*pos)
        if diagnostics_enabled(self):
            self.diagnostics.event("match_search_tap_returned", frame_id=frame_id, action_id=action_id,
                                   source="MergeTacticsBot.search_match",
                                   data={"adb_call_returned": True, "device_confirmation": "unknown"})
        self._deploy_played = False
        self._last_deploy_ts = 0.0
        if diagnostics_enabled(self):
            self.game_logic.start_run(boundary_source="search_tap")
        else:
            self.game_logic.start_run()
        time.sleep(2.5)
        return True

    def handle_end_screen(self, screen):
        """Leave the results/elimination screens and start the next match"""
        for name in ("rematch", "view", "leave"):
            pos = self.image_processor.find_template(screen, END_BUTTONS[name], END_BUTTON_THRESHOLD)
            if not pos:
                continue
            if self._record_action_age("end_screen", screen):
                self._record_stale_tap("end_screen", screen)
                return False
            logger.info(f"[END] tapping {name} at {pos}")
            action_id = None
            if diagnostics_enabled(self):
                frame_id = self.diagnostics.record_frame(screen, reason="match_end_screen")
                action_id = self.diagnostics.new_action_id()
                self.diagnostics.event("match_end_screen", frame_id=frame_id, action_id=action_id,
                                       source="MergeTacticsBot.handle_end_screen",
                                       data={"button": name, "coordinates": list(pos),
                                             "certainty": "observed"})
                self.diagnostics.event("end_screen_action_attempt", frame_id=frame_id, action_id=action_id,
                                       source="MergeTacticsBot.handle_end_screen",
                                       data={"button": name, "coordinates": list(pos), "status": "attempted"})
            if diagnostics_enabled(self):
                try:
                    self.adb.tap(*pos)
                except Exception as exc:
                    self.diagnostics.event("adb_error", frame_id=frame_id, action_id=action_id,
                                           source="MergeTacticsBot.handle_end_screen",
                                           data={"operation": "end_screen_tap", "error_type": type(exc).__name__,
                                                 "certainty": "observed"})
                    raise
            else:
                self.adb.tap(*pos)
            if diagnostics_enabled(self):
                self.diagnostics.event("end_screen_action_returned", frame_id=frame_id, action_id=action_id,
                                       source="MergeTacticsBot.handle_end_screen",
                                       data={"adb_call_returned": True, "device_confirmation": "unknown"})
            if name == "rematch":
                self._deploy_played = False
                if diagnostics_enabled(self):
                    self.game_logic.start_run(boundary_source="rematch_tap")
                else:
                    self.game_logic.start_run()
            time.sleep(1.0)
            return True
        return False

    def handle_deploy_phase(self, screen, captured_at=None):
        """Pop the orbs, then buy and merge; returns the planner summary"""
        now = time.time()
        # Rounds are ~25s apart; a much longer gap means a new match started.
        if now - self._last_deploy_ts > 90:
            if diagnostics_enabled(self):
                self.game_logic.start_run(boundary_source="deploy_gap")
            else:
                self.game_logic.start_run()
        elif diagnostics_enabled(self):
            self.game_logic.consume_pending_first_deploy_boundary()
        self._last_deploy_ts = now

        summary = (self.game_logic.deploy_phase(screen) if captured_at is None
                   else self.game_logic.deploy_phase(screen, captured_at=captured_at))
        if summary["elixir"] is not None:
            self.last_elixir_value = summary["elixir"]
        return summary

    def collect_orbs(self, screen, captured_at=None):
        """Tap every quest orb on the arena; they hand out money"""
        reader = self.game_logic.reader
        detect_orbs = getattr(reader, "orb_detections", None)
        if callable(detect_orbs):
            candidates = detect_orbs(screen)
        else:
            candidates = [{"center": point, "bounds": None, "fill": None}
                          for point in reader.orbs(screen)]
        captured_at = (getattr(self, "_current_frame_captured_at", None) if captured_at is None
                       else captured_at)
        frame_id = self.diagnostics.frame_id(screen) if diagnostics_enabled(self) else None
        if diagnostics_enabled(self):
            self.diagnostics.event("orb_detection", frame_id=frame_id, source="MergeTacticsBot.collect_orbs",
                                   data={"coordinates": [list(item["center"]) for item in candidates],
                                         "count": len(candidates), "certainty": "candidate",
                                         "call_source": "main_loop"})
        if not candidates:
            return screen
        if getattr(self, "verify_actions", False) and self._record_action_age("orb", screen, captured_at):
            if diagnostics_enabled(self):
                for item in candidates:
                    self.diagnostics.event("orb_candidate_rejected", frame_id=frame_id,
                                           source="MergeTacticsBot.collect_orbs",
                                           data={"coordinates": list(item["center"]),
                                                 "reason": "stale_frame"})
            try:
                fresh = self.adb.screenshot()
            except Exception as exc:
                logger.warning("[ORB] could not refresh stale observation: %s", type(exc).__name__)
                return None
            self._current_frame_captured_at = time.monotonic()
            if diagnostics_enabled(self):
                self.diagnostics.record_frame(fresh, reason="stale_orb_refresh")
            return fresh

        # A real collectible stays in place across captures; combat flashes and
        # unit highlights change shape too quickly to pass this second look.
        try:
            confirmation = self.adb.screenshot()
        except Exception as exc:
            logger.warning("[ORB] could not confirm candidate: %s", type(exc).__name__)
            return screen
        confirmation_at = time.monotonic()
        confirmation_id = self.diagnostics.frame_id(confirmation) if diagnostics_enabled(self) else None
        if diagnostics_enabled(self):
            self.diagnostics.record_frame(confirmation, reason="orb_candidate_confirmation")
        if callable(detect_orbs):
            observed = detect_orbs(confirmation)
        else:
            observed = [{"center": point, "bounds": None, "fill": None}
                        for point in reader.orbs(confirmation)]
        if diagnostics_enabled(self):
            self.diagnostics.event("orb_confirmation_scan", frame_id=confirmation_id,
                                   source="MergeTacticsBot.collect_orbs",
                                   data={"candidates": [{"coordinates": list(item["center"]),
                                                         "bounds": item["bounds"]}
                                                        for item in observed],
                                         "count": len(observed),
                                         "certainty": "candidate"})
        recognized = []
        for candidate in candidates:
            x, y = candidate["center"]
            match = next((item for item in observed
                          if max(abs(item["center"][0] - x), abs(item["center"][1] - y)) <= 18
                          and (candidate["bounds"] is None or item["bounds"] is None
                               or all(0.75 <= a / b <= 1.33
                                      for a, b in zip(item["bounds"][2:], candidate["bounds"][2:])))), None)
            if match is None:
                if diagnostics_enabled(self):
                    self.diagnostics.event("orb_candidate_rejected", frame_id=frame_id,
                                           source="MergeTacticsBot.collect_orbs",
                                           data={"coordinates": [x, y],
                                                 "reason": "not_stable_across_captures"})
                continue
            recognized.append(match)
            if diagnostics_enabled(self):
                self.diagnostics.event("orb_recognized", frame_id=confirmation_id,
                                       source="MergeTacticsBot.collect_orbs",
                                       data={"coordinates": list(match["center"]),
                                             "method": "hsv_geometry_persistence",
                                             "certainty": "recognized"})

        if not recognized:
            self._current_frame_captured_at = confirmation_at
            return confirmation

        tapped = False
        for item in recognized:
            x, y = item["center"]
            if getattr(self, "verify_actions", False) and self._record_action_age(
                    "orb", confirmation, confirmation_at):
                self._record_stale_tap("orb", confirmation)
                try:
                    fresh = self.adb.screenshot()
                except Exception as exc:
                    logger.warning("[ORB] could not refresh stale observation: %s", type(exc).__name__)
                    return None
                self._current_frame_captured_at = time.monotonic()
                if diagnostics_enabled(self):
                    self.diagnostics.record_frame(fresh, reason="stale_orb_refresh")
                return fresh
            logger.info(f"[ORB] tap attempt at ({x}, {y})")
            action_id = None
            if diagnostics_enabled(self):
                action_id = self.diagnostics.new_action_id()
                self.diagnostics.event("orb_tap_attempt", frame_id=confirmation_id, action_id=action_id,
                                       source="MergeTacticsBot.collect_orbs",
                                       data={"coordinates": [x, y], "status": "attempted",
                                             "call_source": "main_loop"})
            if diagnostics_enabled(self):
                try:
                    self.adb.tap(x, y)
                except Exception as exc:
                    self.diagnostics.event("adb_error", frame_id=confirmation_id, action_id=action_id,
                                           source="MergeTacticsBot.collect_orbs",
                                           data={"operation": "orb_tap", "error_type": type(exc).__name__,
                                                 "coordinates": [x, y], "certainty": "observed"})
                    raise
            else:
                self.adb.tap(x, y)
            if diagnostics_enabled(self):
                self.diagnostics.event("orb_tap_returned", frame_id=confirmation_id, action_id=action_id,
                                       source="MergeTacticsBot.collect_orbs",
                                       data={"adb_call_returned": True, "device_confirmation": "unknown"})
                self.diagnostics.event("orb_collection_result", frame_id=confirmation_id,
                                       action_id=action_id, source="MergeTacticsBot.collect_orbs",
                                       data={"status": "unknown",
                                             "reason": "no post-tap game-state confirmation"})
            time.sleep(0.15)
            tapped = True
        if tapped:
            try:
                fresh = self.adb.screenshot()
                self._current_frame_captured_at = time.monotonic()
            except Exception as exc:
                logger.warning("[ORB] could not refresh frame: %s", type(exc).__name__)
                if diagnostics_enabled(self):
                    self.diagnostics.event("screenshot_error", frame_id=frame_id,
                                           source="MergeTacticsBot.collect_orbs",
                                           data={"reason": "after_orb_taps",
                                                 "error_type": type(exc).__name__,
                                                 "certainty": "observed"})
                return None
            if diagnostics_enabled(self):
                self.diagnostics.record_frame(fresh, reason="after_orb_taps")
            return fresh
        return screen

    def run(self, auto_search=True, duration=None):
        deadline = time.time() + duration if duration else None
        last_search = 0.0
        logger.info("[BOT] running" + (" (auto search)" if auto_search else ""))
        if diagnostics_enabled(self):
            self.diagnostics.event("session_started", source="MergeTacticsBot.run",
                                   data={"auto_search": auto_search, "duration_seconds": duration,
                                         "strategy": "verify_actions" if getattr(self, "verify_actions", False) else "legacy",
                                         "max_frame_age_seconds": getattr(self, "max_frame_age_seconds", 3.0)})

        previous_iteration_at = None
        while True:
            iteration_at = time.monotonic()
            if previous_iteration_at is not None and diagnostics_enabled(self):
                pause_ms = round((iteration_at - previous_iteration_at) * 1000)
                if pause_ms >= 5000:
                    self.diagnostics.event("long_iteration_pause", source="MergeTacticsBot.run",
                                           data={"pause_ms": pause_ms,
                                                 "threshold_ms": 5000,
                                                 "certainty": "observed"})
            previous_iteration_at = iteration_at
            if deadline and time.time() >= deadline:
                logger.info("[BOT] duration reached, stopping")
                break

            try:
                screen = self.adb.screenshot()
                self._current_frame_captured_at = time.monotonic()
            except Exception as e:
                logger.error(f"[BOT] screenshot failed: {e}")
                if diagnostics_enabled(self):
                    self.diagnostics.event("screenshot_error", source="MergeTacticsBot.run",
                                           data={"error_type": type(e).__name__, "certainty": "observed"})
                time.sleep(ERROR_DELAY)
                continue

            if self.handle_end_screen(screen):
                last_search = time.time()
                continue

            phase = self.game_logic.phase(screen)
            if diagnostics_enabled(self):
                frame_id = self.diagnostics.frame_id(screen)
                self.diagnostics.observe_phase(phase, frame_id=frame_id)
                if phase in ("deploy", "battle"):
                    self.diagnostics.record_frame(screen, reason=f"phase_{phase}")
            # Orbs give money and can show up in any colour during deploy and
            # battle, so grab any on screen every frame.
            if phase in ("deploy", "battle"):
                refreshed = self.collect_orbs(screen, self._current_frame_captured_at)
                if refreshed is None:
                    # Never make phase or purchase decisions from the pre-tap frame.
                    time.sleep(ERROR_DELAY)
                    continue
                screen = refreshed
                phase = self.game_logic.phase(screen)
                if diagnostics_enabled(self):
                    self.diagnostics.observe_phase(phase, frame_id=self.diagnostics.frame_id(screen))
            if phase == "deploy":
                if not self._deploy_played:
                    summary = (self.handle_deploy_phase(screen, self._current_frame_captured_at)
                               if getattr(self, "verify_actions", False)
                               else self.handle_deploy_phase(screen))
                    # Only commit the round when the shop was actually read.
                    # "no_cards" means recognition failed (e.g. purchase
                    # animation), so retry on a later frame instead of skipping.
                    if (summary and summary.get("elixir") is not None
                            and summary.get("reason") not in ("no_cards", "stale_frame",
                                                              "verification_pending")):
                        self._deploy_played = True
                time.sleep(0.8)
            elif phase == "battle":
                self._deploy_played = False
                time.sleep(BATTLE_WAIT)
            else:
                # Lobby or unknown screen: search for a match (throttled).
                if auto_search and time.time() - last_search >= SEARCH_RETRY_SECONDS:
                    if self.search_match(screen):
                        last_search = time.time()
                time.sleep(2.0)

    def _record_stale_tap(self, action, screen):
        captured_at = getattr(self, "_current_frame_captured_at", None)
        age_ms = (max(0, round((time.monotonic() - captured_at) * 1000))
                  if captured_at is not None else None)
        logger.info("[ACTION] discarded stale %s observation (%sms)", action, age_ms)
        if diagnostics_enabled(self):
            self.diagnostics.event("action_discarded_stale",
                                   frame_id=self.diagnostics.frame_id(screen),
                                   source="MergeTacticsBot",
                                   data={"action": action, "frame_age_ms": age_ms,
                                         "max_frame_age_ms": round(getattr(self, "max_frame_age_seconds", 3.0) * 1000),
                                         "reason": "stale_observation"})

    def _record_action_age(self, action, screen, captured_at=None):
        if not getattr(self, "verify_actions", False):
            return False
        captured_at = (getattr(self, "_current_frame_captured_at", None)
                       if captured_at is None else captured_at)
        if captured_at is None:
            if diagnostics_enabled(self):
                self.diagnostics.event("action_frame_age", frame_id=self.diagnostics.frame_id(screen),
                                       source="MergeTacticsBot",
                                       data={"action": action, "frame_age_ms": None,
                                             "max_frame_age_ms": round(getattr(self, "max_frame_age_seconds", 3.0) * 1000),
                                             "stale": True, "reason": "capture_time_unavailable"})
            return True
        age_ms = max(0, round((time.monotonic() - captured_at) * 1000))
        max_age_ms = round(getattr(self, "max_frame_age_seconds", 3.0) * 1000)
        if diagnostics_enabled(self):
            self.diagnostics.event("action_frame_age", frame_id=self.diagnostics.frame_id(screen),
                                   source="MergeTacticsBot",
                                   data={"action": action, "frame_age_ms": age_ms,
                                         "max_frame_age_ms": max_age_ms,
                                         "stale": age_ms > max_age_ms})
        return age_ms > max_age_ms


def main():
    parser = argparse.ArgumentParser(description="Headless Merge Tactics bot")
    parser.add_argument("--no-search", action="store_true",
                        help="do not press Batalla; only play a match already started")
    parser.add_argument("--duration", type=float, default=None,
                        help="stop after this many seconds")
    parser.add_argument("--diagnostics", action="store_true",
                        help="write a structured diagnostics session")
    parser.add_argument("--record-frames", action="store_true",
                        help="also record sampled screenshots (implies --diagnostics)")
    parser.add_argument("--diagnostics-dir", default="recordings",
                        help="root directory for diagnostic sessions (default: recordings)")
    parser.add_argument("--diagnostics-max-mb", type=positive_int, default=512,
                        help="maximum size per diagnostics session in MiB (default: 512)")
    parser.add_argument("--observe-bench", action="store_true",
                        help="observe reserve slots and record diagnostics")
    parser.add_argument("--verify-actions", action="store_true",
                        help="require visual confirmation before updating purchase state")
    parser.add_argument("--max-frame-age", type=positive_seconds, default=3.0,
                        help="maximum observation age before re-capturing (seconds; default: 3)")
    args = parser.parse_args()

    diagnostics = None
    if args.diagnostics or args.record_frames or args.verify_actions or args.observe_bench:
        try:
            diagnostics = DiagnosticsSession.create(
                root=args.diagnostics_dir,
                metadata=_diagnostics_metadata(args),
                record_frames=args.record_frames,
                max_bytes=args.diagnostics_max_mb * 1024 * 1024,
            )
        except Exception as exc:
            logger.warning("[DIAGNOSTICS] could not start session: %s", exc)

    finish_reason = "normal"
    try:
        bot = (MergeTacticsBot(diagnostics=diagnostics, observe_bench=True,
                               verify_actions=args.verify_actions, max_frame_age_seconds=args.max_frame_age)
               if args.observe_bench and args.verify_actions else
               MergeTacticsBot(diagnostics=diagnostics, observe_bench=True)
               if args.observe_bench else
               MergeTacticsBot(diagnostics=diagnostics, verify_actions=True,
                               max_frame_age_seconds=args.max_frame_age)
               if args.verify_actions else
               MergeTacticsBot(diagnostics=diagnostics) if diagnostics is not None else MergeTacticsBot())
        bot.run(auto_search=not args.no_search, duration=args.duration)
    except KeyboardInterrupt:
        finish_reason = "keyboard_interrupt"
        logger.info("[BOT] interrupted by user")
    except Exception:
        finish_reason = "error"
        raise
    finally:
        if diagnostics is not None:
            try:
                diagnostics.close(finish_reason)
            except Exception as exc:
                logger.warning("[DIAGNOSTICS] could not close session: %s", exc)


if __name__ == "__main__":
    main()
