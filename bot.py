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
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
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
            "diagnostics": bool(args.diagnostics or args.record_frames),
            "record_frames": bool(args.record_frames),
            "auto_search": not args.no_search,
            "duration_seconds": args.duration,
            "diagnostics_max_mb": args.diagnostics_max_mb,
            "verify_actions": getattr(args, "verify_actions", False),
        },
    }


def diagnostics_enabled(bot):
    diagnostics = getattr(bot, "diagnostics", None)
    return bool(diagnostics and diagnostics.enabled)


class MergeTacticsBot:
    def __init__(self, diagnostics=None, verify_actions=False):
        self.diagnostics = diagnostics
        self.adb = ADBController(ADB_PATH)
        self.image_processor = ImageProcessor()
        self.game_logic = GameLogic(self.adb, self.image_processor, diagnostics=diagnostics,
                                    verify_actions=verify_actions)
        self.last_elixir_value = 0
        self._deploy_played = False
        self._last_deploy_ts = 0.0
        logger.info(f"[BOT] season {SEASON} | target comp: {', '.join(TARGET_COMP[:8])} ...")

    def search_match(self, screen):
        """Press the lobby 'Batalla' button to start a match"""
        pos = self.image_processor.find_template(screen, SEARCH_BUTTON, SEARCH_THRESHOLD)
        if not pos:
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

    def handle_deploy_phase(self, screen):
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

        summary = self.game_logic.deploy_phase(screen)
        if summary["elixir"] is not None:
            self.last_elixir_value = summary["elixir"]
        return summary

    def collect_orbs(self, screen):
        """Tap every quest orb on the arena; they hand out money"""
        orbs = self.game_logic.reader.orbs(screen)
        frame_id = self.diagnostics.frame_id(screen) if diagnostics_enabled(self) else None
        if diagnostics_enabled(self):
            self.diagnostics.event("orb_detection", frame_id=frame_id, source="MergeTacticsBot.collect_orbs",
                                   data={"coordinates": [list(point) for point in orbs],
                                         "count": len(orbs), "certainty": "observed",
                                         "call_source": "main_loop"})
        tapped = False
        for x, y in orbs:
            logger.info(f"[ORB] collect at ({x}, {y})")
            action_id = None
            if diagnostics_enabled(self):
                action_id = self.diagnostics.new_action_id()
                self.diagnostics.event("orb_tap_attempt", frame_id=frame_id, action_id=action_id,
                                       source="MergeTacticsBot.collect_orbs",
                                       data={"coordinates": [x, y], "status": "attempted",
                                             "call_source": "main_loop"})
            if diagnostics_enabled(self):
                try:
                    self.adb.tap(x, y)
                except Exception as exc:
                    self.diagnostics.event("adb_error", frame_id=frame_id, action_id=action_id,
                                           source="MergeTacticsBot.collect_orbs",
                                           data={"operation": "orb_tap", "error_type": type(exc).__name__,
                                                 "coordinates": [x, y], "certainty": "observed"})
                    raise
            else:
                self.adb.tap(x, y)
            if diagnostics_enabled(self):
                self.diagnostics.event("orb_tap_returned", frame_id=frame_id, action_id=action_id,
                                       source="MergeTacticsBot.collect_orbs",
                                       data={"adb_call_returned": True, "device_confirmation": "unknown"})
            time.sleep(0.15)
            tapped = True
        if tapped:
            fresh = self.adb.screenshot()
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
                                         "strategy": "legacy"})

        while True:
            if deadline and time.time() >= deadline:
                logger.info("[BOT] duration reached, stopping")
                break

            try:
                screen = self.adb.screenshot()
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
                refreshed = self.collect_orbs(screen)
                if refreshed is not None:
                    screen = refreshed
                    phase = self.game_logic.phase(screen)
                    if diagnostics_enabled(self):
                        self.diagnostics.observe_phase(phase, frame_id=self.diagnostics.frame_id(screen))
            if phase == "deploy":
                if not self._deploy_played:
                    summary = self.handle_deploy_phase(screen)
                    # Only commit the round when the shop was actually read.
                    # "no_cards" means recognition failed (e.g. purchase
                    # animation), so retry on a later frame instead of skipping.
                    if (summary and summary.get("elixir") is not None
                            and summary.get("reason") != "no_cards"):
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
    parser.add_argument("--verify-actions", action="store_true",
                        help="require visual confirmation before updating purchase state")
    args = parser.parse_args()

    diagnostics = None
    if args.diagnostics or args.record_frames or args.verify_actions:
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
        bot = (MergeTacticsBot(diagnostics=diagnostics, verify_actions=True)
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
