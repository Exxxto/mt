"""
Core game logic for the Merge Tactics bot.

A deploy phase is the only moment with real decisions: pop the quest orbs, then
spend the elixir on the shop while preferring duplicates (a duplicate merges and
refunds one elixir). The vision side lives in BoardReader, the decision rules in
DeployPlanner, and this class keeps the per-match state.
"""

import logging

from core.board_state import BoardReader
from core.strategy import DeployPlanner, RunState

logger = logging.getLogger(__name__)


class GameLogic:
    def __init__(self, adb_controller, image_processor, diagnostics=None, verify_actions=False):
        self.adb = adb_controller
        self.image_processor = image_processor
        self.diagnostics = diagnostics
        self.reader = (BoardReader(image_processor, include_diagnostics=bool(diagnostics and diagnostics.enabled))
                       if diagnostics is not None else BoardReader(image_processor))
        self.planner = (DeployPlanner(self.reader, self.adb, diagnostics=diagnostics)
                        if diagnostics is not None else DeployPlanner(self.reader, self.adb))
        self.planner.verify_actions = verify_actions
        self.run_state = RunState()
        self._pending_first_deploy = False

    def start_run(self, boundary_source="unspecified"):
        """Forget the troops of the previous match"""
        diagnostics = getattr(self, "diagnostics", None)
        if boundary_source == "deploy_gap" and getattr(self, "_pending_first_deploy", False):
            self._pending_first_deploy = False
            if diagnostics and diagnostics.enabled:
                diagnostics.new_match_id(boundary_source=boundary_source)
                diagnostics.event(
                    "match_boundary_prevented_reset", source="GameLogic.start_run",
                    data={"boundary_source": boundary_source, "run_state_cleared": False},
                )
            return False

        self.run_state = RunState()
        if diagnostics and diagnostics.enabled:
            match_id = diagnostics.new_match_id(boundary_source=boundary_source)
            diagnostics.event(
                "match_boundary_assumed", match_id=match_id, round_index=1,
                source="GameLogic.start_run",
                data={"boundary_source": boundary_source, "certainty": "assumed",
                      "run_state_cleared": True,
                      "deduplicated": diagnostics._last_boundary_deduplicated},
            )
        self._pending_first_deploy = boundary_source in ("search_tap", "rematch_tap")
        logger.info("[RUN] new match state")
        return True

    def deploy_phase(self, screen):
        """Play one deploy phase and return a summary of what happened"""
        return self.planner.play(screen, self.run_state)

    def consume_pending_first_deploy_boundary(self):
        self._pending_first_deploy = False
        diagnostics = getattr(self, "diagnostics", None)
        if diagnostics and diagnostics.enabled:
            diagnostics.consume_pending_first_deploy_boundary()

    def phase(self, screen):
        return self.reader.phase(screen)
