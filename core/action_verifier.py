"""Small, observation based verification for game actions."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import time


class ActionStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    INTERRUPTED = "INTERRUPTED"


class VerificationBudget:
    """One monotonic time budget shared by an action and its rechecks."""

    def __init__(self, seconds):
        self.deadline = time.monotonic() + max(0.0, float(seconds))

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())


@dataclass(frozen=True)
class Observation:
    phase: str | None = None
    elixir: int | None = None
    shop_cards: tuple | None = None
    selected_slot: tuple | None = None
    own_units: int | None = None


@dataclass
class ActionResult:
    action_id: str
    action_type: str
    status: ActionStatus
    adb_status: str
    frame_before: int | None
    frame_after: int | None
    observation_before: Observation
    observation_after: Observation
    evidence: list[str] = field(default_factory=list)
    reason: str = ""
    duration_ms: int = 0
    rechecks: int = 0

    def as_dict(self):
        result = asdict(self)
        result["status"] = self.status.value
        return result


class ActionVerifier:
    """Classify a purchase from independent visible state changes."""

    @staticmethod
    def verify_purchase(*, action_id, adb_status, frame_before, frame_after,
                        before, after, cost, started_at=None, rechecks=0,
                        allow_unit_count=True):
        started_at = time.monotonic() if started_at is None else started_at
        evidence = []
        elapsed = max(0, round((time.monotonic() - started_at) * 1000))
        if before.phase != "deploy" or after.phase != "deploy":
            status, reason = ActionStatus.INTERRUPTED, "deploy phase changed or was not observed"
        elif before.elixir is None or after.elixir is None:
            status, reason = ActionStatus.UNKNOWN, "elixir is not visually recognized before and after"
        else:
            delta = before.elixir - after.elixir
            elixir_matches = delta == cost
            shop_known = before.shop_cards is not None and after.shop_cards is not None
            shop_changed = shop_known and before.shop_cards != after.shop_cards
            slot_known = before.selected_slot is not None and after.selected_slot is not None
            slot_changed = slot_known and before.selected_slot[0] == after.selected_slot[0] and \
                before.selected_slot[1] != after.selected_slot[1]
            units_increased = (allow_unit_count and before.own_units is not None and after.own_units is not None
                               and after.own_units > before.own_units)
            if elixir_matches:
                evidence.append("elixir_delta_matches_cost")
            if shop_changed:
                evidence.append("shop_assortment_changed")
            if slot_changed:
                evidence.append("selected_slot_changed")
            if units_increased:
                evidence.append("own_unit_count_increased")
            independent_change = shop_changed or slot_changed or units_increased
            if elixir_matches and independent_change:
                status, reason = ActionStatus.CONFIRMED, "expected elixir spend and independent game-state change agree"
            elif adb_status != "succeeded":
                status, reason = ActionStatus.UNKNOWN, f"ADB command {adb_status}; game effect is uncertain"
            elif (delta == 0 and shop_known and not shop_changed and slot_known
                  and not slot_changed and not units_increased):
                status, reason = ActionStatus.FAILED, "no visible purchase effect"
            elif delta != cost:
                status, reason = ActionStatus.UNKNOWN, "observed elixir delta does not match purchase cost"
            else:
                status, reason = ActionStatus.UNKNOWN, "purchase lacks an independent shop change"
        return ActionResult(action_id, "purchase", status, adb_status, frame_before, frame_after,
                            before, after, evidence, reason, elapsed, rechecks)
