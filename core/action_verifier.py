"""Small, observation based verification for game actions."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import time


class ActionStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class Observation:
    phase: str | None = None
    elixir: int | None = None
    shop_cards: tuple = ()
    selected_slot: int | None = None


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
                        before, after, cost, started_at=None, rechecks=0):
        started_at = time.monotonic() if started_at is None else started_at
        evidence = []
        elapsed = max(0, round((time.monotonic() - started_at) * 1000))
        if before.phase != "deploy" or after.phase != "deploy":
            status, reason = ActionStatus.INTERRUPTED, "deploy phase changed or was not observed"
        elif adb_status != "succeeded":
            status, reason = ActionStatus.FAILED, f"ADB command {adb_status}"
        elif before.elixir is None or after.elixir is None:
            status, reason = ActionStatus.UNKNOWN, "elixir is not visually recognized before and after"
        else:
            delta = before.elixir - after.elixir
            elixir_matches = delta == cost
            shop_changed = bool(before.shop_cards and after.shop_cards and
                                before.shop_cards != after.shop_cards)
            slot_changed = (before.selected_slot is not None and after.selected_slot is not None
                            and before.selected_slot != after.selected_slot)
            if elixir_matches:
                evidence.append("elixir_delta_matches_cost")
            if shop_changed:
                evidence.append("shop_assortment_changed")
            if slot_changed:
                evidence.append("selected_slot_changed")
            if elixir_matches and (shop_changed or slot_changed):
                status, reason = ActionStatus.CONFIRMED, "elixir delta and shop selection evidence agree"
            elif delta == 0 and not shop_changed and not slot_changed:
                status, reason = ActionStatus.FAILED, "no visible purchase effect"
            elif delta != cost:
                status, reason = ActionStatus.UNKNOWN, "observed elixir delta does not match purchase cost"
            else:
                status, reason = ActionStatus.UNKNOWN, "purchase lacks an independent shop change"
        return ActionResult(action_id, "purchase", status, adb_status, frame_before, frame_after,
                            before, after, evidence, reason, elapsed, rechecks)
