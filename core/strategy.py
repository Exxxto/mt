"""
Purchase planning for a Merge Tactics deploy phase.

What actually wins this mode is merging: buying a duplicate of a troop already
deployed merges it into the next star and refunds one elixir, so a duplicate is
almost always the best thing on the shelf. The shop does not reroll on its own;
a purchase replaces that slot with a new card, so the planner keeps buying - the
strongest target troop (see config/target_comp.py) or, if none is offered, the
cheapest card - to cycle the shop and find duplicates faster. When the bench is
full a purchase has no effect, so it sells one reserve unit and carries on.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field

from config.board_layout import (
    BENCH_CELLS,
    ELIXIR_RECT,
    SELL_BUTTON,
    SELL_THRESHOLD,
)
from config.target_comp import TARGET_COMP, target_rank
from config.board_layout import SHOP_CENTERS
from core.action_verifier import ActionStatus, ActionVerifier, Observation, VerificationBudget
from core.adb_controller import ADBActionResult

logger = logging.getLogger(__name__)
VERIFY_MAX_RECHECKS = 1
VERIFY_ACTION_TIMEOUT_SECONDS = 8.0

# Keeping the roster narrow is what makes merges reachable: every extra kind of
# troop spreads the same elixir over more copies.
MAX_DISTINCT_TROOPS = 6
# Star level thresholds: copies needed for 2, 3 and 4 stars
STAR_STEPS = (2, 4, 8)


@dataclass
class RunState:
    """Troops bought during the current match, and therefore their star level"""

    copies: dict = field(default_factory=dict)
    purchases: list = field(default_factory=list)
    applied_action_ids: set = field(default_factory=set)
    sync_required: bool = False

    def buy(self, name, action_id=None):
        if action_id is not None and action_id in self.applied_action_ids:
            return self.copies.get(name, 0)
        if action_id is not None:
            self.applied_action_ids.add(action_id)
        self.copies[name] = self.copies.get(name, 0) + 1
        self.purchases.append(name)
        return self.copies[name]

    def star(self, name):
        copies = self.copies.get(name, 0)
        star = 1
        for step in STAR_STEPS:
            if copies >= step:
                star += 1
        return star

    def copies_to_next_star(self, name):
        copies = self.copies.get(name, 0)
        for step in STAR_STEPS:
            if copies < step:
                return step - copies
        return 0

    def report(self):
        if not self.copies:
            return "nothing bought yet"
        return ", ".join(
            f"{name} x{copies} ({self.star(name)}*)" for name, copies in sorted(self.copies.items())
        )


class DeployPlanner:
    def __init__(self, reader, adb, tap_delay=0.6, diagnostics=None, verify_actions=False):
        self.reader = reader
        self.adb = adb
        self.tap_delay = tap_delay
        self.diagnostics = diagnostics
        self.verify_actions = verify_actions
        self.action_verifier = ActionVerifier()
        self._last_decision = None

    def choose(self, cards, elixir, state, *, action_id=None, frame_id=None,
               elixir_certainty="observed"):
        """Pick the card to buy, or None to save the elixir"""
        diagnostic_enabled = bool(self.diagnostics and self.diagnostics.enabled)
        decision_started = time.perf_counter() if diagnostic_enabled else None
        affordable = [card for card in cards if card.cost <= elixir]
        if not affordable:
            if diagnostic_enabled:
                reason = "no_cards" if not cards else "insufficient_elixir"
                self._record_decision(cards, affordable, elixir, state, None, reason,
                                      action_id, frame_id, decision_started, elixir_certainty)
            return None

        # 1. A duplicate merges a deployed troop and hands one elixir back.
        duplicates = [card for card in affordable if card.name in state.copies]
        if duplicates:
            duplicates.sort(key=lambda card: (state.copies_to_next_star(card.name), card.cost))
            choice = duplicates[0]
            if diagnostic_enabled:
                tie = sum((state.copies_to_next_star(card.name), card.cost) ==
                          (state.copies_to_next_star(choice.name), choice.cost)
                          for card in cards if card.name in state.copies) > 1
                self._record_decision(cards, affordable, elixir, state, choice, "duplicate",
                                      action_id, frame_id, decision_started, elixir_certainty,
                                      "original_order" if tie else None)
            return choice

        # 2. Strengthen the target comp if a target troop is on offer.
        targets = [card for card in affordable if card.name in TARGET_COMP]
        if targets:
            targets.sort(key=lambda card: (target_rank(card.name), card.cost))
            choice = targets[0]
            if diagnostic_enabled:
                tie = sum((target_rank(card.name), card.cost) ==
                          (target_rank(choice.name), choice.cost)
                          for card in cards if card.name in TARGET_COMP and card.cost <= elixir) > 1
                self._record_decision(cards, affordable, elixir, state, choice, "target_tier_priority",
                                      action_id, frame_id, decision_started, elixir_certainty,
                                      "original_order" if tie else None)
            return choice

        # 3. Nothing we want on the shelf. Buy the cheapest card anyway: a
        #    purchase replaces that shop slot with a new card, so buying rerolls
        #    the shop and lets a target/duplicate appear instead of stalling
        #    with the same three cards every round.
        choice = min(affordable, key=lambda card: card.cost)
        if diagnostic_enabled:
            tie = sum(card.cost == choice.cost for card in affordable) > 1
            self._record_decision(cards, affordable, elixir, state, choice, "cheapest_fallback",
                                  action_id, frame_id, decision_started, elixir_certainty,
                                  "original_order" if tie else None)
        return choice

    def _record_decision(self, cards, affordable, elixir, state, choice, reason,
                         action_id, frame_id, started, elixir_certainty, tiebreak=None):
        if not self.diagnostics or not self.diagnostics.enabled:
            return

        affordable_ids = {id(card) for card in affordable}
        candidates = []
        for card in cards:
            copies = state.copies.get(card.name, 0)
            try:
                slot_index = card.slot_index
                if slot_index is None:
                    slot_index = SHOP_CENTERS.index(tuple(card.center))
            except (AttributeError, TypeError, ValueError):
                slot_index = None
            candidates.append({
                "name": card.name,
                "cost": card.cost,
                "cost_source": getattr(card, "cost_source", None),
                "name_inliers": getattr(card, "name_inliers", None),
                "slot_index": slot_index,
                "center": list(card.center) if getattr(card, "center", None) is not None else None,
                "affordable": id(card) in affordable_ids,
                "is_duplicate": card.name in state.copies,
                "copies_before": {"value": copies, "certainty": "inferred",
                                  "source": "RunState.copies"},
                "copies_to_next_star": {"value": state.copies_to_next_star(card.name),
                                         "certainty": "inferred", "calculation": "RunState"},
                "target_rank": target_rank(card.name),
                "target_member": card.name in TARGET_COMP,
            })

        selected = next((candidate for card, candidate in zip(cards, candidates)
                         if card is choice), None)
        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
        self._last_decision = (action_id, reason)
        self.diagnostics.event("decision", frame_id=frame_id, action_id=action_id,
                               source="DeployPlanner.choose", data={
            "elixir": {"value": elixir, "certainty": elixir_certainty},
            "offered_cards": candidates,
            "affordable_cards": [item for item in candidates if item["affordable"]],
            "unaffordable_cards": [item for item in candidates if not item["affordable"]],
            "duplicate_cards": [item for item in candidates if item["affordable"] and item["is_duplicate"]],
            "target_cards": [item for item in candidates if item["affordable"] and item["target_member"]],
            "selected": selected,
            "reason": reason,
            "legacy_save_reason": self._save_reason(cards, elixir, state) if choice is None else None,
            "selection_tiebreak": tiebreak,
            "run_state_before": {"copies": dict(state.copies), "purchases": list(state.purchases),
                                 "distinct_kinds": len(state.copies), "certainty": "inferred"},
            "decision_duration_ms": elapsed_ms,
        })

    def play(self, screen, state, max_buys=6, dry_run=False):
        """Play one deploy phase: pop the orbs, then spend the elixir"""
        if self.verify_actions and state.sync_required:
            return {"elixir": None, "bought": [], "orbs": 0, "units": 0,
                    "units_before": 0, "roster": state.report(), "reason": "state_sync_required"}
        diagnostic_enabled = bool(self.diagnostics and self.diagnostics.enabled)
        frame_id = self.diagnostics.record_frame(screen, reason="deploy_phase_input") if diagnostic_enabled else None
        orbs = self.reader.orbs(screen)
        if diagnostic_enabled:
            self.diagnostics.event("orb_detection", frame_id=frame_id,
                                   source="DeployPlanner.play",
                                   data={"coordinates": [list(point) for point in orbs],
                                         "count": len(orbs), "certainty": "observed",
                                         "call_source": "deploy_planner"})
        # MergeTacticsBot.collect_orbs owns taps. Keeping planner detection here
        # is useful for diagnostics, but tapping again would reuse the same frame.

        elixir = self._read_elixir(screen)
        elixir_certainty = "observed" if elixir is not None else "unknown"
        if self.verify_actions and (screen is None or self.reader.phase(screen) != "deploy"):
            return {"elixir": elixir, "bought": [], "orbs": len(orbs), "units": 0,
                    "units_before": 0, "roster": state.report(), "reason": "phase_unknown"}
        if elixir is None and not dry_run:
            # A round opens with a modal notification that covers the elixir
            # badge and the shop for a few seconds, so poll until it clears.
            for _ in range(10):
                time.sleep(0.7)
                fresh = self._take_screenshot("elixir_poll")
                if fresh is not None:
                    screen = fresh
                    frame_id = self.diagnostics.frame_id(screen) if diagnostic_enabled else None
                    if self.verify_actions and self.reader.phase(screen) != "deploy":
                        return {"elixir": None, "bought": [], "orbs": len(orbs), "units": 0,
                                "units_before": 0, "roster": state.report(), "reason": "phase_ended"}
                    elixir = self._read_elixir(screen)
                    if elixir is not None:
                        elixir_certainty = "observed"
                if elixir is not None:
                    break
        units_before = self.reader.own_units(screen)
        if diagnostic_enabled:
            self._record_state(screen, frame_id, elixir, elixir_certainty, None, None,
                               units_before, None, state, "deploy_initial_state")

        bought = []
        reason = "done"
        for _ in range(max_buys):
            if screen is None or elixir is None:
                reason = "no_elixir"
                if diagnostic_enabled:
                    self.diagnostics.event("purchase_refused", frame_id=frame_id,
                                           action_id=self.diagnostics.new_action_id(),
                                           source="DeployPlanner.play",
                                           data={"reason": "unknown_elixir" if elixir is None else "no_frame",
                                                 "recognized_elixir": None if elixir is None else elixir,
                                                 "certainty": "unknown" if elixir is None else elixir_certainty})
                break
            cards = self.reader.shop_cards(screen)
            if diagnostic_enabled:
                self._record_state(screen, frame_id, elixir, elixir_certainty,
                                   cards, None, None, None, state, "shop_observation")
            if not cards and not dry_run:
                # The purchase animation briefly hides the shop; retry before
                # giving up so leftover elixir is not wasted.
                for _ in range(5):
                    time.sleep(0.6)
                    screen = self._take_screenshot("shop_recognition_retry")
                    if screen is None:
                        continue
                    frame_id = self.diagnostics.frame_id(screen) if diagnostic_enabled else None
                    if self.verify_actions and self.reader.phase(screen) != "deploy":
                        reason = "phase_ended"
                        break
                    if self.verify_actions:
                        elixir = self._read_elixir(screen)
                        elixir_certainty = "observed" if elixir is not None else "unknown"
                    cards = self.reader.shop_cards(screen)
                    if cards:
                        break
                if reason == "phase_ended":
                    break
                if self.verify_actions and elixir is None:
                    reason = "no_elixir"
                    break
            if not cards:
                # Could not see the shop at all. Do not treat this as a decision:
                # the caller retries on a later frame of the same deploy phase.
                reason = "no_cards"
                logger.info("[SHOP] no shop cards recognised, will retry")
                if diagnostic_enabled:
                    self.diagnostics.event("purchase_refused", frame_id=frame_id,
                                           action_id=self.diagnostics.new_action_id(),
                                           source="DeployPlanner.play",
                                           data={"reason": "no_cards", "recognized_cards": [],
                                                 "elixir": {"value": elixir, "certainty": elixir_certainty}})
                break
            action_id = self.diagnostics.new_action_id() if diagnostic_enabled else None
            frame_id = self.diagnostics.record_frame(screen, reason="shop_decision") if diagnostic_enabled else None
            if diagnostic_enabled:
                choice = self.choose(cards, elixir, state, action_id=action_id, frame_id=frame_id,
                                     elixir_certainty=elixir_certainty)
            else:
                choice = self.choose(cards, elixir, state)
            if choice is None:
                reason = self._save_reason(cards, elixir, state)
                logger.info(
                    f"[SHOP] saving {elixir} elixir ({reason}) | "
                    f"shop {[(c.name, c.cost) for c in cards]}"
                )
                break

            if self.verify_actions and self.reader.phase(screen) != "deploy":
                reason = "phase_ended"
                break

            logger.info(f"[BUY] {choice.name} ({choice.cost}) -> {state.star(choice.name)}*")
            if dry_run:
                if diagnostic_enabled:
                    copies_before = state.copies.get(choice.name, 0)
                    star_before = state.star(choice.name)
                state.buy(choice.name)
                bought.append(choice.name)
                elixir -= choice.cost
                if diagnostic_enabled:
                    self.diagnostics.event("buy_simulated", frame_id=frame_id, action_id=action_id,
                                           source="DeployPlanner.play",
                                           data={"name": choice.name, "cost": choice.cost,
                                                 "certainty": "assumed", "dry_run": True})
                    self._record_upgrade(choice.name, copies_before, star_before, state,
                                         frame_id, action_id)
                continue

            elixir_before = elixir
            before_screen = screen
            before_frame_id = frame_id
            budget = VerificationBudget(VERIFY_ACTION_TIMEOUT_SECONDS) if self.verify_actions else None
            verification_started = time.monotonic()
            if diagnostic_enabled:
                copies_before = state.copies.get(choice.name, 0)
                star_before = state.star(choice.name)
                self.diagnostics.event("buy_attempt", frame_id=frame_id, action_id=action_id,
                                       source="DeployPlanner.play",
                                       data={"name": choice.name, "cost": choice.cost,
                                             "cost_source": getattr(choice, "cost_source", None),
                                             "cost_certainty": ("observed" if getattr(choice, "cost_source", None) == "recognized_digit"
                                                                else "inferred" if getattr(choice, "cost_source", None) == "fallback_unit_costs"
                                                                else "unknown"),
                                             "slot_index": self._slot_index(choice),
                                             "coordinates": list(choice.center),
                                             "elixir_before": {"value": elixir_before,
                                                               "certainty": elixir_certainty},
                                             "run_state_before": {"copies": dict(state.copies),
                                                                  "purchases": list(state.purchases),
                                                                  "certainty": "inferred"},
                                             "selection_reason": self._decision_reason(action_id),
                                             "tap_status": "attempted"})
                try:
                    adb_result = (self.adb.tap(*choice.center, timeout=budget.remaining(), return_result=True)
                                  if self.verify_actions else self.adb.tap(*choice.center))
                except Exception as exc:
                    self.diagnostics.event("adb_error", frame_id=frame_id, action_id=action_id,
                                           source="DeployPlanner.play",
                                           data={"operation": "buy_tap", "error_type": type(exc).__name__,
                                                 "certainty": "observed"})
                    if not self.verify_actions:
                        raise
                    adb_result = ADBActionResult("exception", None,
                                                  max(0, round((time.monotonic() - verification_started) * 1000)))
                self.diagnostics.event("buy_tap_returned", frame_id=frame_id, action_id=action_id,
                                       source="DeployPlanner.play",
                                       data={"adb_call_returned": getattr(adb_result, "status", None) != "exception",
                                             "adb_status": getattr(adb_result, "status", "succeeded"),
                                             "device_confirmation": "unknown"})
            else:
                try:
                    adb_result = (self.adb.tap(*choice.center, timeout=budget.remaining(), return_result=True)
                                  if self.verify_actions else self.adb.tap(*choice.center))
                except Exception:
                    if not self.verify_actions:
                        raise
                    adb_result = ADBActionResult("exception", None,
                                                  max(0, round((time.monotonic() - verification_started) * 1000)))
            time.sleep(min(self.tap_delay, budget.remaining()) if budget else self.tap_delay)
            capture_started = time.monotonic()
            try:
                screen = self._take_screenshot("after_buy", action_id,
                                               timeout=budget.remaining() if budget else None)
            except Exception:
                screen = None
            capture_duration_ms = round((time.monotonic() - capture_started) * 1000)
            frame_id = self.diagnostics.frame_id(screen) if diagnostic_enabled else None
            recognition_started = time.monotonic()
            read = self._read_elixir(screen) if screen is not None else None
            if self.verify_actions:
                after_cards = self.reader.shop_cards(screen) if screen is not None else []
                before_observation = Observation(
                    phase=self.reader.phase(before_screen), elixir=elixir_before,
                    shop_cards=self._shop_signature(cards),
                    selected_slot=self._selected_signature(cards, choice),
                )
                after_observation = Observation(
                    phase=self.reader.phase(screen) if screen is not None else None,
                    elixir=read, shop_cards=self._shop_signature(after_cards),
                    selected_slot=self._selected_signature(after_cards, choice),
                )
                result = self.action_verifier.verify_purchase(
                    action_id=action_id or f"local-{uuid.uuid4().hex}",
                    adb_status=getattr(adb_result, "status", "succeeded"),
                    frame_before=before_frame_id, frame_after=frame_id,
                    before=before_observation, after=after_observation, cost=choice.cost,
                    started_at=verification_started,
                )
                recognition_duration_ms = round((time.monotonic() - recognition_started) * 1000)
                if result.status is ActionStatus.UNKNOWN:
                    for recheck in range(VERIFY_MAX_RECHECKS):
                        if budget.remaining() <= 0:
                            break
                        recheck_capture_started = time.monotonic()
                        try:
                            screen = self._take_screenshot("verification_recheck", action_id,
                                                           timeout=budget.remaining())
                        except Exception:
                            screen = None
                        capture_duration_ms += round((time.monotonic() - recheck_capture_started) * 1000)
                        frame_id = self.diagnostics.frame_id(screen) if diagnostic_enabled else None
                        recheck_recognition_started = time.monotonic()
                        read = self._read_elixir(screen) if screen is not None else None
                        after_cards = self.reader.shop_cards(screen) if screen is not None else []
                        after_observation = Observation(
                            phase=self.reader.phase(screen) if screen is not None else None,
                            elixir=read, shop_cards=self._shop_signature(after_cards),
                            selected_slot=self._selected_signature(after_cards, choice),
                        )
                        recognition_duration_ms += round(
                            (time.monotonic() - recheck_recognition_started) * 1000
                        )
                        result = self.action_verifier.verify_purchase(
                            action_id=result.action_id, adb_status=result.adb_status,
                            frame_before=before_frame_id, frame_after=frame_id,
                            before=before_observation, after=after_observation, cost=choice.cost,
                            started_at=verification_started, rechecks=recheck + 1,
                        )
                        if result.status is not ActionStatus.UNKNOWN:
                            break
                if diagnostic_enabled:
                    self.diagnostics.event(
                        "action_verification_result", frame_id=frame_id, action_id=action_id,
                        source="DeployPlanner.play", data={
                            **result.as_dict(),
                            "adb_command_duration_ms": getattr(adb_result, "duration_ms", None),
                            "frame_capture_duration_ms": capture_duration_ms,
                            "recognition_duration_ms": recognition_duration_ms,
                        },
                    )
                if result.status is not ActionStatus.CONFIRMED:
                    if result.status in (ActionStatus.UNKNOWN, ActionStatus.INTERRUPTED):
                        state.sync_required = True
                    reason = "verification_" + result.status.value.lower()
                    break
                state.buy(choice.name, action_id=result.action_id)
                bought.append(choice.name)
                elixir = read
                elixir_certainty = "observed"
                continue
            # A successful buy spends elixir and takes the card off the shelf, so
            # only treat it as "bench full" when the balance did not drop AND the
            # same card is still on offer.
            if read is not None and read >= elixir_before:
                still_offered = any(
                    c.name == choice.name and c.cost == choice.cost
                    for c in self.reader.shop_cards(screen)
                )
                if still_offered:
                    logger.info("[SHOP] buy had no effect (bench full?), selling a bench unit")
                    if diagnostic_enabled:
                        self.diagnostics.event("buy_outcome", frame_id=frame_id, action_id=action_id,
                                               source="DeployPlanner.play",
                                               data={"legacy_classification": "no_effect_suspected",
                                                     "confirmation": "inferred",
                                                     "recognized_elixir_after": read,
                                                     "same_card_still_offered": True})
                    sold = (self._sell_one(screen, action_id=action_id, reason="suspected_no_effect")
                            if diagnostic_enabled else self._sell_one(screen))
                    if sold:
                        screen = self._take_screenshot("after_sale", action_id)
                        frame_id = self.diagnostics.frame_id(screen) if diagnostic_enabled else None
                        reason = "bench full"
                        continue
                    reason = "bench full, could not sell"
                    break

            state.buy(choice.name)
            bought.append(choice.name)
            elixir = read if read is not None else elixir_before - choice.cost
            elixir_certainty = "observed" if read is not None else "assumed"
            if diagnostic_enabled:
                self.diagnostics.event("buy_outcome", frame_id=frame_id, action_id=action_id,
                                       source="DeployPlanner.play",
                                       data={"legacy_classification": "accepted_by_legacy_logic",
                                             "confirmation": "inferred" if read is not None else "assumed",
                                             "recognized_elixir_after": {"value": read,
                                                                          "certainty": "observed" if read is not None else "unknown"},
                                             "assumed_balance_after": {"value": elixir,
                                                                        "certainty": "inferred" if read is not None else "assumed",
                                                                        "reason": "legacy_cost_subtraction" if read is None else None},
                                             "run_state_after": {"copies": dict(state.copies),
                                                                 "purchases": list(state.purchases),
                                                                 "certainty": "inferred"}})
                self._record_upgrade(choice.name, copies_before, star_before, state,
                                     frame_id, action_id)
        else:
            if diagnostic_enabled:
                self.diagnostics.event("purchase_limit", frame_id=frame_id,
                                       source="DeployPlanner.play",
                                       data={"max_buys": max_buys, "certainty": "observed"})

        units_after = self.reader.own_units(screen) if screen is not None else 0
        if diagnostic_enabled:
            self._record_state(screen, self.diagnostics.frame_id(screen), elixir, elixir_certainty,
                               None, None, None, units_after, state, "deploy_final_state")
        summary = {
            "elixir": elixir,
            "bought": bought,
            "orbs": len(orbs),
            "units": units_after,
            "units_before": units_before,
            "roster": state.report(),
            "reason": reason,
        }
        logger.info(
            f"[DEPLOY] bought {bought or 'nothing'} | elixir left {elixir} ({reason}) | "
            f"units {units_before}->{units_after} | roster: {state.report()}"
        )
        return summary

    def _diagnostics_enabled(self):
        return bool(self.diagnostics and self.diagnostics.enabled)

    def _take_screenshot(self, reason, action_id=None, timeout=None):
        capture = (lambda: self.adb.screenshot(timeout=5 if timeout is None else timeout)) \
            if self.verify_actions else self.adb.screenshot
        if not self._diagnostics_enabled():
            return capture()
        try:
            shot = capture()
        except Exception as exc:
            self.diagnostics.event("screenshot_error", action_id=action_id,
                                   source="DeployPlanner.play",
                                   data={"reason": reason, "error_type": type(exc).__name__,
                                         "certainty": "observed"})
            raise
        if shot is not None:
            self.diagnostics.record_frame(shot, reason=reason)
        return shot

    def _record_state(self, screen, frame_id, elixir, elixir_certainty,
                      cards, orbs, units_before, units_after, state, source):
        if not self._diagnostics_enabled():
            return
        card_records = None
        if cards is not None:
            card_records = [{
                "name": card.name,
                "cost": {"value": card.cost,
                         "certainty": ("observed" if getattr(card, "cost_source", None) == "recognized_digit"
                                       else "inferred" if getattr(card, "cost_source", None) == "fallback_unit_costs"
                                       else "unknown"),
                         "source": getattr(card, "cost_source", None)},
                "slot_index": self._slot_index(card),
                "coordinates": list(card.center),
                "name_inliers": getattr(card, "name_inliers", None),
                "certainty": "observed",
            } for card in cards]
        self.diagnostics.event("state_snapshot", frame_id=frame_id,
                               source="DeployPlanner.play",
                               data={"snapshot_source": source,
                                     "screen_available": screen is not None,
                                     "elixir": {"value": elixir, "certainty": elixir_certainty},
                                     "shop_cards": card_records,
                                     "orbs": {"coordinates": [list(p) for p in orbs],
                                              "certainty": "observed"} if orbs is not None else None,
                                     "own_units": {
                                         "before": {"value": units_before, "certainty": "approximate"}
                                         if units_before is not None else None,
                                         "after": {"value": units_after, "certainty": "approximate"}
                                         if units_after is not None else None,
                                     },
                                     "run_state": {"copies": dict(state.copies),
                                                   "purchases": list(state.purchases),
                                                   "distinct_kinds": len(state.copies),
                                                   "calculated_stars": {
                                                       name: state.star(name) for name in state.copies
                                                   },
                                                   "certainty": "inferred"}})

    def _record_upgrade(self, name, copies_before, star_before, state, frame_id, action_id):
        if not self._diagnostics_enabled():
            return
        copies_after = state.copies.get(name, 0)
        star_after = state.star(name)
        self.diagnostics.event("calculated_upgrade", frame_id=frame_id, action_id=action_id,
                               source="RunState.star",
                               data={"name": name,
                                     "copies_before": {"value": copies_before, "certainty": "inferred"},
                                     "copies_after": {"value": copies_after, "certainty": "inferred"},
                                     "stars_before": {"value": star_before, "certainty": "calculated"},
                                     "stars_after": {"value": star_after, "certainty": "calculated"},
                                     "threshold_reached": star_after > star_before,
                                     "merge_observed": False})

    @staticmethod
    def _slot_index(card):
        try:
            if card.slot_index is not None:
                return card.slot_index
            return SHOP_CENTERS.index(tuple(card.center))
        except (AttributeError, TypeError, ValueError):
            return None

    def _shop_signature(self, cards):
        entries = tuple((self._slot_index(card), card.name, card.cost) for card in cards)
        if len(entries) != len(SHOP_CENTERS) or {entry[0] for entry in entries} != set(range(len(SHOP_CENTERS))):
            return None
        return tuple(sorted(entries))

    def _selected_signature(self, cards, choice):
        slot = self._slot_index(choice)
        selected = [card for card in cards if self._slot_index(card) == slot]
        if slot is None or len(selected) != 1:
            return None
        return slot, (selected[0].name, selected[0].cost)

    def _decision_reason(self, action_id):
        if self._last_decision and self._last_decision[0] == action_id:
            return self._last_decision[1]
        return None

    def _sell_one(self, screen, *, action_id=None, reason=None):
        """Sell one reserve (bench) unit to free space; True if one was sold.

        Only bench cells are tapped, never the board, so a merged or deployed
        troop is never sold.
        """
        ip = self.reader.image_processor
        diagnostic_enabled = self._diagnostics_enabled()
        initial_frame_id = self.diagnostics.frame_id(screen) if diagnostic_enabled else None
        if diagnostic_enabled:
            self.diagnostics.event("sale_attempt_started", frame_id=initial_frame_id,
                                   action_id=action_id, source="DeployPlanner._sell_one",
                                   data={"reason": reason, "sold_unit": None})
        for cx, cy in BENCH_CELLS:
            if diagnostic_enabled:
                self.diagnostics.event("sale_slot_tap_attempt", frame_id=initial_frame_id,
                                       action_id=action_id, source="DeployPlanner._sell_one",
                                       data={"slot_center": [cx, cy], "status": "attempted"})
                try:
                    self.adb.tap(cx, cy)
                except Exception as exc:
                    self.diagnostics.event("adb_error", frame_id=initial_frame_id, action_id=action_id,
                                           source="DeployPlanner._sell_one",
                                           data={"operation": "sale_slot_tap", "error_type": type(exc).__name__,
                                                 "coordinates": [cx, cy], "certainty": "observed"})
                    raise
            else:
                self.adb.tap(cx, cy)
            if diagnostic_enabled:
                self.diagnostics.event("sale_slot_tap_returned", frame_id=initial_frame_id,
                                       action_id=action_id, source="DeployPlanner._sell_one",
                                       data={"slot_center": [cx, cy], "device_confirmation": "unknown"})
            time.sleep(0.4)
            shot = self._take_screenshot("sale_slot_probe", action_id)
            shot_frame_id = self.diagnostics.frame_id(shot) if diagnostic_enabled else None
            pos = ip.find_template(shot, SELL_BUTTON, SELL_THRESHOLD)
            if diagnostic_enabled:
                self.diagnostics.event("sale_slot_probe", frame_id=shot_frame_id,
                                       action_id=action_id, source="DeployPlanner._sell_one",
                                       data={"slot_center": [cx, cy], "sell_button_found": bool(pos),
                                             "sell_button_coordinates": list(pos) if pos else None,
                                             "certainty": "observed"})
            if pos:
                logger.info(f"[SELL] selling a bench unit at ({cx}, {cy})")
                if diagnostic_enabled:
                    self.diagnostics.event("sale_button_tap_attempt", frame_id=shot_frame_id,
                                           action_id=action_id, source="DeployPlanner._sell_one",
                                           data={"coordinates": list(pos), "sold_unit": None,
                                                 "status": "attempted"})
                    try:
                        self.adb.tap(*pos)
                    except Exception as exc:
                        self.diagnostics.event("adb_error", frame_id=shot_frame_id, action_id=action_id,
                                               source="DeployPlanner._sell_one",
                                               data={"operation": "sale_button_tap", "error_type": type(exc).__name__,
                                                     "coordinates": list(pos), "certainty": "observed"})
                        raise
                else:
                    self.adb.tap(*pos)
                if diagnostic_enabled:
                    self.diagnostics.event("sale_outcome", frame_id=shot_frame_id,
                                           action_id=action_id, source="DeployPlanner._sell_one",
                                           data={"legacy_return": True, "confirmation": "unknown",
                                                 "button_tap_path": True, "sold_unit": None})
                time.sleep(0.6)
                return True
        if diagnostic_enabled:
            self.diagnostics.event("sale_outcome", frame_id=initial_frame_id,
                                   action_id=action_id, source="DeployPlanner._sell_one",
                                   data={"legacy_return": False, "confirmation": "unknown",
                                         "sell_button_found": False, "sold_unit": None})
        return False

    def _save_reason(self, cards, elixir, state):
        """Human-readable reason the planner chose to save elixir"""
        affordable = [card for card in cards if card.cost <= elixir]
        if not affordable:
            return "nothing affordable"
        if len(state.copies) >= MAX_DISTINCT_TROOPS and all(
            card.name not in state.copies for card in affordable
        ):
            return f"roster full ({MAX_DISTINCT_TROOPS} kinds), waiting for duplicates"
        return "no target troop on offer"

    def _read_elixir(self, screen):
        if screen is None:
            return None
        return self.reader.image_processor.read_elixir(screen, list(ELIXIR_RECT), force=True)
