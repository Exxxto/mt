import unittest
from unittest.mock import Mock, patch

from core.board_state import ShopCard
from core.game_logic import GameLogic
from core.strategy import DeployPlanner, RunState
from bot import MergeTacticsBot
from config.constants import BATTLE_WAIT


def card(name, cost, x=100):
    return ShopCard(name=name, cost=cost, center=(x, 100))


class DeployPlannerChoiceTests(unittest.TestCase):
    def setUp(self):
        self.planner = DeployPlanner(reader=None, adb=None)

    def test_empty_shop_returns_no_choice(self):
        self.assertIsNone(self.planner.choose([], 10, RunState()))

    def test_no_affordable_cards_returns_no_choice(self):
        self.assertIsNone(self.planner.choose([card("knight", 2)], 1, RunState()))

    def test_unaffordable_target_is_excluded_from_available_choices(self):
        cheap = card("goblin_cage", 1)
        expensive_target = card("boss_bandit", 5)

        self.assertIs(self.planner.choose([expensive_target, cheap], 1, RunState()), cheap)

    def test_zero_elixir_returns_no_choice(self):
        self.assertIsNone(self.planner.choose([card("knight", 2)], 0, RunState()))

    def test_bought_duplicate_precedes_target_comp(self):
        state = RunState()
        state.buy("goblin")
        target = card("boss_bandit", 5)
        duplicate = card("goblin", 2)

        self.assertIs(self.planner.choose([target, duplicate], 5, state), duplicate)

    def test_duplicate_with_fewer_copies_to_next_star_wins(self):
        state = RunState()
        state.copies.update(knight=2, archer=3)
        knight = card("knight", 2)
        archer = card("archer", 3)

        self.assertIs(self.planner.choose([knight, archer], 5, state), archer)

    def test_equal_duplicate_priority_uses_lower_cost(self):
        state = RunState()
        state.copies.update(knight=1, archer=1)
        knight = card("knight", 2)
        archer = card("archer", 3)

        self.assertIs(self.planner.choose([archer, knight], 5, state), knight)

    def test_duplicate_priority_uses_copies_before_cost(self):
        state = RunState()
        state.copies.update(knight=2, archer=3)
        expensive_closer = card("archer", 5)
        cheaper_farther = card("knight", 1)

        self.assertIs(
            self.planner.choose([cheaper_farther, expensive_closer], 5, state),
            expensive_closer,
        )

    def test_target_comp_card_is_chosen_when_no_duplicate_is_available(self):
        target = card("knight", 2)
        non_target = card("goblin_cage", 1)

        self.assertIs(self.planner.choose([non_target, target], 5, RunState()), target)

    def test_higher_target_rank_precedes_cheaper_lower_rank(self):
        highest_tier = card("boss_bandit", 5)
        lower_tier = card("knight", 2)

        self.assertIs(
            self.planner.choose([lower_tier, highest_tier], 5, RunState()),
            highest_tier,
        )

    def test_equal_target_rank_uses_lower_cost(self):
        first = card("knight", 4)
        second = card("knight", 2)

        self.assertIs(self.planner.choose([first, second], 5, RunState()), second)

    def test_cheapest_non_target_is_bought_when_no_target_is_offered(self):
        expensive = card("dark_prince", 5)
        cheap = card("goblin_cage", 2)

        self.assertIs(self.planner.choose([expensive, cheap], 5, RunState()), cheap)

    def test_equal_cost_non_targets_keep_shop_order(self):
        first = card("dark_prince", 2)
        second = card("goblin_cage", 2)

        self.assertIs(self.planner.choose([first, second], 5, RunState()), first)

    def test_six_distinct_kinds_do_not_block_a_new_non_target(self):
        state = RunState()
        state.copies.update({f"unit_{n}": 1 for n in range(6)})
        non_target = card("goblin_cage", 2)

        self.assertIs(self.planner.choose([non_target], 5, state), non_target)

    def test_safe_mode_does_not_use_uncertain_copy_counts(self):
        state = RunState(copies={"goblin": 1})
        target = card("knight", 3)
        apparent_duplicate = card("goblin", 1)

        selected = self.planner.choose([apparent_duplicate, target], 5, state,
                                       ignore_copies=True)

        self.assertIs(selected, target)


class RunStateTests(unittest.TestCase):
    def test_initial_state_has_no_copies_or_purchases(self):
        state = RunState()

        self.assertEqual(state.copies, {})
        self.assertEqual(state.purchases, [])
        self.assertEqual(state.report(), "nothing bought yet")

    def test_buy_registers_a_copy_and_returns_total(self):
        state = RunState()

        self.assertEqual(state.buy("knight"), 1)
        self.assertEqual(state.copies, {"knight": 1})
        self.assertEqual(state.purchases, ["knight"])

    def test_same_confirmed_action_id_updates_run_state_once(self):
        state = RunState()

        state.buy("knight", action_id="action-1")
        state.buy("knight", action_id="action-1")

        self.assertEqual(state.copies, {"knight": 1})
        self.assertEqual(state.purchases, ["knight"])

    def test_repeated_and_distinct_buys_are_counted(self):
        state = RunState()
        state.buy("knight")
        state.buy("knight")
        state.buy("archer")

        self.assertEqual(state.copies, {"knight": 2, "archer": 1})
        self.assertEqual(state.purchases, ["knight", "knight", "archer"])
        self.assertEqual(len(state.copies), 2)

    def test_star_thresholds_at_one_two_four_and_eight_copies(self):
        state = RunState()
        expected_stars = {1: 1, 2: 2, 4: 3, 8: 4}

        for copies, stars in expected_stars.items():
            state.copies["knight"] = copies
            self.assertEqual(state.star("knight"), stars)

    def test_copies_to_next_star_uses_two_four_and_eight_thresholds(self):
        state = RunState()
        expected = {0: 2, 1: 1, 2: 2, 3: 1, 4: 4, 7: 1, 8: 0}

        for copies, remaining in expected.items():
            state.copies["knight"] = copies
            self.assertEqual(state.copies_to_next_star("knight"), remaining)

    def test_report_sorts_names_and_uses_current_star_count(self):
        state = RunState(copies={"knight": 2, "archer": 4})

        self.assertEqual(state.report(), "archer x4 (3*), knight x2 (2*)")

    def test_start_run_resets_run_state(self):
        logic = GameLogic.__new__(GameLogic)
        logic.run_state = RunState(copies={"knight": 2}, purchases=["knight", "knight"])

        logic.start_run()

        self.assertEqual(logic.run_state.copies, {})
        self.assertEqual(logic.run_state.purchases, [])


class DeployPlannerExecutionTests(unittest.TestCase):
    class FakeADB:
        def __init__(self, frames=()):
            self.frames = list(frames)
            self.taps = []
            self.screenshots = 0

        def tap(self, x, y):
            self.taps.append((x, y))

        def screenshot(self, timeout=None):
            self.screenshots += 1
            return self.frames.pop(0) if self.frames else object()

    class FakeImageProcessor:
        def __init__(self, elixir_by_frame, sell_button=None):
            self.elixir_by_frame = elixir_by_frame
            self.sell_button = sell_button
            self.elixir_reads = []

        def read_elixir(self, screen, _rect, force=False):
            self.elixir_reads.append((screen, force))
            return self.elixir_by_frame.get(screen)

        def find_template(self, _screen, _path, _threshold):
            return self.sell_button

    class FakeReader:
        def __init__(self, image_processor, cards_by_frame=None, cards=None, orbs=(), phases=None):
            self.image_processor = image_processor
            self.cards_by_frame = cards_by_frame or {}
            self.cards = cards or []
            self.orb_list = list(orbs)
            self.phases = phases or {}
            self.shop_frames = []

        def orbs(self, _screen):
            return list(self.orb_list)

        def own_units(self, _screen):
            return 2

        def shop_cards(self, screen):
            self.shop_frames.append(screen)
            return self.cards_by_frame.get(screen, self.cards)

        def phase(self, screen):
            return self.phases.get(screen, "deploy")

    def planner(self, frames=(), elixir_by_frame=None, cards=None, cards_by_frame=None,
                orbs=(), sell_button=None):
        image_processor = self.FakeImageProcessor(elixir_by_frame or {}, sell_button)
        reader = self.FakeReader(image_processor, cards_by_frame, cards, orbs)
        adb = self.FakeADB(frames)
        return DeployPlanner(reader, adb, tap_delay=0), reader, adb, image_processor

    @patch("core.strategy.time.sleep")
    def test_dry_run_stops_at_six_purchases(self, _sleep):
        planner, reader, _adb, _ip = self.planner(
            elixir_by_frame={"frame": 100}, cards=[card("knight", 1)]
        )
        state = RunState()

        result = planner.play("frame", state, dry_run=True)

        self.assertEqual(len(result["bought"]), 6)
        self.assertEqual(state.copies["knight"], 6)
        self.assertEqual(result["elixir"], 94)
        self.assertEqual(reader.shop_frames, ["frame"] * 6)

    @patch("core.strategy.time.sleep")
    def test_unknown_elixir_polls_then_returns_no_elixir(self, _sleep):
        planner, _reader, adb, _ip = self.planner(elixir_by_frame={})

        result = planner.play("initial", RunState())

        self.assertEqual(result["reason"], "no_elixir")
        self.assertIsNone(result["elixir"])
        self.assertEqual(adb.screenshots, 10)
        self.assertEqual(adb.taps, [])

    @patch("core.strategy.time.sleep")
    def test_unrecognized_shop_retries_five_frames_without_buying(self, _sleep):
        planner, reader, adb, _ip = self.planner(elixir_by_frame={"initial": 4})

        result = planner.play("initial", RunState())

        self.assertEqual(result["reason"], "no_cards")
        self.assertEqual(adb.screenshots, 5)
        self.assertEqual(adb.taps, [])
        self.assertEqual(len(reader.shop_frames), 6)

    @patch("core.strategy.time.sleep")
    def test_successful_purchase_uses_the_updated_screenshot(self, _sleep):
        initial, updated = object(), object()
        planner, reader, adb, ip = self.planner(
            frames=[updated],
            elixir_by_frame={initial: 3, updated: 0},
            cards_by_frame={initial: [card("knight", 2)], updated: [card("archer", 1)]},
        )

        result = planner.play(initial, RunState())

        self.assertEqual(result["bought"], ["knight"])
        self.assertEqual(reader.shop_frames, [initial, updated])
        self.assertEqual(adb.taps, [(100, 100)])
        self.assertTrue(all(force for _screen, force in ip.elixir_reads))

    @patch("core.strategy.time.sleep")
    def test_verify_mode_requires_balance_and_shop_change_and_updates_once(self, _sleep):
        from core.adb_controller import ADBActionResult

        initial, updated = object(), object()
        planner, reader, _adb, _ip = self.planner(
            frames=[updated], elixir_by_frame={initial: 3, updated: 1},
            cards_by_frame={initial: [
                ShopCard("knight", 2, (100, 100), slot_index=0),
                ShopCard("goblin", 1, (200, 100), slot_index=1),
                ShopCard("archer", 1, (300, 100), slot_index=2)], updated: [
                ShopCard("archer", 1, (100, 100), slot_index=0),
                ShopCard("goblin", 1, (200, 100), slot_index=1),
                ShopCard("knight", 2, (300, 100), slot_index=2)]},
        )
        reader.phase = lambda _frame: "deploy"
        planner.verify_actions = True
        planner.adb.tap = lambda *args, **_kwargs: ADBActionResult("succeeded", 0, 1)
        state = RunState()

        result = planner.play(initial, state)

        self.assertEqual(result["bought"], ["knight"])
        self.assertEqual(state.copies, {"knight": 1})

    @patch("core.strategy.time.sleep")
    def test_full_spend_and_unit_increase_confirm_when_shop_is_unrecognized(self, _sleep):
        from core.adb_controller import ADBActionResult

        initial, updated = object(), object()
        planner, reader, adb, _ip = self.planner(
            frames=[updated], elixir_by_frame={initial: 5, updated: 0},
            cards_by_frame={initial: [card("mega_knight", 5)], updated: []},
        )
        reader.own_units = lambda frame: 1 if frame is initial else 2
        reader.phase = lambda _frame: "deploy"
        planner.verify_actions = True
        planner.adb.tap = lambda *args, **kwargs: (
            adb.taps.append(args) or ADBActionResult("succeeded", 0, 1)
        )
        state = RunState()

        result = planner.play(initial, state)

        self.assertEqual(result["bought"], ["mega_knight"])
        self.assertEqual(state.copies, {"mega_knight": 1})
        self.assertEqual(adb.taps, [(100, 100)])

    @patch("core.strategy.time.sleep")
    def test_stale_purchase_frame_is_discarded_without_tapping(self, _sleep):
        import time
        from core.adb_controller import ADBActionResult

        stale, fresh, after = object(), object(), object()
        planner, _reader, adb, _ip = self.planner(
            frames=[after], elixir_by_frame={stale: 5, fresh: 5, after: 3},
            cards_by_frame={stale: [card("knight", 2)], fresh: [card("knight", 2)],
                            after: [card("archer", 1)]},
        )
        _reader.phase = lambda _frame: "deploy"
        _reader.own_units = lambda frame: 2 if frame in (stale, fresh) else 3
        planner.verify_actions = True
        planner.max_frame_age_seconds = 3
        planner.adb.tap = lambda *args, **kwargs: (
            adb.taps.append(args) or ADBActionResult("succeeded", 0, 1)
        )

        state = RunState()
        stale_result = planner.play(stale, state, captured_at=time.monotonic() - 35)
        fresh_result = planner.play(fresh, state, max_buys=1, captured_at=time.monotonic())

        self.assertEqual(stale_result["reason"], "stale_frame")
        self.assertEqual(fresh_result["bought"], ["knight"])
        self.assertEqual(adb.taps, [(100, 100)])

    @patch("core.strategy.time.sleep")
    def test_verify_mode_unknown_result_does_not_buy_or_sell(self, _sleep):
        from core.adb_controller import ADBActionResult

        initial, updated = object(), object()
        planner, reader, adb, _ip = self.planner(
            frames=[updated], elixir_by_frame={initial: 3, updated: None},
            cards_by_frame={initial: [card("knight", 2)], updated: [card("knight", 2)]},
        )
        reader.phase = lambda _frame: "deploy"
        planner.verify_actions = True
        planner.adb.tap = lambda *args, **_kwargs: (
            adb.taps.append(args) or ADBActionResult("succeeded", 0, 1)
        )
        state = RunState()

        result = planner.play(initial, state)

        self.assertEqual(result["bought"], [])
        self.assertEqual(state.copies, {})
        self.assertEqual(adb.taps, [(100, 100)])
        self.assertEqual(adb.screenshots, 2)
        self.assertTrue(state.sync_required)

    @patch("core.strategy.time.sleep")
    def test_verify_mode_phase_change_after_tap_marks_state_unsynced(self, _sleep):
        from core.adb_controller import ADBActionResult

        initial, after = object(), object()
        planner, reader, adb, _ip = self.planner(
            frames=[after], elixir_by_frame={initial: 3, after: 1},
            cards_by_frame={initial: [card("knight", 2)], after: [card("archer", 1)]},
        )
        reader.phase = lambda frame: "battle" if frame is after else "deploy"
        planner.verify_actions = True
        planner.adb.tap = lambda *args, **kwargs: (
            adb.taps.append(args) or ADBActionResult("succeeded", 0, 1)
        )
        state = RunState()

        result = planner.play(initial, state)

        self.assertEqual(result["reason"], "verification_interrupted")
        self.assertEqual(state.copies, {})
        self.assertTrue(state.sync_required)
        self.assertEqual(adb.taps, [(100, 100)])

    @patch("core.strategy.time.sleep")
    def test_verify_mode_adb_exception_is_unknown_without_repeat_or_sale(self, _sleep):
        initial, after = object(), object()
        planner, _reader, adb, _ip = self.planner(
            frames=[after, object()], elixir_by_frame={initial: 3, after: 3},
            cards_by_frame={initial: [card("knight", 2)], after: [card("knight", 2)]},
        )
        planner.verify_actions = True
        def fail_tap(*args, **kwargs):
            adb.taps.append(args)
            raise OSError("transport disappeared")
        planner.adb.tap = fail_tap
        state = RunState()

        result = planner.play(initial, state)

        self.assertEqual(result["reason"], "verification_pending")
        self.assertEqual(state.copies, {})
        self.assertTrue(state.sync_required)
        self.assertEqual(adb.taps, [(100, 100)])

    @patch("core.strategy.time.sleep")
    def test_later_reliable_observation_resolves_pending_purchase_once(self, _sleep):
        from core.adb_controller import ADBActionResult

        initial, after, recheck, fresh = object(), object(), object(), object()
        original = [ShopCard("knight", 2, (100, 100), slot_index=0),
                    ShopCard("goblin", 1, (200, 100), slot_index=1),
                    ShopCard("archer", 1, (300, 100), slot_index=2)]
        changed = [ShopCard("archer", 1, (100, 100), slot_index=0),
                   ShopCard("goblin", 1, (200, 100), slot_index=1),
                   ShopCard("knight", 2, (300, 100), slot_index=2)]
        planner, reader, adb, _ip = self.planner(
            frames=[after, recheck],
            elixir_by_frame={initial: 5, after: None, recheck: None, fresh: 3},
            cards_by_frame={initial: original, after: [], recheck: [], fresh: changed},
        )
        reader.phase = lambda _frame: "deploy"
        planner.verify_actions = True
        planner.adb.tap = lambda *args, **kwargs: (
            adb.taps.append(args) or ADBActionResult("succeeded", 0, 1)
        )
        state = RunState()

        first = planner.play(initial, state)
        second = planner.play(fresh, state, max_buys=0)

        self.assertEqual(first["bought"], [])
        self.assertEqual(second["reason"], "done")
        self.assertEqual(state.copies, {"knight": 1})
        self.assertEqual(state.purchases, ["knight"])
        self.assertFalse(state.sync_required)
        self.assertEqual(adb.taps, [(100, 100)])

    @patch("core.strategy.time.sleep")
    def test_unknown_purchase_keeps_last_observed_balance_in_final_result(self, _sleep):
        initial, after, recheck = object(), object(), object()
        planner, reader, adb, _ip = self.planner(
            frames=[after, recheck], elixir_by_frame={initial: 5, after: 0, recheck: 0},
            cards_by_frame={initial: [card("knight", 2)], after: [], recheck: []},
        )
        reader.phase = lambda _frame: "deploy"
        planner.verify_actions = True
        planner.adb.tap = lambda *args, **kwargs: adb.taps.append(args)
        state = RunState()

        result = planner.play(initial, state)

        self.assertEqual(result["elixir"], 0)
        self.assertEqual(state.copies, {})
        self.assertEqual(adb.taps, [(100, 100)])

    @patch("core.strategy.time.sleep")
    def test_unresolved_purchase_enters_limited_mode_without_copy_based_choices(self, _sleep):
        from core.adb_controller import ADBActionResult

        initial, after, recheck, fresh1, fresh2, fresh3, after_safe = [object() for _ in range(7)]
        original = [ShopCard("knight", 2, (100, 100), slot_index=0),
                    ShopCard("goblin_cage", 1, (200, 100), slot_index=1),
                    ShopCard("archer", 1, (300, 100), slot_index=2)]
        changed = [ShopCard("dark_prince", 2, (100, 100), slot_index=0),
                   ShopCard("goblin_cage", 1, (200, 100), slot_index=1),
                   ShopCard("boss_bandit", 3, (300, 100), slot_index=2)]
        rerolled = [ShopCard("dark_prince", 2, (100, 100), slot_index=0),
                    ShopCard("knight", 2, (200, 100), slot_index=1),
                    ShopCard("archer", 1, (300, 100), slot_index=2)]
        planner, reader, adb, _ip = self.planner(
            frames=[after, recheck, after_safe],
            elixir_by_frame={initial: 5, after: 4, recheck: 4, fresh1: 5, fresh2: 5,
                             fresh3: 5, after_safe: 2},
            cards_by_frame={initial: original, after: [], recheck: [], fresh1: changed,
                            fresh2: changed, fresh3: changed, after_safe: rerolled},
        )
        reader.phase = lambda _frame: "deploy"
        reader.own_units = lambda _frame: 2
        planner.verify_actions = True
        planner.adb.tap = lambda *args, **kwargs: (
            adb.taps.append(args) or ADBActionResult("succeeded", 0, 1)
        )
        state = RunState(copies={"knight": 1})

        first = planner.play(initial, state)
        second = planner.play(fresh1, state)
        third = planner.play(fresh2, state, max_buys=0)
        fourth = planner.play(fresh3, state, max_buys=1)

        self.assertEqual(first["reason"], "verification_pending")
        self.assertEqual(second["reason"], "verification_pending")
        self.assertTrue(state.safe_mode)
        self.assertTrue(state.sync_required)
        self.assertEqual(fourth["bought"], ["boss_bandit"])
        self.assertEqual(state.copies, {"knight": 1, "boss_bandit": 1})
        self.assertEqual(adb.taps, [(100, 100), (300, 100)])

    @patch("core.strategy.time.sleep")
    def test_verify_mode_refuses_unknown_phase_before_tapping(self, _sleep):
        initial = object()
        planner, reader, adb, _ip = self.planner(
            elixir_by_frame={initial: 4}, cards=[card("knight", 2)]
        )
        reader.phase = lambda _frame: None
        planner.verify_actions = True

        result = planner.play(initial, RunState())

        self.assertEqual(result["bought"], [])
        self.assertEqual(adb.taps, [])

    @patch("core.strategy.time.sleep")
    def test_shop_retry_stops_if_deploy_phase_ends(self, _sleep):
        initial, battle = object(), object()
        planner, reader, adb, _ip = self.planner(
            frames=[battle], elixir_by_frame={initial: 3, battle: 3},
            cards_by_frame={initial: [], battle: [card("knight", 2)]},
        )
        reader.phase = lambda frame: "battle" if frame is battle else "deploy"
        planner.verify_actions = True

        result = planner.play(initial, RunState())

        self.assertEqual(result["bought"], [])
        self.assertEqual(adb.taps, [])
        self.assertEqual(result["reason"], "phase_ended")

    @patch("core.strategy.time.sleep")
    def test_shop_retry_replaces_balance_with_value_from_same_fresh_frame(self, _sleep):
        from core.adb_controller import ADBActionResult

        initial, retry, after = object(), object(), object()
        selected = ShopCard("knight", 2, (100, 100), slot_index=0)
        replacement = ShopCard("archer", 2, (100, 100), slot_index=0)
        planner, reader, adb, _ip = self.planner(
            frames=[retry, after], elixir_by_frame={initial: 1, retry: 3, after: 1},
            cards_by_frame={initial: [], retry: [selected], after: [replacement]},
        )
        reader.phase = lambda _frame: "deploy"
        planner.verify_actions = True
        planner.adb.tap = lambda *args, **kwargs: (
            adb.taps.append(args) or ADBActionResult("succeeded", 0, 1)
        )

        result = planner.play(initial, RunState())

        self.assertEqual(result["bought"], ["knight"])
        self.assertEqual(adb.taps, [(100, 100)])

    @patch("core.strategy.time.sleep")
    def test_missing_post_purchase_shop_stays_unknown_and_marks_state_unsynced(self, _sleep):
        from core.adb_controller import ADBActionResult

        initial, after, recheck = object(), object(), object()
        selected = ShopCard("knight", 2, (100, 100), slot_index=0)
        planner, reader, adb, _ip = self.planner(
            frames=[after, recheck], elixir_by_frame={initial: 3, after: 1, recheck: 1},
            cards_by_frame={initial: [selected], after: [], recheck: []},
        )
        reader.phase = lambda _frame: "deploy"
        planner.verify_actions = True
        planner.adb.tap = lambda *args, **kwargs: (
            adb.taps.append(args) or ADBActionResult("succeeded", 0, 1)
        )
        state = RunState()

        result = planner.play(initial, state)

        self.assertEqual(result["bought"], [])
        self.assertTrue(state.sync_required)
        self.assertEqual(adb.taps, [(100, 100)])

    @patch("core.strategy.time.sleep")
    def test_unsynced_state_blocks_purchases_on_later_round(self, _sleep):
        initial = object()
        planner, _reader, adb, _ip = self.planner(
            elixir_by_frame={initial: 5}, cards=[card("knight", 2)]
        )
        planner.verify_actions = True
        state = RunState(sync_required=True)

        result = planner.play(initial, state)

        self.assertEqual(result["reason"], "state_sync_required")
        self.assertEqual(adb.taps, [])

    @patch("core.strategy.time.sleep")
    def test_failed_buy_attempts_to_sell_a_bench_unit(self, _sleep):
        planner, reader, adb, _ip = self.planner(
            frames=["initial"],
            elixir_by_frame={"initial": 3}, cards=[card("knight", 2)]
        )

        result = planner.play("initial", RunState())

        self.assertEqual(result["reason"], "bench full, could not sell")
        self.assertEqual(adb.taps[0], (100, 100))
        self.assertEqual(adb.taps[1:], [(210, 1280), (315, 1280), (410, 1280),
                                      (500, 1280), (585, 1280)])
        self.assertEqual(result["bought"], [])

    @patch("core.strategy.time.sleep")
    def test_sell_one_taps_the_first_slot_with_a_sell_button(self, _sleep):
        planner, _reader, adb, _ip = self.planner(
            frames=["selected-bench-frame"], sell_button=(800, 700)
        )

        sold = planner._sell_one("before-selection")

        self.assertTrue(sold)
        self.assertEqual(adb.taps, [(210, 1280), (800, 700)])
        self.assertEqual(adb.screenshots, 1)


class GameLogicDelegationTests(unittest.TestCase):
    def test_deploy_phase_passes_frame_and_run_state_to_planner(self):
        logic = GameLogic.__new__(GameLogic)
        logic.planner = Mock()
        logic.run_state = RunState(copies={"knight": 2})

        result = logic.deploy_phase("frame")

        logic.planner.play.assert_called_once_with("frame", logic.run_state)
        self.assertIs(result, logic.planner.play.return_value)


class BotLoopTests(unittest.TestCase):
    class StopLoop(BaseException):
        pass

    def test_orb_refresh_failure_returns_no_frame(self):
        bot = MergeTacticsBot.__new__(MergeTacticsBot)
        bot.adb = Mock()
        bot.adb.screenshot.side_effect = OSError("device offline")
        bot.game_logic = Mock()
        bot.game_logic.reader.orbs.return_value = [(10, 20)]
        bot.diagnostics = None

        result = bot.collect_orbs("source-frame")

        self.assertIsNone(result)
        bot.adb.tap.assert_called_once_with(10, 20)

    def test_loop_drops_old_frame_after_orb_refresh_failure(self):
        bot = MergeTacticsBot.__new__(MergeTacticsBot)
        bot.adb = Mock()
        bot.adb.screenshot.return_value = "old-frame"
        bot.image_processor = Mock()
        bot.game_logic = Mock()
        bot.game_logic.phase.return_value = "deploy"
        bot._deploy_played = False
        bot.handle_end_screen = Mock(return_value=False)
        bot.collect_orbs = Mock(return_value=None)
        bot.handle_deploy_phase = Mock()

        with patch("bot.time.sleep", side_effect=self.StopLoop):
            with self.assertRaises(self.StopLoop):
                bot.run(auto_search=False)

        bot.game_logic.phase.assert_called_once_with("old-frame")
        bot.handle_deploy_phase.assert_not_called()

    def test_stale_orb_observation_is_not_tapped(self):
        import time

        bot = MergeTacticsBot.__new__(MergeTacticsBot)
        bot.adb = Mock()
        bot.adb.screenshot.return_value = "fresh-frame"
        bot.game_logic = Mock()
        bot.game_logic.reader.orbs.return_value = [(10, 20)]
        bot.diagnostics = None
        bot.verify_actions = True
        bot.max_frame_age_seconds = 3

        refreshed = bot.collect_orbs("stale-frame", captured_at=time.monotonic() - 62)

        self.assertEqual(refreshed, "fresh-frame")
        bot.adb.tap.assert_not_called()

    def test_deploy_and_battle_frames_follow_the_existing_branches(self):
        bot = MergeTacticsBot.__new__(MergeTacticsBot)
        bot.adb = Mock()
        bot.adb.screenshot.side_effect = ["deploy-frame", "battle-frame", self.StopLoop()]
        bot.image_processor = Mock()
        bot.game_logic = Mock()
        bot.game_logic.phase.side_effect = ["deploy", "deploy", "battle", "battle"]
        bot.game_logic.reader.orbs.return_value = []
        bot._deploy_played = False
        bot.handle_end_screen = Mock(return_value=False)
        bot.collect_orbs = Mock(side_effect=["deploy-frame", "battle-frame"])
        bot.handle_deploy_phase = Mock(return_value={"elixir": 4, "reason": "done"})

        def stop_after_battle_wait(delay):
            if delay == BATTLE_WAIT:
                raise self.StopLoop()

        with patch("bot.time.sleep", side_effect=stop_after_battle_wait):
            with self.assertRaises(self.StopLoop):
                bot.run(auto_search=False)

        self.assertEqual(bot.game_logic.phase.call_args_list, [
            unittest.mock.call("deploy-frame"), unittest.mock.call("deploy-frame"),
            unittest.mock.call("battle-frame"), unittest.mock.call("battle-frame"),
        ])
        bot.handle_deploy_phase.assert_called_once_with("deploy-frame")
        self.assertEqual(bot.collect_orbs.call_count, 2)
        self.assertFalse(bot._deploy_played)


if __name__ == "__main__":
    unittest.main()
