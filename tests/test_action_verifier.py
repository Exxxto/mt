import unittest

from unittest.mock import patch

from core.action_verifier import ActionStatus, ActionVerifier, Observation, VerificationBudget


class ActionVerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = ActionVerifier()

    def test_purchase_confirms_balance_and_shop_change_together(self):
        result = self.verifier.verify_purchase(
            action_id="a1", adb_status="succeeded", frame_before=1, frame_after=2,
            before=Observation(phase="deploy", elixir=5, shop_cards=((0, "knight", 2),)),
            after=Observation(phase="deploy", elixir=3, shop_cards=((0, "archer", 1),)),
            cost=2,
        )
        self.assertEqual(result.status, ActionStatus.CONFIRMED)

    def test_purchase_is_unknown_when_balance_is_not_visual(self):
        result = self.verifier.verify_purchase(
            action_id="a2", adb_status="succeeded", frame_before=1, frame_after=2,
            before=Observation(phase="deploy", elixir=None, shop_cards=((0, "knight", 2),)),
            after=Observation(phase="deploy", elixir=3, shop_cards=((0, "archer", 1),)),
            cost=2,
        )
        self.assertEqual(result.status, ActionStatus.UNKNOWN)

    def test_phase_change_interrupts_purchase(self):
        result = self.verifier.verify_purchase(
            action_id="a3", adb_status="succeeded", frame_before=1, frame_after=2,
            before=Observation(phase="deploy", elixir=5),
            after=Observation(phase="battle", elixir=3), cost=2,
        )
        self.assertEqual(result.status, ActionStatus.INTERRUPTED)

    def test_unchanged_balance_and_shop_means_failed(self):
        observation = Observation(phase="deploy", elixir=5,
                                  shop_cards=((0, "knight", 2), (1, "archer", 1), (2, "goblin", 1)),
                                  selected_slot=(0, ("knight", 2)))
        result = self.verifier.verify_purchase(
            action_id="a4", adb_status="succeeded", frame_before=1, frame_after=2,
            before=observation, after=observation, cost=2,
        )
        self.assertEqual(result.status, ActionStatus.FAILED)

    def test_unexpected_elixir_delta_with_changed_shop_is_unknown(self):
        result = self.verifier.verify_purchase(
            action_id="a5", adb_status="succeeded", frame_before=1, frame_after=2,
            before=Observation(phase="deploy", elixir=5, shop_cards=((0, "knight", 2),)),
            after=Observation(phase="deploy", elixir=4, shop_cards=((0, "archer", 1),)), cost=2,
        )
        self.assertEqual(result.status, ActionStatus.UNKNOWN)

    def test_adb_failure_is_separate_from_game_confirmation(self):
        result = self.verifier.verify_purchase(
            action_id="a6", adb_status="timeout", frame_before=1, frame_after=2,
            before=Observation(phase="deploy", elixir=5),
            after=Observation(phase="deploy", elixir=3), cost=2,
        )
        self.assertEqual(result.status, ActionStatus.UNKNOWN)
        self.assertEqual(result.adb_status, "timeout")

    def test_missing_shop_observation_is_unknown_not_failed(self):
        result = self.verifier.verify_purchase(
            action_id="a7", adb_status="succeeded", frame_before=1, frame_after=2,
            before=Observation(phase="deploy", elixir=5, shop_cards=None),
            after=Observation(phase="deploy", elixir=5, shop_cards=None), cost=2,
        )
        self.assertEqual(result.status, ActionStatus.UNKNOWN)

    def test_same_slot_index_with_new_card_content_confirms_change(self):
        result = self.verifier.verify_purchase(
            action_id="a8", adb_status="succeeded", frame_before=1, frame_after=2,
            before=Observation(phase="deploy", elixir=5, shop_cards=None,
                               selected_slot=(1, ("knight", 2))),
            after=Observation(phase="deploy", elixir=3, shop_cards=None,
                              selected_slot=(1, ("archer", 1))),
            cost=2,
        )
        self.assertEqual(result.status, ActionStatus.CONFIRMED)

    def test_slot_recognition_loss_is_not_slot_change(self):
        result = self.verifier.verify_purchase(
            action_id="a9", adb_status="succeeded", frame_before=1, frame_after=2,
            before=Observation(phase="deploy", elixir=5, shop_cards=None,
                               selected_slot=(1, ("knight", 2))),
            after=Observation(phase="deploy", elixir=3, shop_cards=None,
                              selected_slot=None),
            cost=2,
        )
        self.assertEqual(result.status, ActionStatus.UNKNOWN)

    def test_verification_budget_shrinks_and_expires(self):
        with patch("core.action_verifier.time.monotonic", side_effect=[10.0, 10.25, 11.0]):
            budget = VerificationBudget(1.0)
            self.assertAlmostEqual(budget.remaining(), 0.75)
            self.assertEqual(budget.remaining(), 0.0)


if __name__ == "__main__":
    unittest.main()
