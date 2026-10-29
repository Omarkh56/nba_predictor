"""The learned game model must also beat hand-tuned on the latest held-out season.

Retraining in the 2026-27 configuration won on 2024-25 validation (log loss
0.6014 vs 0.6113) but lost on the 2025-26 test season (0.6010 vs 0.5977), and
the deployed model also lost on 2025-26 (0.6065). Validation alone said "swap".
"""

from unittest import mock

import pytest


def test_apply_test_gate_vetoes_a_swap_that_lost_on_test():
    import train_model as tm

    assert tm.apply_test_gate("swap", 0.6010, 0.2067, 0.5977, 0.2034) == ("monitor", "failed")
    assert tm.apply_test_gate("swap", 0.5960, 0.2020, 0.5977, 0.2034) == ("swap", "passed")
    assert tm.apply_test_gate("swap", None, None, None, None) == ("monitor", "not_run")
    assert tm.apply_test_gate("keep_hand_tuned", 0.5, 0.2, 0.6, 0.21) == ("keep_hand_tuned", "passed")


@pytest.fixture
def nb(monkeypatch):
    with mock.patch(
        "requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")
    ):
        import newnbapredictor

    monkeypatch.setattr(newnbapredictor, "validation_season", lambda *a, **k: "2025-26")
    return newnbapredictor


BEAT = {"test_log_loss": 0.5960, "test_brier": 0.2020,
        "ht_test_log_loss": 0.5977, "ht_test_brier": 0.2034}


def test_gate_requires_the_latest_completed_season(nb):
    ok, reason = nb._held_out_gate({"tested_on": "2024-25"}, BEAT)
    assert not ok and "2025-26" in reason
    assert nb._held_out_gate({"tested_on": "2025-26"}, BEAT)[0]


def test_gate_rejects_a_model_that_lost_on_test(nb):
    lost = {**BEAT, "test_log_loss": 0.6010, "test_brier": 0.2067}
    ok, reason = nb._held_out_gate({"tested_on": "2025-26"}, lost)
    assert not ok and "lost" in reason


def test_gate_fails_closed_without_metrics(nb):
    assert not nb._held_out_gate({"tested_on": "2025-26"}, {})[0]
