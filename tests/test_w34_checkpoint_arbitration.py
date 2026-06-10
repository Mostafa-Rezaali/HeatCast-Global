from w34_checkpoint_arbitration import choose_checkpoint


def test_w34_arbitration_prioritizes_window_tac():
    rows = {
        "best_tac": {"window_tac": 0.10, "window_auc": 0.70},
        "best_monitor": {"window_tac": 0.12, "window_auc": 0.60},
    }
    selected, reason = choose_checkpoint(rows, 1e-4)
    assert selected == "best_monitor"
    assert "14-day-mean TAC" in reason


def test_w34_arbitration_uses_auc_when_tac_is_tied():
    rows = {
        "best_tac": {"window_tac": 0.10000, "window_auc": 0.65},
        "best_monitor": {"window_tac": 0.10005, "window_auc": 0.67},
    }
    selected, reason = choose_checkpoint(rows, 1e-4)
    assert selected == "best_monitor"
    assert "AUC" in reason
