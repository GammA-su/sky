from sbpt.eval.metrics import exact_match, expected_calibration_error


def test_exact_match() -> None:
    preds = ["a", "b", "c"]
    targets = ["a", "x", "c"]
    assert exact_match(preds, targets) == 2 / 3


def test_ece_basic() -> None:
    confidences = [0.9, 0.2, 0.8, 0.1]
    correct = [True, False, True, False]
    ece = expected_calibration_error(confidences, correct, n_bins=2)
    assert ece >= 0.0
