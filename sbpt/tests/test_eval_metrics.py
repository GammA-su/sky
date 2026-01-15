from sbpt.eval.metrics import equiv_consistency, exact_match, expected_calibration_error, format_strict_pass


def test_exact_match() -> None:
    preds = ["a", "b", "c"]
    targets = ["a", "x", "c"]
    assert exact_match(preds, targets) == 2 / 3


def test_ece_basic() -> None:
    confidences = [0.9, 0.2, 0.8, 0.1]
    correct = [True, False, True, False]
    ece = expected_calibration_error(confidences, correct, n_bins=2)
    assert ece >= 0.0


def test_format_strict_pass() -> None:
    assert format_strict_pass('{"a":1}')
    assert format_strict_pass(' { "a" : 1 } ')
    assert not format_strict_pass('{"a":1} trailing')


def test_equiv_consistency() -> None:
    outputs = ["x", "x", "y", "z"]
    equiv_ids = [1, 1, 2, 2]
    assert equiv_consistency(outputs, equiv_ids) == 0.5
