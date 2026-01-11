from sbpt.data.enumerators import enumerate_hypotheses


def test_enumerator_basic() -> None:
    example = {"prompt": "Add 2 and 3. Answer:"}
    hypotheses, correct = enumerate_hypotheses(example, max_h=4)
    assert len(hypotheses) <= 4
    assert len(hypotheses) == len(correct)
    assert any(correct)
