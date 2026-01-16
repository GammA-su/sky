from sbpt.data.synth_traces import generate_add_carry_trace_samples, generate_addition_samples


def test_synth_deterministic() -> None:
    rows1 = generate_addition_samples(5, seed=42)
    rows2 = generate_addition_samples(5, seed=42)
    assert rows1 == rows2
    row = rows1[0]
    assert "prompt" in row and "completion" in row
    assert "state_ids" in row
    assert len(row["state_ids"]) == 3


def test_add_carry_trace_deterministic() -> None:
    max_digits = 9
    rows1 = generate_add_carry_trace_samples(5, seed=7, min_digits=6, max_digits=max_digits)
    rows2 = generate_add_carry_trace_samples(5, seed=7, min_digits=6, max_digits=max_digits)
    assert rows1 == rows2
    for row in rows1:
        assert row["task_type"] == "add_carry_trace"
        state_ids = row["state_ids"]
        assert state_ids
        assert len(state_ids) == max_digits
        assert all(0 <= state_id <= (2 * max_digits - 1) for state_id in state_ids)
