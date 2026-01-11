from sbpt.data.synth_traces import generate_addition_samples


def test_synth_deterministic() -> None:
    rows1 = generate_addition_samples(5, seed=42)
    rows2 = generate_addition_samples(5, seed=42)
    assert rows1 == rows2
    row = rows1[0]
    assert "prompt" in row and "completion" in row
    assert "state_ids" in row
    assert len(row["state_ids"]) == 3
