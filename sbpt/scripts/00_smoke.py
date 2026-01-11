"""Smoke test for SBPT."""

from __future__ import annotations

import torch

from sbpt.data.schemas import ByteTokenizer
from sbpt.losses.lm import compute_lm_loss
from sbpt.losses.state_transition import compute_state_losses
from sbpt.model.sbpt import SBPTConfig, SBPTModel
from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.smoke")
    runtime = configure_runtime(logger=logger)
    device = runtime["device"]
    logger.info("smoke_start=true")
    tokenizer = ByteTokenizer()
    config = SBPTConfig(vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=2, d_ff=128, n_state_slots=4)
    model = SBPTModel(config).to(device)
    model.eval()

    text = "Add 12 and 3. Answer: 15"
    input_ids = torch.tensor([tokenizer.encode(text, add_bos=True, add_eos=True)], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention, metadata=None)
        lm_loss = compute_lm_loss(outputs["logits"], labels)

        state_logits = outputs["aux"].get("state_logits")
        state_targets = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        state_losses = compute_state_losses(state_logits, state_targets)

        logger.info("lm_loss=%s", float(lm_loss.detach().cpu()))
        logger.info("state_loss=%s", float(state_losses["loss_state"].detach().cpu()))
    logger.info("smoke_done=true")


if __name__ == "__main__":
    main()
