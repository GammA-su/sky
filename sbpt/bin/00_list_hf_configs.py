"""List available Hugging Face model config names."""

from __future__ import annotations

from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from sbpt.utils.runtime import configure_runtime, setup_logging


def main() -> None:
    logger = setup_logging("sbpt.list_hf_configs")
    configure_runtime(logger=logger)
    names = sorted(CONFIG_MAPPING.keys())
    for name in names:
        print(name)
    logger.info("hf_configs=%s", len(names))


if __name__ == "__main__":
    main()
