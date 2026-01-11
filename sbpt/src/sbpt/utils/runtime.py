"""Runtime configuration and logging."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch


def setup_logging(name: str = "sbpt") -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    return logging.getLogger(name)


def _init_faiss_gpu(logger: logging.Logger) -> bool:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        logger.info(
            "faiss_gpu=false detail=import_failed error=%s hint=uv_add_faiss_gpu",
            exc,
        )
        return False
    try:
        if hasattr(faiss, "StandardGpuResources"):
            _ = faiss.StandardGpuResources()
            logger.info("faiss_gpu=true detail=resources_initialized")
            return True
        logger.info("faiss_gpu=false detail=no_gpu_resources")
        return False
    except Exception as exc:  # pragma: no cover - optional dep
        logger.info("faiss_gpu=false detail=init_failed error=%s", exc)
        return False


def configure_runtime(
    *,
    prefer_gpu: bool = True,
    cpu_threads: int = 16,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    if logger is None:
        logger = setup_logging()

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))

    try:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
    except RuntimeError:
        pass

    use_gpu = prefer_gpu and torch.cuda.is_available()
    if use_gpu:
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        logger.info(
            "runtime use_gpu=true device=cuda gpus=%s names=%s",
            gpu_count,
            ",".join(gpu_names) if gpu_names else "unknown",
        )
        _init_faiss_gpu(logger)
        device = torch.device("cuda")
    else:
        logger.info("runtime use_gpu=false device=cpu threads=%s", cpu_threads)
        device = torch.device("cpu")
    return {"device": device, "use_gpu": use_gpu}
