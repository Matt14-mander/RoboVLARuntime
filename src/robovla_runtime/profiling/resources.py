"""Optional resource samplers with no import-time accelerator dependency."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class MemorySampler(Protocol):
    def begin_sample(self) -> None: ...

    def snapshot(self) -> Mapping[str, int | float | str | bool | None]: ...


class NullMemorySampler:
    def begin_sample(self) -> None:
        return None

    def snapshot(self) -> Mapping[str, int | float | str | bool | None]:
        return {}


class CudaMemorySampler:
    """Read current and per-request peak CUDA memory from PyTorch."""

    def __init__(self, device: Any = None) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory profiling requires torch.cuda")
        self.torch = torch
        self.device = torch.cuda.current_device() if device is None else device

    def begin_sample(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self) -> Mapping[str, int | float | str | bool | None]:
        cuda = self.torch.cuda
        return {
            "cuda_memory_allocated_bytes": int(cuda.memory_allocated(self.device)),
            "cuda_memory_reserved_bytes": int(cuda.memory_reserved(self.device)),
            "cuda_peak_memory_allocated_bytes": int(
                cuda.max_memory_allocated(self.device)
            ),
            "cuda_peak_memory_reserved_bytes": int(
                cuda.max_memory_reserved(self.device)
            ),
        }


class CudaEventTimer:
    """Measure work submitted to the current PyTorch CUDA stream."""

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA event timing requires torch.cuda")
        self.torch = torch

    def start(self) -> tuple[Any, Any]:
        started = self.torch.cuda.Event(enable_timing=True)
        finished = self.torch.cuda.Event(enable_timing=True)
        started.record()
        return started, finished

    def stop(self, token: tuple[Any, Any]) -> None:
        token[1].record()

    def elapsed_s(self, token: tuple[Any, Any]) -> float:
        return float(token[0].elapsed_time(token[1])) / 1000.0
