from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.exceptions import GPUNotAvailableError


_gpu_available = None


def is_gpu_available() -> bool:
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available

    try:
        import cupy

        _gpu_available = True
    except ImportError:
        _gpu_available = False

    return _gpu_available


class GPUFockBuilder:
    def __init__(self, mol, basis):
        self.mol = mol
        self.basis = basis

        if not is_gpu_available():
            raise GPUNotAvailableError(
                "GPU acceleration requires cupy. Install with: pip install cupy-cuda12x"
            )

    def build(self, dm: np.ndarray) -> np.ndarray:
        raise NotImplementedError("GPU Fock build not implemented yet")

    def __repr__(self) -> str:
        return f"GPUFockBuilder(mol={self.mol})"
