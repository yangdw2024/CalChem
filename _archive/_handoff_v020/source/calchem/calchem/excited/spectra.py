from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class Spectrum:
    def __init__(
        self,
        energies: np.ndarray,
        intensities: np.ndarray,
        units: str = "ev",
    ):
        self.energies = np.array(energies)
        self.intensities = np.array(intensities)
        self.units = units

    def broaden(
        self,
        method: str = "gaussian",
        sigma: float = 0.1,
        n_points: int = 1000,
        x_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if x_range is None:
            x_min = self.energies.min() - 5 * sigma
            x_max = self.energies.max() + 5 * sigma
        else:
            x_min, x_max = x_range

        x = np.linspace(x_min, x_max, n_points)
        y = np.zeros_like(x)

        if method == "gaussian":
            for e, i in zip(self.energies, self.intensities):
                y += i * np.exp(-((x - e) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        elif method == "lorentzian":
            for e, i in zip(self.energies, self.intensities):
                y += i * (sigma / (2 * np.pi)) / ((x - e) ** 2 + (sigma / 2) ** 2)
        else:
            raise ValueError(f"Unknown broadening method: {method}")

        return x, y

    def __repr__(self) -> str:
        return f"Spectrum(n_peaks={len(self.energies)}, units='{self.units}')"
