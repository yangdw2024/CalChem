"""Native two-body Grimme D3(BJ)/D3(0), coordinates in Bohr.

Reference-state CN interpolation and gradients are evaluated by Rust.
No approximate Python fallback; ATM three-body dispersion is not included.
"""
from __future__ import annotations

import numpy as np


def _evaluate(symbols, coordinates, functional, s6, s8, a1, a2, damping):
    if any(v is not None for v in (s6, s8, a1, a2)):
        raise ValueError(
            "Custom D3 parameters are not supported by this wrapper; "
            "use a documented named functional and damping scheme."
        )
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.shape != (len(symbols), 3) or not np.isfinite(xyz).all():
        raise ValueError("D3 coordinates must be finite and shaped (n_atoms, 3), in Bohr")
    from .. import _core

    return _core.run_d3_dispersion(list(symbols), xyz.tolist(), functional, damping)


def compute_d3_energy(symbols, coordinates, functional='pbe', s6=None, s8=None,
                      a1=None, a2=None, damping='zero') -> float:
    """Return two-body D3 energy in Hartree; invalid inputs raise ValueError."""
    return float(_evaluate(symbols, coordinates, functional, s6, s8, a1, a2,
                           damping).energy_hartree)


def compute_d3_gradient(symbols, coordinates, functional='pbe', s6=None, s8=None,
                        a1=None, a2=None, damping='zero') -> np.ndarray:
    """Return the analytic two-body D3 nuclear derivative in Hartree/Bohr."""
    return np.asarray(_evaluate(symbols, coordinates, functional, s6, s8, a1, a2,
                                damping).gradient, dtype=np.float64)
