"""Point-mass lens amplification factor."""
from __future__ import annotations

import numpy as np


def amplification_factor(omega, y):
    """Vectorized point-mass-lens amplification factor F(omega, y).

    Uses SciPy for fast array evaluation. Inputs can be scalars or
    NumPy-broadcastable arrays.
    """
    from scipy import special

    om, yy = np.broadcast_arrays(np.asarray(omega, dtype=np.float64), np.asarray(y, dtype=np.float64))
    if np.any(om <= 0):
        raise ValueError("omega must be positive")
    if np.any(yy < 0):
        raise ValueError("y must be non-negative")

    x_m = (yy + np.sqrt(yy**2 + 4.0)) / 2.0
    phi_m = 0.5 * (x_m - yy) ** 2 - np.log(x_m)
    phase = np.pi * om / 4.0 + 1j * (om / 2.0) * (np.log(om / 2.0) - 2.0 * phi_m)
    gamma_term = special.gamma(1.0 - 1j * om / 2.0)
    hyp_term = special.hyp1f1(1j * om / 2.0, 1.0, 1j * om * yy**2 / 2.0)
    out = np.exp(phase) * gamma_term * hyp_term
    if out.shape == ():
        return complex(out)
    return out.astype(np.complex128)


def amplification_factor_scalar(omega: float, y: float) -> complex:
    return complex(amplification_factor(float(omega), float(y)))
