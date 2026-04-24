"""Singular Isothermal Sphere (SIS) amplification factor.

This project uses the semi-analytic SIS evaluator from GLoW when available,
which matches the original experimental scripts. Install GLoW separately and
then call ``sis.amplification_factor``.
"""
from __future__ import annotations

import numpy as np


def _load_glow_sis():
    try:
        from glow.freq_domain import Fw_SemiAnalyticSIS  # type: ignore
        return Fw_SemiAnalyticSIS
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "SIS data generation requires GLoW with glow.freq_domain.Fw_SemiAnalyticSIS. "
            "Install GLoW in the active environment, or use lens=pm for the built-in point-mass evaluator."
        ) from exc


def amplification_factor(omega, y: float, psi0: float = 1.0):
    """Return SIS F(omega, y) using GLoW's semi-analytic implementation."""
    Fw_SemiAnalyticSIS = _load_glow_sis()
    sis_fw = Fw_SemiAnalyticSIS(y=float(y), psi0=float(psi0))
    return sis_fw(np.asarray(omega, dtype=float))
