from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_OMEGA_POINTS, DEFAULT_Y_POINTS, OMEGA_BOUNDS, TRAIN_INTERVALS, Normalization
from .physics import point_mass, sis

LensName = Literal["pm", "sis"]


def omega_grid(n_omega: int, omega_min: float, omega_max: float, log_ratio: float = 0.3, split: float = 1.0) -> np.ndarray:
    """Hybrid grid: logarithmic at low frequency and linear at high frequency."""
    if n_omega < 2:
        raise ValueError("n_omega must be >= 2")
    n_log = max(1, int(log_ratio * n_omega))
    n_lin = n_omega - n_log
    w_log = np.logspace(np.log10(omega_min), np.log10(split), n_log, dtype=np.float64)
    w_lin = np.linspace(split, omega_max, n_lin, dtype=np.float64)
    return np.unique(np.concatenate([w_log, w_lin]))


def y_grid(interval: str, n_y: int, y_min: float, y_max: float) -> np.ndarray:
    """Default y sampling used in the paper experiments."""
    if interval == "I1":
        return np.logspace(np.log10(y_min), np.log10(y_max), n_y, dtype=np.float64)
    return np.linspace(y_min, y_max, n_y, dtype=np.float64)


def evaluate_lens(lens: LensName, omega_values: np.ndarray, y_value: float) -> np.ndarray:
    if lens == "pm":
        return point_mass.amplification_factor(omega_values, y_value)
    if lens == "sis":
        return sis.amplification_factor(omega_values, y_value)
    raise ValueError(f"Unsupported lens: {lens}")


def generate_interval_dataset(
    lens: LensName,
    interval: str,
    output_dir: str | Path,
    n_y: int | None = None,
    n_omega: int | None = None,
    omega_min: float = OMEGA_BOUNDS[0],
    omega_max: float = OMEGA_BOUNDS[1],
    log_ratio: float = 0.3,
    overwrite: bool = False,
) -> Path:
    """Generate [omega, y, Re(F), Im(F)] data for one interval."""
    if interval not in TRAIN_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(TRAIN_INTERVALS)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_y = int(n_y or DEFAULT_Y_POINTS[interval])
    n_omega = int(n_omega or DEFAULT_OMEGA_POINTS[interval])
    y_min, y_max = TRAIN_INTERVALS[interval]
    save_path = output_dir / f"{lens}_{interval}_y{n_y}_w{n_omega}.npy"
    if save_path.exists() and not overwrite:
        return save_path

    omegas = omega_grid(n_omega, omega_min, omega_max, log_ratio=log_ratio)
    ys = y_grid(interval, n_y, y_min, y_max)
    blocks = []
    for y in ys:
        F = evaluate_lens(lens, omegas, float(y))
        block = np.column_stack([
            omegas.astype(np.float32),
            np.full_like(omegas, float(y), dtype=np.float32),
            np.real(F).astype(np.float32),
            np.imag(F).astype(np.float32),
        ])
        blocks.append(block)
    data = np.vstack(blocks).astype(np.float32)
    np.save(save_path, data)
    return save_path


class AmplificationDataset(Dataset):
    """Dataset backed by an npy file with columns omega, y, Re(F), Im(F)."""

    def __init__(self, npy_path: str | Path, norm: Normalization):
        data = np.load(npy_path)
        omega_n, y_n = norm.normalize(data[:, 0], data[:, 1])
        self.x = torch.tensor(np.stack([omega_n, y_n], axis=1), dtype=torch.float32)
        self.y = torch.tensor(data[:, 2:4], dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
