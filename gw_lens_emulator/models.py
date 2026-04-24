from __future__ import annotations

import math
from typing import Sequence, Union

import torch
import torch.nn as nn

Scale = Union[float, Sequence[float]]


class FourierFeature(nn.Module):
    """Fixed Gaussian Fourier feature mapping for coordinate inputs."""

    def __init__(self, input_dim: int, mapping_size: int = 64, scale: Scale = 10.0, seed: int | None = 42):
        super().__init__()
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        B = torch.randn(mapping_size, input_dim, generator=generator)
        if isinstance(scale, (list, tuple)):
            if len(scale) != input_dim:
                raise ValueError("When scale is a list/tuple, len(scale) must equal input_dim")
            scale_tensor = torch.tensor(scale, dtype=torch.float32).view(1, input_dim)
            B = B * scale_tensor
        else:
            B = B * float(scale)
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2.0 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Sine(nn.Module):
    def __init__(self, w0: float = 80.0):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SIRENLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, w0: float = 80.0):
        super().__init__()
        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.w0 = float(w0)
        self.linear = nn.Linear(in_features, out_features)
        self.activation = Sine(w0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class SIRENModel(nn.Module):
    """Fourier-feature SIREN emulator for Re(F) and Im(F)."""

    def __init__(
        self,
        in_dim: int = 2,
        fourier_feats: int = 64,
        hidden_dim: int = 512,
        out_dim: int = 2,
        depth: int = 8,
        w0: float = 80.0,
        scale: Scale = (30.0, 0.5),
        fourier_seed: int | None = 42,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.ff = FourierFeature(in_dim, mapping_size=fourier_feats, scale=scale, seed=fourier_seed)
        current_dim = fourier_feats * 2
        layers: list[nn.Module] = [SIRENLayer(current_dim, hidden_dim, is_first=True, w0=w0)]
        for _ in range(depth - 1):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, w0=w0))
        final = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / w0
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.ff(x))
