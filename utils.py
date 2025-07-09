#!/usr/bin/env python3
import os
import argparse
import numpy as np
import mpmath
import torch
import torch.nn as nn
from model import SIRENModel

# ——— Configuration ———
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— MLP definition ———
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=1024, out_dim=2, depth=8):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ——— Analytic amplification factor ———
def amplification_factor(om, y):
    om_mp = mpmath.mpf(om)
    y_mp  = mpmath.mpf(y)
    xm    = (y_mp + mpmath.sqrt(y_mp**2 + 4)) / 2
    phi   = (xm - y_mp)**2/2 - mpmath.log(xm)
    phase = mpmath.pi*om_mp/4 + 1j*(om_mp/2)*(mpmath.log(om_mp/2) - 2*phi)
    g     = mpmath.gamma(1 - 1j*om_mp/2)
    h     = mpmath.hyp1f1(1j*om_mp/2, 1, 1j*om_mp*y_mp**2/2)
    return complex(mpmath.exp(phase) * g * h)

# ——— Interval definitions ———
intervals = [
    {"name":"I0", "y_min":0.05, "y_max":0.2,
     "om_min":0.00670904, "om_max":43.9322,
     "checkpoint":"checkpoints/model0.pth",
     "model_type":"mlp"},
    {"name":"I1", "y_min":0.2, "y_max":1.0,
     "om_min":0.00670904, "om_max":43.9322,
     "checkpoint":"checkpoints/model1.pth",
     "model_type":"siren"},
    {"name":"I2", "y_min":1.0, "y_max":3.0,
     "om_min":0.00670904, "om_max":43.9322,
     "checkpoint":"checkpoints/model2.pth",
     "model_type":"siren"},
    {"name":"I3", "y_min":3.0, "y_max":6.0,
     "om_min":0.00670904, "om_max":43.9322,
     "checkpoint":"checkpoints/model3.pth",
     "model_type":"siren"},
    {"name":"I4", "y_min":6.0, "y_max":10.0,
     "om_min":0.00670904, "om_max":43.9322,
     "checkpoint":"checkpoints/model4.pth",
     "model_type":"siren"},
]
# attach index for normalization branching
for idx, cfg in enumerate(intervals, start=1):
    cfg['idx'] = idx

# # ——— Load all models ———
# models = {}
# for cfg in intervals:
#     if cfg["model_type"] == "mlp":
#         net = MLP().to(DEVICE)
#     else:
#         net = SIRENModel(in_dim=2, fourier_feats=20,
#                          hidden_dim=512, out_dim=2,
#                          depth=8, w0=80.0).to(DEVICE)
#     state = torch.load(cfg["checkpoint"], map_location=DEVICE)
#     net.load_state_dict(state)
#     net.eval()
#     models[cfg["name"]] = net

# Load models once
def load_models():
    models = {}
    for cfg in intervals:
        if cfg['model_type'] == 'mlp':
            net = MLP().to(device)
        else:
            net = SIRENModel(in_dim=2, fourier_feats=20,
                             hidden_dim=512, out_dim=2,
                             depth=8, w0=80.0).to(device)
        state = torch.load(cfg['checkpoint'], map_location=device)
        net.load_state_dict(state)
        net.eval()
        models[cfg['name']] = net
    return models

# ——— Normalization functions ———
def normalize_mlp(om, y):
    om_min, om_max = 0.00670904, 47.9322
    y_min,  y_max  = 0.04,       0.22
    om_n = 2*(om - om_min)/(om_max - om_min) - 1
    y_scalar = 2*(y - y_min)/(y_max - y_min) - 1
    # 将标量 y_n 扩展到与 om_n 同样的形状
    y_n = np.full_like(om_n, y_scalar, dtype=om_n.dtype)
    return om_n, y_n


def normalize_siren(om, y, cfg):
    # same logic as in predict_combined_rand_new.py
    om_n = 2*(om - cfg["om_min"])/(cfg["om_max"] - cfg["om_min"]) - 1
    if cfg['idx'] == 2:
        # first SIREN subdomain uses [0.05,3.0] for y_norm
        y_scalar = 2*(y - 0.05)/(3.0 - 0.05) - 1
    else:
        y_scalar = 2*(y - cfg["y_min"])/(cfg["y_max"] - cfg["y_min"]) - 1
    y_n = np.full_like(om_n, y_scalar, dtype=om_n.dtype)
    return om_n, y_n

# ——— Interval selector ———
def select_interval(y_val):
    # left-closed, right-open except last interval
    for cfg in intervals:
        if y_val >= cfg["y_min"] and (y_val < cfg["y_max"] or cfg['idx'] == len(intervals)):
            return cfg
    raise ValueError(f"y={y_val} outside all intervals")