# infer.py
import argparse
import numpy as np
import torch
from utils import load_models, normalize_mlp, normalize_siren, select_interval, amplification_factor, device

models = load_models()

# ——— Main: predict for a single (omega, y) ———
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict |F| for given ω and y using the appropriate surrogate model"
    )
    parser.add_argument("--omega", type=float, default=10.0, help="dimensionless frequency ω")
    parser.add_argument("--y", type=float, default=5.0, help="impact parameter y")

    args = parser.parse_args()

    om_val = args.omega
    y_val  = args.y

    cfg = select_interval(y_val)
    model = models[cfg["name"]]

    # normalize inputs
    if cfg["model_type"] == "mlp":
        om_n, y_n = normalize_mlp(np.array([om_val], dtype=np.float32), y_val)
    else:
        om_n, y_n = normalize_siren(np.array([om_val], dtype=np.float32), y_val, cfg)

    # inference
    X = torch.tensor(np.stack([om_n, y_n], axis=1),
                     dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(X).cpu().numpy()[0]
    Fpred = out[0] + 1j*out[1]
    amp_pred = abs(Fpred)

    # ground truth
    # Ftrue = amplification_factor(om_val, y_val)
    # amp_true = abs(Ftrue)

    # relative error in amplitude
    # rel_err = abs(amp_pred - amp_true) / amp_true

    # output results
    print(f"Model interval: {cfg['name']} ({cfg['model_type']})")
    print(f"ω = {om_val:.5f}, y = {y_val:.5f}")
    print(f"Predicted |F| = {amp_pred:.5e}")
    # print(f"True      |F| = {amp_true:.5e}")
    # print(f"Relative error = {rel_err:.2e}")
