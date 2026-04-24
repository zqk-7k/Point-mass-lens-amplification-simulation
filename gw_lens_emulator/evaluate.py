from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import OMEGA_BOUNDS, TRAIN_INTERVALS, Normalization
from .data import evaluate_lens
from .models import SIRENModel


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = SIRENModel(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm = Normalization(**ckpt["normalization"])
    return model, norm, ckpt


def predict(model: SIRENModel, norm: Normalization, omega: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 65536) -> np.ndarray:
    om_n, y_n = norm.normalize(omega, y)
    x = np.stack([om_n, y_n], axis=1).astype(np.float32)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            outs.append(model(xb).cpu().numpy())
    out = np.vstack(outs)
    return out[:, 0] + 1j * out[:, 1]


def evaluate_grid(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, norm, ckpt = load_checkpoint(args.checkpoint, device)
    omegas = np.linspace(args.omega_min, args.omega_max, args.n_omega, dtype=np.float64)
    ys = np.linspace(args.y_min, args.y_max, args.n_y, dtype=np.float64)

    all_err = []
    for yy in ys:
        omega_vec = omegas
        y_vec = np.full_like(omega_vec, yy)
        pred = predict(model, norm, omega_vec, y_vec, device=device, batch_size=args.batch_size)
        true = evaluate_lens(args.lens, omega_vec, float(yy))
        err = np.abs(np.abs(pred) - np.abs(true)) / (np.abs(true) + 1e-12)
        all_err.append(err)
    err = np.concatenate(all_err)
    result = {
        "checkpoint": str(args.checkpoint),
        "lens": args.lens,
        "n_points": int(err.size),
        "mean_relative_absF_error": float(np.mean(err)),
        "median_relative_absF_error": float(np.median(err)),
        "p95_relative_absF_error": float(np.quantile(err, 0.95)),
        "max_relative_absF_error": float(np.max(err)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint against reference lens calculations.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lens", choices=["pm", "sis"], required=True)
    p.add_argument("--interval", choices=sorted(TRAIN_INTERVALS), default=None)
    p.add_argument("--y-min", type=float, default=None)
    p.add_argument("--y-max", type=float, default=None)
    p.add_argument("--omega-min", type=float, default=OMEGA_BOUNDS[0])
    p.add_argument("--omega-max", type=float, default=OMEGA_BOUNDS[1])
    p.add_argument("--n-y", type=int, default=20)
    p.add_argument("--n-omega", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--device", default=None)
    p.add_argument("--output-json", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.interval and (args.y_min is None or args.y_max is None):
        args.y_min, args.y_max = TRAIN_INTERVALS[args.interval]
    if args.y_min is None or args.y_max is None:
        raise SystemExit("Provide --interval or both --y-min and --y-max.")
    result = evaluate_grid(args)
    print(json.dumps(result, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
