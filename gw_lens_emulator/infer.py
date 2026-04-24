from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .evaluate import load_checkpoint, predict


def main() -> None:
    p = argparse.ArgumentParser(description="Infer F(omega,y) from a trained checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--omega", type=float, required=True)
    p.add_argument("--y", type=float, required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, norm, ckpt = load_checkpoint(args.checkpoint, device)
    F = predict(model, norm, np.array([args.omega]), np.array([args.y]), device=device)[0]
    print(json.dumps({"omega": args.omega, "y": args.y, "F_real": float(F.real), "F_imag": float(F.imag), "absF": float(abs(F))}, indent=2))


if __name__ == "__main__":
    main()
