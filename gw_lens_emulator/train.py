from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from .config import OMEGA_BOUNDS, TRAIN_INTERVALS, Normalization
from .data import AmplificationDataset
from .models import SIRENModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_scale(values: list[float]) -> float | list[float]:
    if len(values) == 1:
        return values[0]
    return values


def train_from_args(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    y_min, y_max = TRAIN_INTERVALS[args.interval]
    norm = Normalization(OMEGA_BOUNDS[0], OMEGA_BOUNDS[1], y_min, y_max)
    dataset = AmplificationDataset(args.data, norm)

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = SIRENModel(
        fourier_feats=args.fourier_feats,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        w0=args.w0,
        scale=parse_scale(args.scale),
        fourier_seed=args.seed,
    ).to(device)
    if torch.cuda.device_count() > 1 and args.data_parallel:
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    criterion = nn.MSELoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = vars(args).copy()
    metadata.update({"normalization": norm.__dict__, "model": "FourierFeature+SIREN"})
    (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item() * x_batch.size(0)
        scheduler.step()
        train_loss = total / len(train_set)

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    val_total += criterion(model(x_batch), y_batch).item() * x_batch.size(0)
            val_loss = val_total / len(val_set)
            print(f"epoch={epoch:06d} train_mse={train_loss:.6e} val_mse={val_loss:.6e} lr={scheduler.get_last_lr()[0]:.3e}")
            state_model = model.module if isinstance(model, nn.DataParallel) else model
            ckpt = {
                "epoch": epoch,
                "model_state_dict": state_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "normalization": norm.__dict__,
                "model_kwargs": {
                    "fourier_feats": args.fourier_feats,
                    "hidden_dim": args.hidden_dim,
                    "depth": args.depth,
                    "w0": args.w0,
                    "scale": parse_scale(args.scale),
                    "fourier_seed": args.seed,
                },
            }
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, out_dir / "best.pt")
            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save(ckpt, out_dir / f"epoch_{epoch:06d}.pt")

    return out_dir / "best.pt"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a SIREN emulator for F(omega,y).")
    p.add_argument("--data", required=True, help="Path to npy file with omega,y,Re(F),Im(F).")
    p.add_argument("--interval", required=True, choices=sorted(TRAIN_INTERVALS))
    p.add_argument("--output-dir", default="runs/debug")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--fourier-feats", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--w0", type=float, default=80.0)
    p.add_argument("--scale", type=float, nargs="+", default=[30.0, 0.5])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--data-parallel", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    best_path = train_from_args(args)
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
