#!/usr/bin/env python
from __future__ import annotations

import argparse

from gw_lens_emulator.config import DEFAULT_OMEGA_POINTS, DEFAULT_Y_POINTS, OMEGA_BOUNDS, TRAIN_INTERVALS
from gw_lens_emulator.data import generate_interval_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Generate training data for PM or SIS lens emulators.")
    p.add_argument("--lens", choices=["pm", "sis"], required=True)
    p.add_argument("--interval", choices=sorted(TRAIN_INTERVALS), required=True)
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--n-y", type=int, default=None)
    p.add_argument("--n-omega", type=int, default=None)
    p.add_argument("--omega-min", type=float, default=OMEGA_BOUNDS[0])
    p.add_argument("--omega-max", type=float, default=OMEGA_BOUNDS[1])
    p.add_argument("--log-ratio", type=float, default=0.3)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    path = generate_interval_dataset(
        lens=args.lens,
        interval=args.interval,
        output_dir=args.output_dir,
        n_y=args.n_y or DEFAULT_Y_POINTS[args.interval],
        n_omega=args.n_omega or DEFAULT_OMEGA_POINTS[args.interval],
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        log_ratio=args.log_ratio,
        overwrite=args.overwrite,
    )
    print(path)


if __name__ == "__main__":
    main()
