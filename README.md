# Gravitational-Lens Amplification Factor Emulator

This repository contains reproducible code for the paper **"Efficient Evaluation of Gravitational Lensing Amplification Factors: A Deep Learning Framework"**.  It trains coordinate-based neural emulators for the complex wave-optics amplification factor

\[
F(\omega, y) = \Re F(\omega,y) + i\Im F(\omega,y),
\]

where `omega` is the dimensionless frequency and `y` is the dimensionless source--lens impact parameter.

The current version supports two lens models:

- **PM / PML**: point-mass lens, evaluated with the closed-form expression using `mpmath`.
- **SIS**: singular isothermal sphere lens, evaluated through GLoW's semi-analytic SIS implementation when GLoW is installed.

The neural model is a **Fourier-feature SIREN** that maps normalized coordinates `(omega, y)` to `[Re(F), Im(F)]`.

---

## Repository contents

```text
gw-lens-emulator-repro/
├── gw_lens_emulator/
│   ├── config.py              # shared omega/y intervals and normalization
│   ├── data.py                # dataset generation and PyTorch Dataset wrapper
│   ├── models.py              # Fourier features + SIREN implementation
│   ├── train.py               # training loop and checkpoint saving
│   ├── evaluate.py            # validation against reference calculations
│   ├── infer.py               # single-point inference from a checkpoint
│   └── physics/
│       ├── point_mass.py      # built-in point-mass lens F(omega,y)
│       └── sis.py             # SIS interface via GLoW
├── scripts/
│   ├── generate_dataset.py    # command-line data generation
│   ├── train_interval.py      # command-line training entry point
│   ├── evaluate_checkpoint.py # command-line evaluation entry point
│   └── infer.py               # command-line inference entry point
├── configs/
│   ├── pm_I1_example.json     # example PM run configuration
│   └── sis_I1_example.json    # example SIS run configuration
├── examples/
│   └── quick_start.sh         # minimal reproducibility workflow
├── requirements.txt
├── environment.yml
├── CITATION.cff
└── .gitignore
```

Large generated arrays and trained checkpoints are intentionally not committed to Git. They should be placed under `data/processed/` and `runs/`, respectively, or archived in a permanent data repository such as Zenodo when submitting the final paper.

---

## Installation

Create a clean Python environment:

```bash
conda create -n gwlens-emulator python=3.10 -y
conda activate gwlens-emulator
pip install -r requirements.txt
```

Install PyTorch following the command recommended for your CUDA version on the official PyTorch website. For example, on a CUDA-enabled server:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For SIS data generation, install GLoW in the same environment. The PM workflow does not require GLoW.

---

## Reproducing the workflow

### 1. Generate reference data

Point-mass lens example for interval `I1`:

```bash
python scripts/generate_dataset.py \
  --lens pm \
  --interval I1 \
  --output-dir data/processed \
  --n-y 40 \
  --n-omega 500
```

SIS example, requiring GLoW:

```bash
python scripts/generate_dataset.py \
  --lens sis \
  --interval I1 \
  --output-dir data/processed \
  --n-y 40 \
  --n-omega 500
```

For the full-scale experiments, use the default interval sizes or specify the larger values used in the manuscript:

| interval | expanded training range in y | default y points | default omega points |
|---|---:|---:|---:|
| I1 | 0.15--1.05 | 400 | 5000 |
| I2 | 0.95--3.05 | 600 | 5000 |
| I3 | 2.95--6.05 | 600 | 10000 |
| I4 | 5.95--10.05 | 800 | 20000 |

The expanded intervals include small overlaps to reduce boundary artifacts. For reporting final metrics, evaluate on the physical non-overlapping ranges `[0.2,1.0]`, `[1.0,3.0]`, `[3.0,6.0]`, and `[6.0,10.0]`.

### 2. Train one interval

```bash
python scripts/train_interval.py \
  --data data/processed/pm_I1_y40_w500.npy \
  --interval I1 \
  --output-dir runs/pm_I1_demo \
  --epochs 200 \
  --batch-size 8192 \
  --fourier-feats 64 \
  --hidden-dim 512 \
  --depth 8 \
  --w0 80.0 \
  --scale 30.0 0.5
```

For multi-GPU single-process training with PyTorch `DataParallel`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_interval.py \
  --data data/processed/pm_I1_y40_w500.npy \
  --interval I1 \
  --output-dir runs/pm_I1_dp \
  --data-parallel
```

### 3. Evaluate a checkpoint

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint runs/pm_I1_demo/best.pt \
  --lens pm \
  --interval I1 \
  --n-y 20 \
  --n-omega 1000 \
  --output-json runs/pm_I1_demo/eval.json
```

The script reports mean, median, 95th-percentile, and maximum relative error of `|F|`.

### 4. Single-point inference

```bash
python scripts/infer.py \
  --checkpoint runs/pm_I1_demo/best.pt \
  --omega 10.0 \
  --y 0.8
```

---

## Notes on reproducibility and AAS-style code availability

- The random seed is fixed by default (`--seed 42`) and is written to `run_config.json` in each run directory.
- Each checkpoint stores model hyperparameters, normalization bounds, the training epoch, optimizer state, and validation loss.
- Generated datasets use a transparent column format: `[omega, y, Re(F), Im(F)]` in `.npy` files.
- The repository separates data generation, model definition, training, evaluation, and inference to make independent reproduction easier.
- Full generated datasets and final trained checkpoints should be archived with a DOI for the published version; GitHub should keep only source code, lightweight configs, and documentation.

---

## Minimal smoke test

A very small run can be used to verify the environment:

```bash
bash examples/quick_start.sh
```

This will generate a small PM dataset, train a short run, evaluate it, and perform one inference call. The smoke-test accuracy is not expected to match the manuscript because it uses a much smaller dataset and far fewer epochs.
