from dataclasses import dataclass
from typing import Dict, Tuple

OMEGA_BOUNDS: Tuple[float, float] = (0.005, 45.0)

# Expanded training intervals. Evaluation should normally use the physical
# intervals below to avoid counting overlap-buffer regions twice.
TRAIN_INTERVALS: Dict[str, Tuple[float, float]] = {
    "I1": (0.15, 1.05),
    "I2": (0.95, 3.05),
    "I3": (2.95, 6.05),
    "I4": (5.95, 10.05),
}

EVAL_INTERVALS: Dict[str, Tuple[float, float]] = {
    "I1": (0.2, 1.0),
    "I2": (1.0, 3.0),
    "I3": (3.0, 6.0),
    "I4": (6.0, 10.0),
}

DEFAULT_Y_POINTS = {"I1": 400, "I2": 600, "I3": 600, "I4": 800}
DEFAULT_OMEGA_POINTS = {"I1": 5000, "I2": 5000, "I3": 10000, "I4": 20000}

@dataclass(frozen=True)
class Normalization:
    omega_min: float
    omega_max: float
    y_min: float
    y_max: float

    def normalize(self, omega, y):
        omega_n = 2.0 * (omega - self.omega_min) / (self.omega_max - self.omega_min) - 1.0
        y_n = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        return omega_n, y_n
