"""
train_model.py
--------------
Trains the on-device anomaly-detection model for predictive maintenance.

Approach: unsupervised anomaly detection with IsolationForest.
Rationale: in real predictive-maintenance settings you have LOTS of "healthy"
data and very few labeled failures. So we train only on healthy data to learn
what "normal" looks like, then flag anything that deviates. This mirrors how
TinyML anomaly detectors (and tools like Edge Impulse) actually work.

We use a short sliding window of samples and extract simple statistical
features (mean / std / min / max / peak-to-peak). These features are cheap to
compute on a microcontroller — important for the eventual edge deployment.

Artifacts written to models/:
  - anomaly_model.pkl : trained IsolationForest + fitted scaler + config
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Allow running from repo root or from src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensor_simulator import MotorSensorSimulator  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")

WINDOW_SIZE = 20   # samples per feature window (~2s at 10 Hz)
WINDOW_STEP = 5    # slide step between windows


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Turn a window of raw samples into a compact feature vector.

    window shape: (WINDOW_SIZE, 2)  columns = [vibration, temperature]
    Returns a 1-D feature vector. These stats are MCU-friendly (no FFT needed
    for Phase 1; FFT features can be added later for the real edge model).
    """
    vib = window[:, 0]
    temp = window[:, 1]
    return np.array([
        vib.mean(), vib.std(), vib.min(), vib.max(), vib.max() - vib.min(),
        temp.mean(), temp.std(), temp.max(),
    ])


def build_windows(readings: np.ndarray) -> np.ndarray:
    """Slide over a (N, 2) array of readings and produce a feature matrix."""
    feats = []
    for start in range(0, len(readings) - WINDOW_SIZE + 1, WINDOW_STEP):
        window = readings[start:start + WINDOW_SIZE]
        feats.append(extract_features(window))
    return np.array(feats)


def generate_healthy_dataset(n_samples: int = 6000, seed: int = 42) -> np.ndarray:
    """Produce a long stream of HEALTHY readings for training."""
    sim = MotorSensorSimulator(fault_mode="none", seed=seed)
    raw = np.array([[r.vibration, r.temperature] for r in sim.stream(n_samples)])
    return raw


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Generating healthy training data...")
    raw = generate_healthy_dataset()
    X = build_windows(raw)
    print(f"  raw samples: {len(raw)}  ->  feature windows: {X.shape[0]} "
          f"(each {X.shape[1]} features)")

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training IsolationForest on healthy data...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,   # expect ~1% noise even in healthy data
        random_state=42,
    )
    model.fit(X_scaled)

    # Quick sanity check: score healthy vs. a faulty stream.
    faulty = MotorSensorSimulator(fault_mode="combined", fault_severity=0.9, seed=7)
    faulty_raw = np.array([[r.vibration, r.temperature] for r in faulty.stream(2000)])
    Xf = scaler.transform(build_windows(faulty_raw))

    healthy_flagged = (model.predict(X_scaled) == -1).mean()
    faulty_flagged = (model.predict(Xf) == -1).mean()
    print(f"  healthy windows flagged as anomaly: {healthy_flagged:.1%} (want low)")
    print(f"  faulty  windows flagged as anomaly: {faulty_flagged:.1%} (want high)")

    bundle = {
        "model": model,
        "scaler": scaler,
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "feature_names": [
            "vib_mean", "vib_std", "vib_min", "vib_max", "vib_ptp",
            "temp_mean", "temp_std", "temp_max",
        ],
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
