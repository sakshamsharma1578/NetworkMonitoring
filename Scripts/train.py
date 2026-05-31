"""
train.py
--------
Trains an Isolation Forest on the full simulated dataset and saves
the fitted model to model/model.pkl.

Run once (or whenever you want to retrain):
    python scripts/train.py
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

# Local import — works whether you run from project root or scripts/
try:
    from preprocess import preprocess_all
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from preprocess import preprocess_all

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH  = "data/live.json"
MODEL_PATH = "model/model.pkl"

# contamination = expected fraction of anomalies in training data.
# Keep low (≤0.05) since training data is mostly "normal".
CONTAMINATION = 0.05
RANDOM_STATE  = 42
N_ESTIMATORS  = 100   # number of trees in the forest


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> None:
    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    print("[train] Loading data from:", data_path)
    X, raw_list = preprocess_all(data_path)

    if len(X) == 0:
        raise ValueError("No valid records found — cannot train.")

    print(f"[train] Samples loaded : {len(X)}")
    print(f"[train] Feature shape  : {X.shape}")

    # ── 2. Scale features ────────────────────────────────────────────────────
    # StandardScaler helps Isolation Forest treat all features equally
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3. Train Isolation Forest ────────────────────────────────────────────
    print(f"[train] Training Isolation Forest  "
          f"(n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION}) …")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,          # use all available CPU cores
    )
    model.fit(X_scaled)

    # Quick training-set evaluation
    preds = model.predict(X_scaled)
    n_anomalies = (preds == -1).sum()
    print(f"[train] Training complete.")
    print(f"[train] Anomalies flagged on training data : "
          f"{n_anomalies} / {len(X)} "
          f"({100 * n_anomalies / len(X):.1f}%)")

    # ── 4. Save model + scaler together ──────────────────────────────────────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, model_path)
    print(f"[train] Model saved to : {model_path}")


if __name__ == "__main__":
    train()
