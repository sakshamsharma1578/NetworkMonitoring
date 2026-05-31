"""
preprocess.py
-------------
Reads the latest signal data from data/live.json and returns a
clean numerical feature vector ready for model input.
"""

import json
import numpy as np
import os

# Features the model expects — order must stay consistent
FEATURES = ["rssi", "rsrp", "rsrq", "snr"]

# Typical ranges for basic sanity / clipping
FEATURE_RANGES = {
    "rssi": (-120, -30),
    "rsrp": (-140, -44),
    "rsrq": (-20, -3),
    "snr":  (-10,  30),
}


def load_json(path: str = "data/live.json") -> list[dict]:
    """Load the JSON file and return a list of signal records."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    # Support both a single dict and a list of dicts
    if isinstance(data, dict):
        data = [data]
    return data


def extract_features(record: dict) -> dict:
    """
    Pull only the four signal features from one record.
    Raises KeyError if a required field is missing.
    """
    return {feature: float(record[feature]) for feature in FEATURES}


def clip_features(features: dict) -> dict:
    """Clip values to known physical ranges to remove sensor glitches."""
    clipped = {}
    for feat, value in features.items():
        lo, hi = FEATURE_RANGES[feat]
        clipped[feat] = max(lo, min(hi, value))
    return clipped


def to_vector(features: dict) -> np.ndarray:
    """Convert feature dict to a numpy row vector (1 × n_features)."""
    return np.array([[features[f] for f in FEATURES]], dtype=np.float32)


def preprocess_latest(path: str = "data/live.json") -> tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline for the most recent record.

    Returns
    -------
    vector : np.ndarray, shape (1, 4)
        Feature vector ready for model.predict()
    raw : dict
        The raw feature values (useful for logging)
    """
    records = load_json(path)
    latest = records[-1]                    # take the most recent entry
    features = extract_features(latest)
    features = clip_features(features)
    vector = to_vector(features)
    return vector, features


def preprocess_all(path: str = "data/live.json") -> tuple[np.ndarray, list[dict]]:
    """
    Preprocess every record in the file — used during training.

    Returns
    -------
    matrix : np.ndarray, shape (n_samples, 4)
    raw_list : list[dict]
    """
    records = load_json(path)
    raw_list = []
    rows = []
    for rec in records:
        try:
            features = extract_features(rec)
            features = clip_features(features)
            raw_list.append(features)
            rows.append([features[f] for f in FEATURES])
        except KeyError as e:
            print(f"[preprocess] Skipping record — missing field: {e}")
    matrix = np.array(rows, dtype=np.float32)
    return matrix, raw_list


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    vec, raw = preprocess_latest()
    print("Latest record features :", raw)
    print("Feature vector shape   :", vec.shape)
    print("Feature vector         :", vec)
