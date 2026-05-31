"""
detect.py
---------
Loads the trained model and predicts whether a given feature vector
is Normal (1) or an Anomaly (-1).

Can also be imported as a module by main.py.
"""

import os
import joblib
import numpy as np

# Local import
try:
    from preprocess import preprocess_latest, FEATURES
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from preprocess import preprocess_latest, FEATURES

MODEL_PATH = "model/model.pkl"

# ── Labels ───────────────────────────────────────────────────────────────────
LABEL = {
    1:  "NORMAL",
    -1: "⚠  ANOMALY DETECTED",
}


def load_model(model_path: str = MODEL_PATH) -> tuple:
    """Load the model + scaler bundle saved by train.py."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run train.py first."
        )
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["scaler"]


def predict(vector: np.ndarray, model, scaler) -> dict:
    """
    Run inference on a (1 × n_features) vector.

    Returns a dict with:
        label       : "NORMAL" or "⚠  ANOMALY DETECTED"
        raw_label   : 1 or -1
        anomaly_score : float  (more negative = more anomalous)
    """
    scaled = scaler.transform(vector)
    raw_label     = int(model.predict(scaled)[0])          # 1 or -1
    anomaly_score = float(model.decision_function(scaled)[0])

    return {
        "label":        LABEL[raw_label],
        "raw_label":    raw_label,
        "anomaly_score": round(anomaly_score, 4),
    }


def detect_from_file(data_path: str = "data/live.json",
                     model_path: str = MODEL_PATH) -> dict:
    """
    Convenience wrapper: read latest record → preprocess → predict.
    Returns the result dict plus the raw feature values.
    """
    model, scaler = load_model(model_path)
    vector, features = preprocess_latest(data_path)
    result = predict(vector, model, scaler)
    result["features"] = features
    return result


def format_result(result: dict) -> str:
    """Pretty-print a result dict to a single string."""
    f = result["features"]
    feat_str = "  ".join(
        f"{k.upper()}={v:.1f}" for k, v in f.items()
    )
    score_str = f"score={result['anomaly_score']:+.4f}"
    return f"[{result['label']}]  {feat_str}  ({score_str})"


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = detect_from_file()
    print(format_result(result))
