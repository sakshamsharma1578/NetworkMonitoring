"""
main.py
-------
Main real-time detection loop.

Usage
-----
    # From project root:
    python scripts/main.py

    # First-time setup (generate data + train model):
    python scripts/collect.py      # generates data/live.json
    python scripts/train.py        # trains and saves model/model.pkl
    python scripts/main.py         # starts the detection loop
"""

import os
import sys
import time
from datetime import datetime

def interpret(features):
    rssi = features.get("rssi", 0)
    rsrp = features.get("rsrp", 0)
    rsrq = features.get("rsrq", 0)
    snr  = features.get("snr", 0)

    if rsrp < -110:
        return "Poor Signal (Distance/Obstacle)"

    if rsrp > -90 and snr < 5:
        return "Possible Network Congestion"

    if snr < 3:
        return "High Interference"

    if rsrq < -15:
        return "Poor Signal Quality"

    return "Normal Condition"

# ── Path fix so imports work from any working directory ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from Scripts.detect import load_model, format_result, detect_from_file
from preprocess import preprocess_latest

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH      = "data/live.json"
MODEL_PATH     = "model/model.pkl"
POLL_INTERVAL  = 3          # seconds between each check
ANOMALY_WINDOW = 3          # consecutive anomalies before raising a CRITICAL alert

# ── ANSI color codes (work on most terminals, harmless if not supported) ─────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════╗
║   ML-Based Real-Time Network Anomaly Detection       ║
║   Model: Isolation Forest  |  Features: RSSI/RSRP/   ║
║          RSRQ/SNR          |  Press Ctrl+C to stop   ║
╚══════════════════════════════════════════════════════╝{RESET}
""")


def run_loop(data_path: str = DATA_PATH,
             model_path: str = MODEL_PATH,
             poll_interval: int = POLL_INTERVAL) -> None:

    banner()

    # Load model once outside the loop (faster per-iteration)
    print(f"[main] Loading model from {model_path} …")
    model, scaler = load_model(model_path)
    print(f"[main] Model loaded.  Starting detection loop "
          f"(interval = {poll_interval}s)\n")

    consecutive_anomalies = 0
    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            try:
                from detect import predict
                vector, features = preprocess_latest(data_path)
                result = predict(vector, model, scaler)
                result["features"] = features
                reason = interpret(features)

                # ── Colour output based on result ────────────────────────
                if result["raw_label"] == -1:
                    consecutive_anomalies += 1
                    color = RED
                    # Critical alert after N consecutive anomalies
                    if consecutive_anomalies >= ANOMALY_WINDOW:
                        crit = (f"  {YELLOW}{BOLD}[!] CRITICAL:"
                                f" {consecutive_anomalies} consecutive anomalies{RESET}")
                    else:
                        crit = ""
                    line = (f"{color}[{timestamp}] #{iteration:04d}  "
                    f"{format_result(result)} | Reason: {reason}{RESET}{crit}")
                else:
                    consecutive_anomalies = 0
                    color = GREEN
                    line = (f"{color}[{timestamp}] #{iteration:04d}  "
                    f"{format_result(result)} | Status: {reason}{RESET}")

                print(line)

            except FileNotFoundError as e:
                print(f"{YELLOW}[main] Waiting for data file … ({e}){RESET}")
            except KeyError as e:
                print(f"{YELLOW}[main] Incomplete record, skipping … ({e}){RESET}")
            except Exception as e:
                print(f"{RED}[main] Unexpected error: {e}{RESET}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n{CYAN}[main] Detection loop stopped by user.{RESET}")
        print(f"[main] Total iterations: {iteration}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_loop()
