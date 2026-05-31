import random
import json
import time
import os
from datetime import datetime

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "..", "data", "live.json")

def generate_data():
    return {
        "timestamp": datetime.now().isoformat(),
        "rssi": random.randint(-110, -70),
        "rsrp": random.randint(-120, -80),
        "rsrq": random.randint(-20, -5),
        "snr": random.randint(0, 30)
    }

while True:
    data = generate_data()

    # Load existing data (if any)
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except:
                existing_data = []
    else:
        existing_data = []

    # Append new data
    existing_data.append(data)

    # Save back
    with open(FILE_PATH, "w") as f:
        json.dump(existing_data, f, indent=4)

    print("Generated:", data)

    time.sleep(3)