"""
Where the data lives. Locally: ./data and app/data/today.json. On Railway: the volume
(RAILWAY_VOLUME_MOUNT_PATH, set by Railway when a volume is attached), or QUINTE_DATA if set.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("QUINTE_DATA") or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") or ROOT / "data")
APP_JSON = Path(os.environ.get("QUINTE_APP_JSON") or ROOT / "app" / "data" / "today.json")
