"""
Phase 4b: download each runner's recent past races (performances détaillées), one JSON file per race.

Endpoint verified on 2026-04-26 R1C3 and 2026-10-27 R1C1: the history stops before the
race date, so there is no look-ahead. Saves to data/raw/performances/YYYY-MM-DD/R{n}C{n}.json.

Usage (Python 3.9+, no extra packages):
    python fetch_performances.py --quinte     # only the Quinté+ race of each day

Re-running skips races already downloaded, so it is safe to stop and restart.
Personal use only: one request per second, no parallel calls.
"""

from fetch_participants import download, parse_scope
from paths import DATA

ENDPOINT = "performances-detaillees/pretty"
OUT = DATA / "raw" / "performances"

if __name__ == "__main__":
    download(parse_scope(__doc__), ENDPOINT, OUT)
