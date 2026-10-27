"""
Phase 4: download the raw payouts (rapports) of each race, one JSON file per race.

The endpoint name comes from public examples and is NOT verified yet: run it on
one race by hand first, then inspect a saved file with inspect_json.py before
writing any parser. Saves to data/raw/rapports/YYYY-MM-DD/R{n}C{n}.json.

Usage (Python 3.9+, no extra packages):
    python fetch_rapports.py --quinte     # only the Quinté+ race of each day
    python fetch_rapports.py --france     # all races at French meetings

Re-running skips races already downloaded, so it is safe to stop and restart.
Personal use only: one request per second, no parallel calls.
"""

from fetch_participants import download, parse_scope
from paths import DATA

ENDPOINT = "rapports-definitifs"
OUT = DATA / "raw" / "rapports"

if __name__ == "__main__":
    download(parse_scope(__doc__), ENDPOINT, OUT)
