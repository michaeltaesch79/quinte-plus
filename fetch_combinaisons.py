"""
Download the pool stakes (combinaisons) of each race, one JSON file per race.

Tested on one race (2026-10-28 R1C4) and one past race (2024-11-12 R1C4): for each bet type,
the total stakes and the 12 most-played combinations with their stakes. These are the FINAL
stakes (after betting closed), not what was known in the morning.
Saves to data/raw/combinaisons/YYYY-MM-DD/R{n}C{n}.json.

Usage (Python 3.9+, no extra packages):
    python fetch_combinaisons.py --quinte     # only the Quinté+ race of each day

Re-running skips races already downloaded, so it is safe to stop and restart.
Personal use only: one request per second, no parallel calls.
"""

from fetch_participants import date_range, download, parse_scope
from paths import DATA

ENDPOINT = "combinaisons?specialisation=INTERNET"
OUT = DATA / "raw" / "combinaisons"

if __name__ == "__main__":
    download(parse_scope(__doc__), ENDPOINT, OUT, *date_range())
