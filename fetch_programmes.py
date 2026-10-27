"""
Phase 1: download the raw PMU daily programmes, one JSON file per day.

Why raw first: the PMU feed is undocumented, so we save exactly what it returns
and only write the parser after looking at real files (see inspect_json.py).

Usage (Python 3.9+, no extra packages):
    python fetch_programmes.py                      # last 183 days up to yesterday
    python fetch_programmes.py 2026-04-26 2026-10-25

Re-running skips days already downloaded, so it is safe to stop and restart.
Personal use only: one request per second, no parallel calls.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

from paths import DATA

# Both hosts appear in public examples. Which one keeps older days is not
# documented, so we try "online" first and fall back to "offline".
HOSTS = [
    "https://online.turfinfo.api.pmu.fr/rest/client/7/programme/",
    "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme/",
]

OUT = DATA / "raw" / "programme"
PAUSE_S = 1.0
MAX_MISSING_STREAK = 10  # stop rather than hammer the API if a whole range is gone
HEADERS = {"User-Agent": "quinte-perso/0.1 (usage personnel)", "Accept": "application/json"}


def fetch(url: str, tries: int = 3):
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return e.code, None  # no data for that day on this host
            err = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError) as e:
            err = str(e)
        print(f"    retry {attempt}/{tries} after error: {err}")
        time.sleep(PAUSE_S * 5 * attempt)
    return None, None


def main():
    if len(sys.argv) == 3:
        start, end = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2])
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=182)
    OUT.mkdir(parents=True, exist_ok=True)

    log, d, streak = [], start, 0
    while d <= end:
        key = d.strftime("%d%m%Y")
        target = OUT / f"{d.isoformat()}.json"
        if target.exists():
            d += timedelta(days=1)
            continue
        got = None
        for host in HOSTS:
            status, body = fetch(host + key)
            time.sleep(PAUSE_S)
            if body:
                try:
                    json.loads(body)
                except ValueError:
                    print(f"{d}  {host.split('//')[1].split('.')[0]}: reply is not JSON, skipped")
                    continue
                target.write_bytes(body)
                got = (host, len(body))
                break
        line = f"{d}  " + (f"ok  {got[0].split('//')[1].split('.')[0]}  {got[1] // 1024} KB" if got else "MISSING")
        print(line)
        log.append(line)
        streak = 0 if got else streak + 1
        if streak >= MAX_MISSING_STREAK:
            print(f"\nStopped: {streak} days in a row MISSING (last {d}).")
            log.append(f"STOPPED after {streak} consecutive MISSING days")
            break
        d += timedelta(days=1)

    (OUT.parent / "fetch_log.txt").write_text("\n".join(log), encoding="utf-8")
    missing = sum("MISSING" in l for l in log)
    print(f"\nDone: {len(log) - missing} days saved, {missing} missing. Files in {OUT}")


if __name__ == "__main__":
    main()
