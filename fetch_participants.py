"""
Phase 3: download the raw runners (participants) of each race, one JSON file per race.

Reads the programmes saved by fetch_programmes.py and saves exactly what the feed
returns, to data/raw/participants/YYYY-MM-DD/R{n}C{n}.json. Inspect before parsing.

Usage (Python 3.9+, no extra packages):
    python fetch_participants.py --quinte     # only the Quinté+ race of each day
    python fetch_participants.py --france     # all races at French meetings
    python fetch_participants.py --all        # every race, including foreign meetings

Re-running skips races already downloaded, so it is safe to stop and restart.
Personal use only: one request per second, no parallel calls.
"""
import json
import sys
import time
from pathlib import Path

from fetch_programmes import HOSTS, PAUSE_S, MAX_MISSING_STREAK, fetch

ROOT = Path(__file__).parent / "data" / "raw"
PROG = ROOT / "programme"
OUT = ROOT / "participants"


def races(scope: str):
    """Yield (iso_date, R, C) for each race in scope, from the saved programmes."""
    for f in sorted(PROG.glob("*.json")):
        prog = json.loads(f.read_bytes())["programme"]
        for reunion in prog["reunions"]:
            if scope != "all" and reunion["pays"]["code"] != "FRA":
                continue
            for course in reunion["courses"]:
                is_quinte = any(p.get("codePari") == "QUINTE_PLUS" for p in course.get("paris", []))
                if scope == "quinte" and not is_quinte:
                    continue
                yield f.stem, reunion["numOfficiel"], course["numOrdre"]


def parse_scope(usage: str) -> str:
    flags = {"--quinte": "quinte", "--france": "france", "--all": "all"}
    if len(sys.argv) != 2 or sys.argv[1] not in flags:
        sys.exit(usage)
    return flags[sys.argv[1]]


def download(scope: str, endpoint: str, out: Path):
    """Fetch {host}DDMMYYYY/R{n}/C{n}/{endpoint} for each race in scope, saving raw replies under out."""
    todo = list(races(scope))
    print(f"{len(todo)} races in scope '{scope}'")
    log, streak = [], 0
    for iso, r, c in todo:
        target = out / iso / f"R{r}C{c}.json"
        if target.exists():
            continue
        key = f"{iso[8:10]}{iso[5:7]}{iso[0:4]}"
        got = None
        for host in HOSTS:
            status, body = fetch(f"{host}{key}/R{r}/C{c}/{endpoint}")
            time.sleep(PAUSE_S)
            if body:
                try:
                    json.loads(body)
                except ValueError:
                    print(f"{iso} R{r}C{c}  {host.split('//')[1].split('.')[0]}: reply is not JSON, skipped")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
                got = (host, len(body))
                break
        line = f"{iso} R{r}C{c}  " + (f"ok  {got[0].split('//')[1].split('.')[0]}  {got[1] // 1024} KB" if got else "MISSING")
        print(line)
        log.append(line)
        streak = 0 if got else streak + 1
        if streak >= MAX_MISSING_STREAK:
            print(f"\nStopped: {streak} races in a row MISSING (last {iso} R{r}C{c}).")
            log.append(f"STOPPED after {streak} consecutive MISSING races")
            break

    out.mkdir(parents=True, exist_ok=True)
    (ROOT / f"{out.name}_log_{scope}.txt").write_text("\n".join(log), encoding="utf-8")
    missing = sum("MISSING" in l for l in log)
    print(f"\nDone: {len(log) - missing} races saved, {missing} missing. Files in {out}")


def main():
    download(parse_scope(__doc__), "participants", OUT)


if __name__ == "__main__":
    main()
