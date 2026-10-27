"""
Railway service: serves the phone app (app/) and runs the daily job at 06:00 Paris time.

- GET /                 the app (app/index.html, styles.css, app.js)
- GET /data/today.json  the day's file, kept on the volume so it survives redeploys
- GET /status           last lines of the job log (plain text)

First start on an empty volume: downloads 12 months of Quinté data (~1,460 requests at one per
second, ~25 min), builds the tables and runs the backtest, then the day's job. Every day at 06:00
(Paris): catches up the last 7 days of raw files (skips what is saved), rebuilds the tables, runs
daily.py. If a run fails it retries every 30 minutes until noon.

Environment (all optional):
    PORT                       set by Railway
    RAILWAY_VOLUME_MOUNT_PATH  set by Railway when a volume is attached: data goes there
    APP_PASSWORD               if set, the app asks for it once (HTTP basic auth, any user name)

Usage: python server.py   (Python 3.9+, no extra packages)
"""
import base64
import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from paths import DATA, ROOT

os.environ.setdefault("QUINTE_APP_JSON", str(DATA / "today.json"))  # inherited by the job scripts
APP_JSON = Path(os.environ["QUINTE_APP_JSON"])
LOG = DATA / "server_log.txt"
BOOTSTRAPPED = DATA / "bootstrap_done.txt"
LAST_RUN = DATA / "last_run.txt"
RUN_AT_HOUR, GIVE_UP_HOUR, RETRY_MIN = 6, 12, 30
PASSWORD = os.environ.get("APP_PASSWORD", "")


# ---------------------------------------------------------------- Paris time without tz database

def last_sunday(year, month):
    d = date(year, month, 31)  # March and October both have 31 days
    return d - timedelta(days=(d.weekday() + 1) % 7)


def paris_now():
    """EU summer time: from the last Sunday of March to the last Sunday of October, 01:00 UTC."""
    utc = datetime.now(timezone.utc)
    start = datetime.combine(last_sunday(utc.year, 3), datetime.min.time(), timezone.utc) + timedelta(hours=1)
    end = datetime.combine(last_sunday(utc.year, 10), datetime.min.time(), timezone.utc) + timedelta(hours=1)
    return (utc + timedelta(hours=2 if start <= utc < end else 1)).replace(tzinfo=None)


# ---------------------------------------------------------------- jobs

def log(text):
    DATA.mkdir(parents=True, exist_ok=True)
    line = f"{paris_now():%Y-%m-%d %H:%M:%S} {text}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(*args, timeout=3600):
    """Run one pipeline script; log its last lines. Returns True on success."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    try:
        p = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"{' '.join(args)}: timeout after {timeout} s")
        return False
    tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
    log(f"{' '.join(args)}: exit {p.returncode}\n    " + "\n    ".join(tail))
    return p.returncode == 0


def catch_up(start, end):
    """Raw files for every Quinté from start to end (the scripts skip what is already saved)."""
    return (run("fetch_programmes.py", start.isoformat(), end.isoformat(), timeout=3 * 3600)
            and run("fetch_participants.py", "--quinte", timeout=3 * 3600)
            and run("fetch_performances.py", "--quinte", timeout=3 * 3600)
            and run("fetch_rapports.py", "--quinte", timeout=3 * 3600))


def bootstrap(today):
    log("bootstrap: downloading 12 months of Quinté data")
    ok = (catch_up(today - timedelta(days=366), today - timedelta(days=1))
          and run("build_dataset.py") and run("features.py") and run("backtest.py", timeout=3 * 3600))
    if ok:
        BOOTSTRAPPED.write_text(today.isoformat(), encoding="utf-8")
    return ok


def daily(today):
    log(f"daily job for {today}")
    ok = (catch_up(today - timedelta(days=7), today - timedelta(days=1))
          and run("build_dataset.py") and run("daily.py", today.isoformat(), timeout=1800))
    if ok:
        LAST_RUN.write_text(today.isoformat(), encoding="utf-8")
    return ok


def scheduler():
    next_try = datetime.min
    while True:
        now = paris_now()
        today = now.date()
        if now >= next_try:
            if not BOOTSTRAPPED.exists():
                if not bootstrap(today):
                    next_try = now + timedelta(hours=1)
                    continue
            done = LAST_RUN.read_text(encoding="utf-8").strip() if LAST_RUN.exists() else ""
            due = RUN_AT_HOUR <= now.hour < GIVE_UP_HOUR or not APP_JSON.exists()
            if done != today.isoformat() and due:
                if not daily(today):
                    next_try = now + timedelta(minutes=RETRY_MIN)
        time.sleep(30)


# ---------------------------------------------------------------- web

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("X-Robots-Tag", "noindex, nofollow")
        super().end_headers()

    def authorised(self):
        if not PASSWORD:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                if base64.b64decode(header[6:]).decode("utf-8").split(":", 1)[1] == PASSWORD:
                    return True
            except (ValueError, IndexError):
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Quinte", charset="UTF-8"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def send_bytes(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.authorised():
            return
        path = self.path.split("?", 1)[0]
        if path == "/data/today.json":
            if APP_JSON.exists():
                return self.send_bytes(APP_JSON.read_bytes(), "application/json; charset=utf-8")
            return self.send_error(404, "today.json not generated yet (see /status)")
        if path == "/status":
            lines = LOG.read_text(encoding="utf-8").splitlines()[-80:] if LOG.exists() else ["no run yet"]
            return self.send_bytes("\n".join(lines).encode("utf-8"), "text/plain; charset=utf-8")
        return super().do_GET()

    def do_HEAD(self):
        if self.authorised():
            super().do_HEAD()

    def log_message(self, fmt, *args):
        pass  # keep the Railway log for the job


def main():
    port = int(os.environ.get("PORT", "8000"))
    threading.Thread(target=scheduler, daemon=True).start()
    log(f"server on port {port}, data in {DATA}, app file {APP_JSON}")
    ThreadingHTTPServer(("", port), partial(Handler, directory=str(ROOT / "app"))).serve_forever()


if __name__ == "__main__":
    main()
