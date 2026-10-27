"""
Phase 6a: compute pre-race features for each runner of each Quinté race.

Reads data/runners.csv and data/past_runs.csv (built by build_dataset.py) and writes
data/features.csv: one row per runner that started (statut PARTANT), with
    race_id, date, numPmu, nom, specialite, res_place, res_top5, cote_reference, f_* features.
Only information available before the race is used: past runs are dated before the race
(checked), jockey and trainer rates use earlier dates only. Missing values are left empty;
the model fills them with the race average.

Usage (Python 3.9+, no extra packages):
    python features.py

daily.py calls compute() directly for the day's race.
"""
import csv
import math
import re
import unicodedata
from collections import defaultdict
from datetime import date, timedelta

from paths import DATA


# Going, from firmest to heaviest. PSF (all-weather) is kept apart.
GOING = {"tres leger": 0, "leger": 0, "bon leger": 1, "sec": 0, "bon": 1, "bon souple": 2,
         "souple": 3, "tres souple": 4, "collant": 5, "lourd": 5, "tres lourd": 6}
FAILED = {"DISQUALIFIE", "DISQUALIFIE_POUR_ALLURE_IRREGULIERE", "DISQUALIFIE_POTEAU_GALOP",
          "ARRETE", "TOMBE", "DISTANCE"}


def plain(s):
    """Upper-case ASCII letters only: 'Paris-Vincennes' -> 'PARISVINCENNES'."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z]", "", s.upper())


def track_key(name):
    """Past runs write 'Vincennes' or 'PARIS-VINCENNES', runners 'VINCENNES': drop the PARIS prefix."""
    k = plain(name)
    return k[5:] if k.startswith("PARIS") and len(k) > 5 else k


def going(s):
    """-> ('PSF', None), ('TURF', 0..6) or (None, None) when unknown."""
    t = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().strip()
    t = re.sub(r"^tr ", "tres ", t)
    if not t:
        return None, None
    if "psf" in t or t in ("standard", "rapide", "lente", "standard rapide", "standard lente"):
        return "PSF", None
    return ("TURF", GOING[t]) if t in GOING else (None, None)


def family(discipline):
    """Group past-race and race disciplines into the same 4 families."""
    d = (discipline or "").upper()
    if d in ("PLAT",):
        return "PLAT"
    if d in ("ATTELE", "TROT_ATTELE"):
        return "ATTELE"
    if d in ("MONTE", "TROT_MONTE"):
        return "MONTE"
    return "OBSTACLE"


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def rate(hits, n, prior, k=3.0):
    """Smoothed rate: few runs stay close to the prior."""
    return (hits + prior * k) / (n + k)


def load(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def past_by_runner(past):
    out = defaultdict(list)
    for p in past:
        status = p["statusArrivee"]
        place = num(p["place"]) if status in ("PLACE", "NON_PLACE") else None
        out[(p["race_id"], p["numPmu"])].append({
            "date": date.fromisoformat(p["past_date"]),
            "days": int(p["jours_avant"]),
            "place": place,
            "failed": status in FAILED,
            "top3": place is not None and place <= 3,
            "win": place == 1,
            "family": family(p["discipline"]),
            "distance": num(p["distance"]),
            "track": track_key(p["hippodrome"]),
            "going": going(p["terrain"]),
            "allocation": num(p["allocation"]),
            "jockey": p["jockey"],
        })
    for runs in out.values():
        runs.sort(key=lambda r: r["date"], reverse=True)
    return out


def jockey_index(past):
    """Unique past rides (same horse, date and prize can appear in several files) per jockey."""
    seen, rides = set(), defaultdict(list)
    for p in past:
        key = (p["nom"], p["past_date"], p["prix"])
        if key in seen or not p["jockey"]:
            continue
        seen.add(key)
        ok = p["statusArrivee"] in ("PLACE", "NON_PLACE") and num(p["place"]) is not None
        rides[p["jockey"]].append((date.fromisoformat(p["past_date"]), ok and num(p["place"]) <= 3))
    return rides


def compute(runners, past):
    """Feature rows for every starter, from runner and past-run rows shaped like the CSVs (all strings)."""
    runners = [r for r in runners if r["statut"] == "PARTANT"]
    runs_of = past_by_runner(past)
    rides = jockey_index(past)
    jockey_prior = sum(t for v in rides.values() for _, t in v) / max(1, sum(len(v) for v in rides.values()))

    by_race = defaultdict(list)
    for r in runners:
        by_race[r["race_id"]].append(r)
    trainer_hist = defaultdict(list)  # trainer -> [(date, top5)] from earlier Quinté races
    for r in runners:
        if r["res_top5"] != "":
            trainer_hist[r["entraineur"]].append((date.fromisoformat(r["date"]), r["res_top5"] == "1"))

    rows = []
    for race_id, field in sorted(by_race.items()):
        r0 = field[0]
        day = date.fromisoformat(r0["date"])
        fam = family(r0["specialite"] if r0["specialite"] != "OBSTACLE" else "HAIE")
        dist = num(r0["distance"])
        track = track_key(r0["hippodrome"])
        today_going = going(r0["terrain"])
        prize = num(r0["montantPrix"])
        n = len(field)
        means = {k: _mean(num(x[k]) for x in field) for k in ("handicapPoids", "handicapValeur", "handicapDistance", "age")}
        min_hd = min((num(x["handicapDistance"]) for x in field if num(x["handicapDistance"]) is not None), default=None)
        odds = {x["numPmu"]: num(x["cote_reference"]) or num(x["cote_finale"]) for x in field}
        inv = sum(1 / o for o in odds.values() if o)

        for r in field:
            runs = runs_of.get((race_id, r["numPmu"]), [])
            last5 = runs[:5]
            same_fam = [x for x in runs if x["family"] == fam]
            f = {}
            f["f_n_past"] = len(runs)
            f["f_days_since"] = min(runs[0]["days"], 365) if runs else None
            f["f_runs_60d"] = sum(x["days"] <= 60 for x in runs)
            f["f_form_place"] = _mean(min(x["place"], 10) if x["place"] is not None else 10 for x in last5)
            f["f_form_top3"] = rate(sum(x["top3"] for x in last5), len(last5), 0.3) if last5 else None
            f["f_form_win"] = rate(sum(x["win"] for x in last5), len(last5), 0.1) if last5 else None
            f["f_form_failed"] = (sum(x["failed"] for x in last5) / len(last5)) if last5 else None
            f["f_last_place"] = (min(runs[0]["place"], 10) if runs[0]["place"] is not None else 10) if runs else None
            f["f_same_family_share"] = len(same_fam) / len(runs) if runs else None

            if today_going[0] == "PSF":
                g = [x for x in runs if x["going"][0] == "PSF"]
            elif today_going[0] == "TURF":
                g = [x for x in runs if x["going"][0] == "TURF" and abs(x["going"][1] - today_going[1]) <= 1]
            else:
                g = None
            f["f_going_n"] = len(g) if g is not None else None
            f["f_going_top3"] = rate(sum(x["top3"] for x in g), len(g), 0.3) if g is not None else None

            d = [x for x in same_fam if dist and x["distance"] and abs(x["distance"] - dist) <= 200]
            f["f_dist_n"] = len(d)
            f["f_dist_top3"] = rate(sum(x["top3"] for x in d), len(d), 0.3)
            t = [x for x in runs if x["track"] == track]
            f["f_track_n"] = len(t)
            f["f_track_top3"] = rate(sum(x["top3"] for x in t), len(t), 0.3)
            alloc = _mean(x["allocation"] for x in last5 if x["allocation"])
            f["f_class"] = math.log(alloc / prize) if alloc and prize else None

            hp, hv, hd = num(r["handicapPoids"]), num(r["handicapValeur"]), num(r["handicapDistance"])
            f["f_weight_rel"] = (hp - means["handicapPoids"]) / 10 if hp is not None else None
            f["f_handicap_rel"] = hv - means["handicapValeur"] if hv is not None else None
            f["f_dist_handicap"] = (hd - min_hd) / 25 if hd is not None and min_hd is not None else None
            draw = num(r["placeCorde"])
            f["f_draw_rel"] = draw / n if draw else None
            f["f_age_rel"] = num(r["age"]) - means["age"]
            f["f_female"] = int(r["sexe"] == "FEMELLES")
            f["f_entire"] = int(r["sexe"] == "MALES")
            f["f_blinkers"] = int(r["oeilleres"] != "SANS_OEILLERES")
            f["f_unshod4"] = int(r["deferre"] == "DEFERRE_ANTERIEURS_POSTERIEURS") if fam in ("ATTELE", "MONTE") else None
            f["f_avis"] = {"POSITIF": 1, "NEGATIF": -1, "NEUTRE": 0}.get(r["avisEntraineur"])
            f["f_driver_change"] = int(r["driverChange"] == "True")

            nc, nv, npl = num(r["nombreCourses"]) or 0, num(r["nombreVictoires"]) or 0, num(r["nombrePlaces"]) or 0
            f["f_career_win"] = rate(nv, nc, 0.1, k=5)
            f["f_career_place"] = rate(npl, nc, 0.35, k=5)
            f["f_log_gains_per_run"] = math.log1p((num(r["gainsCarriere"]) or 0) / 100 / max(nc, 1))
            f["f_log_gains_year"] = math.log1p((num(r["gainsAnneeEnCours"]) or 0) / 100)

            jr = [t3 for dt, t3 in rides.get(r["driver"], []) if day - timedelta(days=90) <= dt < day]
            f["f_jockey_top3"] = rate(sum(jr), len(jr), jockey_prior, k=10)
            f["f_jockey_rides"] = math.log1p(len(jr))
            th = [t5 for dt, t5 in trainer_hist.get(r["entraineur"], []) if dt < day]
            f["f_trainer_top5"] = rate(sum(th), len(th), 5 / 15, k=5)

            o = odds[r["numPmu"]]
            f["f_mkt_logp"] = math.log((1 / o) / inv) if o and inv else None

            rows.append({"race_id": race_id, "date": r["date"], "numPmu": r["numPmu"], "nom": r["nom"],
                         "specialite": r["specialite"], "partants": n, "res_place": r["res_place"],
                         "res_top5": r["res_top5"], "cote_reference": odds[r["numPmu"]], **f})
    return [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in rows]


def main():
    rows = compute(load("runners.csv"), load("past_runs.csv"))
    with (DATA / "features.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"data/features.csv: {len(rows)} runners, {len({r['race_id'] for r in rows})} races, "
          f"{sum(k.startswith('f_') for k in rows[0])} features")


def _mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


if __name__ == "__main__":
    main()
