"""
Phase 6a: compute pre-race features for each runner of each Quinté race.

Reads data/runners.csv and data/past_runs.csv (built by build_dataset.py) and writes
data/features.csv: one row per runner that started (statut PARTANT), with
    race_id, date, numPmu, nom, specialite, res_place, res_top5, cote_reference, f_* features.
Only information available before the race is used (research/checks.py tests this):
- past runs are dated before the race;
- jockey rides come only from past-performance files of races up to the race date, so a ride
  is only known once a file that lists it was available (no look-ahead through later files);
- trainer rates use earlier Quinté results only;
- optional jockey/trainer form from all French races (data/all_runs.csv, when downloaded): races
  strictly before the race day, same discipline group, actual vs expected top-3 finishes;
- the market features (ODDS) use the reference odds (30 min before the start), never the odds
  taken after the start (cote_finale).
Missing values are left empty; the model fills them with the race average.

Usage (Python 3.9+, no extra packages):
    python features.py

daily.py calls compute() directly for the day's race.
"""
import csv
import math
import re
import unicodedata
from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta

from paths import DATA


# Going, from firmest to heaviest. PSF (all-weather) is kept apart.
GOING = {"tres leger": 0, "leger": 0, "bon leger": 1, "sec": 0, "bon": 1, "bon souple": 2,
         "souple": 3, "tres souple": 4, "collant": 5, "lourd": 5, "tres lourd": 6}
FAILED = {"DISQUALIFIE", "DISQUALIFIE_POUR_ALLURE_IRREGULIERE", "DISQUALIFIE_POTEAU_GALOP",
          "ARRETE", "TOMBE", "DISTANCE"}
DISQUALIFIED = {"DISQUALIFIE", "DISQUALIFIE_POUR_ALLURE_IRREGULIERE", "DISQUALIFIE_POTEAU_GALOP"}
# Features computed from the odds: left out of the model without odds.
ODDS = ("f_mkt_logp", "f_outsider")


def outsider(o):
    """Rank outsiders (odds above 30) get far fewer places than their odds suggest: log(odds / 30), else 0."""
    return max(0.0, math.log(o / 30)) if o else None


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
            "dq": status in DISQUALIFIED,
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
    """Unique past rides per jockey: (date of the ride, date it became known, top 3).
    The same ride can be listed in several files (one per later Quinté of the horse); it is known
    from the first race date whose file lists it."""
    first = {}
    for p in past:
        if not p["jockey"]:
            continue
        key = (p["nom"], p["past_date"], p["prix"])
        known = date.fromisoformat(p["race_id"][:10])
        if key not in first or known < first[key][1]:
            ok = p["statusArrivee"] in ("PLACE", "NON_PLACE") and num(p["place"]) is not None
            first[key] = (p["jockey"], known, date.fromisoformat(p["past_date"]), ok and num(p["place"]) <= 3)
    rides = defaultdict(list)
    for jockey, known, ride_day, top3 in first.values():
        rides[jockey].append((ride_day, known, top3))
    return rides


def prior_as_of(rides):
    """day -> average top-3 rate of the rides known by that day (the shrinkage target)."""
    by_known = defaultdict(lambda: [0, 0])
    for v in rides.values():
        for _, known, top3 in v:
            by_known[known][0] += top3
            by_known[known][1] += 1
    days, hits, n, out = sorted(by_known), 0, 0, {}
    for d in days:
        hits, n = hits + by_known[d][0], n + by_known[d][1]
        out[d] = hits / n
    return days, out


def form_index(all_runs):
    """(kind, name, group) -> (dates, cumulative top-3 finishes, cumulative expected top-3) over all
    French races; expected = 3 / starters, so big and small fields count fairly."""
    rides, first = defaultdict(list), None
    for r in all_runs:
        n = int(num(r["partants"]) or 0)
        if n < 4:
            continue
        d = date.fromisoformat(r["date"])
        first = d if first is None or d < first else first
        place = num(r["place"])
        group = "TROT" if (r["specialite"] or "").startswith("TROT") else "GALOP"
        for kind, name in (("jockey", r["driver"]), ("trainer", r["entraineur"])):
            if name:
                rides[(kind, name, group)].append((d, place is not None and place <= 3, 3 / n))
    index = {}
    for key, v in rides.items():
        v.sort()
        cum_a, cum_e = [0.0], [0.0]
        for _, a, e in v:
            cum_a.append(cum_a[-1] + a)
            cum_e.append(cum_e[-1] + e)
        index[key] = ([d for d, _, _ in v], cum_a, cum_e)
    return index, first


def form_ratio(index, key, day, days, k=3.0):
    """Smoothed log(actual / expected top-3 finishes) and log(1 + runs) in [day - days, day)."""
    if key not in index:
        return 0.0, 0.0
    dates, cum_a, cum_e = index[key]
    lo, hi = bisect_left(dates, day - timedelta(days=days)), bisect_left(dates, day)
    a, e = cum_a[hi] - cum_a[lo], cum_e[hi] - cum_e[lo]
    return math.log((a + k) / (e + k)), math.log1p(hi - lo)


def compute(runners, past, all_runs=None):
    """Feature rows for every starter, from runner and past-run rows shaped like the CSVs (all strings).
    all_runs (rows of data/all_runs.csv) adds jockey/trainer form across all French races."""
    runners = [r for r in runners if r["statut"] == "PARTANT"]
    forms, form_start = form_index(all_runs) if all_runs else ({}, None)
    runs_of = past_by_runner(past)
    rides = jockey_index(past)
    prior_days, prior_at = prior_as_of(rides)

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
        odds = {x["numPmu"]: num(x["cote_reference"]) for x in field}
        known_days = [d for d in prior_days if d <= day]
        jockey_prior = prior_at[known_days[-1]] if known_days else 0.3
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
            f["f_dq_rate"] = rate(sum(x["dq"] for x in last5), len(last5), 0.1) if last5 else None
            f["f_last_failed"] = int(runs[0]["failed"]) if runs else None
            f["f_long_break"] = int(runs[0]["days"] >= 90) if runs else None
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
            # two signals, not one -1/0/+1 scale: a negative opinion matters far more (research/signal_eval.py)
            avis = r["avisEntraineur"] if r["avisEntraineur"] in ("POSITIF", "NEGATIF", "NEUTRE") else None
            f["f_avis_neg"] = int(avis == "NEGATIF") if avis else None
            f["f_avis_pos"] = int(avis == "POSITIF") if avis else None
            f["f_driver_change"] = int(r["driverChange"] == "True")

            nc, nv, npl = num(r["nombreCourses"]) or 0, num(r["nombreVictoires"]) or 0, num(r["nombrePlaces"]) or 0
            f["f_career_win"] = rate(nv, nc, 0.1, k=5)
            f["f_career_place"] = rate(npl, nc, 0.35, k=5)
            f["f_log_gains_per_run"] = math.log1p((num(r["gainsCarriere"]) or 0) / 100 / max(nc, 1))
            f["f_log_gains_year"] = math.log1p((num(r["gainsAnneeEnCours"]) or 0) / 100)

            jr = [t3 for dt, known, t3 in rides.get(r["driver"], []) if known <= day and day - timedelta(days=90) <= dt < day]
            f["f_jockey_top3"] = rate(sum(jr), len(jr), jockey_prior, k=10)
            f["f_jockey_rides"] = math.log1p(len(jr))
            th = [t5 for dt, t5 in trainer_hist.get(r["entraineur"], []) if dt < day]
            f["f_trainer_top5"] = rate(sum(th), len(th), 5 / 15, k=5)

            if all_runs is not None:  # valid once 30 days of all-race history exist before the race
                ok = form_start is not None and day >= form_start + timedelta(days=30)
                group = "TROT" if fam in ("ATTELE", "MONTE") else "GALOP"
                for kind, name, spans in (("jockey", r["driver"], (90,)), ("trainer", r["entraineur"], (90, 30))):
                    for span in spans:
                        ratio, runs = form_ratio(forms, (kind, name, group), day, span) if ok else (None, None)
                        f[f"f_{kind}_form{span}"] = ratio
                        if span == 90:
                            f[f"f_{kind}_runs{span}"] = runs

            o = odds[r["numPmu"]]
            f["f_mkt_logp"] = math.log((1 / o) / inv) if o and inv else None
            f["f_outsider"] = outsider(o)

            rows.append({"race_id": race_id, "date": r["date"], "numPmu": r["numPmu"], "nom": r["nom"],
                         "specialite": r["specialite"], "partants": n, "res_place": r["res_place"],
                         "res_top5": r["res_top5"], "cote_reference": odds[r["numPmu"]], **f})
    return [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in rows]


def main():
    all_runs = load("all_runs.csv") if (DATA / "all_runs.csv").exists() else None
    rows = compute(load("runners.csv"), load("past_runs.csv"), all_runs)
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
