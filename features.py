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
  taken after the start (cote_finale);
- the morning features (MORNING) are for the model without odds only: how the bettors rated the horse
  in its earlier Quintés (reference odds of those races), form against the field, the 10-run form
  line and trot times. With the day's odds they add nothing (research/REPORT.md §15).
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
# Features for the model without odds only (adopted 2026-11-16, research/ideas.py): the day's odds already
# hold what they say, so the model with odds leaves them out.
MORNING = ("f_prev_mkt", "f_prev_runs", "f_prev_top5", "f_form_pct", "f_form_pct_best", "f_class_perf",
           "f_mus_place", "f_mus_top3", "f_mus_fail", "f_mus_win", "f_trot_gap", "f_trot_best_rel")


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


def musique_tokens(text):
    """Results of the form line, most recent first: an int place (0 = unplaced beyond 9th) or a letter
    (D disqualified, A pulled up, T fell...). Two characters per run (result, discipline); a year
    marker is "(25)" in recent files and a bare "25" in older ones."""
    out, i, s = [], 0, text or ""
    while i < len(s):
        if s[i] == "(":
            j = s.find(")", i)
            i = j + 1 if j >= 0 else len(s)
        elif s[i:i + 2].isdigit():
            i += 2
        elif s[i].isalnum() and i + 1 < len(s) and s[i + 1].isalpha():
            out.append(int(s[i]) if s[i].isdigit() else s[i].upper())
            i += 2
        else:
            i += 1
    return out


def field_share(run):
    """Finishing position as a share of the field: 0 = won, 1 = last or unplaced. None for a run not
    finished (disqualified, fell, pulled up): f_form_failed and f_dq_rate already count those."""
    if run["status"] == "PLACE" and run["place"] and run["n"] and run["n"] >= 2:
        return min(max((run["place"] - 1) / (run["n"] - 1), 0.0), 1.0)
    return 1.0 if run["status"] == "NON_PLACE" else None


def trot_times(run):
    """(seconds per km, seconds per km behind the winner capped at 4) of a past trot run; the feed gives
    centiseconds: the horse's time per km and the winner's total time."""
    rk = run["rk"] / 100 if run["rk"] else None
    ok = rk and run["t1"] and run["distance"]
    return rk, min(max(rk - run["t1"] * 10 / run["distance"], 0.0), 4.0) if ok else None


def quinte_starts(by_race):
    """Horse name -> its Quinté starts: (day, log(market win chance x runners) from the reference odds
    of that race or None, finished in the first five or None when the race has no result)."""
    starts = defaultdict(list)
    for field in by_race.values():
        odds = [num(r["cote_reference"]) for r in field]
        inv = sum(1 / o for o in odds if o)
        for r, o in zip(field, odds):
            rel = math.log((1 / o) / inv * len(field)) if o and all(odds) else None
            starts[r["nom"]].append((date.fromisoformat(r["date"]), rel, (r["res_top5"] == "1") if r["res_top5"] != "" else None))
    return starts


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
            "status": status,
            "n": num(p["nbParticipants"]),
            "rk": num(p["reductionKilometrique"]),
            "t1": num(p["tempsDuPremier"]),
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
    starts = quinte_starts(by_race)
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
        trot = fam in ("ATTELE", "MONTE")
        times = {r["numPmu"]: [trot_times(x) for x in runs_of.get((race_id, r["numPmu"]), [])
                               if x["family"] in ("ATTELE", "MONTE")][:5] if trot else [] for r in field}
        best = {k: min((rk for rk, _ in v if rk), default=None) for k, v in times.items()}
        known_best = sorted(b for b in best.values() if b)
        median_best = known_best[len(known_best) // 2] if known_best else None

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

            # morning features (MORNING). Earlier Quinté starts of the horse, last 365 days:
            prev = sorted((s for s in starts[r["nom"]] if day - timedelta(days=365) <= s[0] < day),
                          key=lambda s: s[0], reverse=True)
            rels = [rel for _, rel, _ in prev if rel is not None][:3]
            done = [t5 for _, _, t5 in prev if t5 is not None]
            f["f_prev_mkt"] = sum(rels) / len(rels) if rels else None
            f["f_prev_runs"] = math.log1p(len(prev))
            f["f_prev_top5"] = rate(sum(done), len(done), 0.3)
            # form against the field, last 5 runs
            shares = [(field_share(x), x["allocation"]) for x in last5]
            shares = [(sh, alloc) for sh, alloc in shares if sh is not None]
            f["f_form_pct"] = _mean(sh for sh, _ in shares)
            f["f_form_pct_best"] = min((sh for sh, _ in shares), default=None)
            level = [(1 - sh, math.log(alloc / prize)) for sh, alloc in shares if alloc and prize]
            weight = sum(w for w, _ in level)
            f["f_class_perf"] = sum(w * v for w, v in level) / weight if weight else None
            # the form line of the runner file, up to 10 runs (unplaced or not finished = 10)
            mus = musique_tokens(r["musique"])[:10]
            places = [x if isinstance(x, int) and x else 10 for x in mus]
            f["f_mus_place"] = sum(places) / len(places) if mus else None
            f["f_mus_top3"] = rate(sum(1 <= p <= 3 for p in places), len(mus), 0.3) if mus else None
            f["f_mus_fail"] = rate(sum(not isinstance(x, int) for x in mus), len(mus), 0.1) if mus else None
            f["f_mus_win"] = rate(sum(p == 1 for p in places), len(mus), 0.1) if mus else None
            # trot times of the last 5 trot runs (trot races only)
            gaps = [g for _, g in times[r["numPmu"]] if g is not None]
            f["f_trot_gap"] = sum(gaps) / len(gaps) if gaps else None
            b = best[r["numPmu"]]
            f["f_trot_best_rel"] = min(max(b - median_best, -5.0), 5.0) if b else None  # a few records are tens of seconds off

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
