"""
Data audit of the Quinté+ dataset: identifiers, duplicates, scratches, non-finishers and
disqualifications, missing values, odds timestamps, payout units and rule consistency.

Reads data/raw/ (for the feed timestamps) and the CSV tables built by build_dataset.py.
Writes research/results/audit.json and prints a readable summary. No downloads.

Usage (plain Python 3.9+):
    python research/audit.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_dataset as bd  # noqa: E402
from paths import DATA  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
QUINTE_CATS = ("QUINTE_PLUS:ORDRE", "QUINTE_PLUS:ORDRE_TIRELIRE", "QUINTE_PLUS:DESORDRE",
               "QUINTE_PLUS:BONUS_4SUR5", "QUINTE_PLUS:BONUS_4", "QUINTE_PLUS:BONUS_3")


def load(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def quantiles(v):
    if not v:
        return None
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return {"n": len(v), "min": v[0], "p05": q(0.05), "median": q(0.5), "p95": q(0.95), "max": v[-1]}


def odds_timestamps():
    """Minutes between each runner's odds timestamp and the scheduled start (negative = before)."""
    ref, direct, kinds = [], [], Counter()
    for iso, reunion, course in bd.quinte_races():
        f = bd.RAW / "participants" / iso / f"R{reunion['numOfficiel']}C{course['numOrdre']}.json"
        if not f.exists():
            continue
        start = course["heureDepart"]
        for p in json.loads(f.read_bytes())["participants"]:
            for key, out in (("dernierRapportReference", ref), ("dernierRapportDirect", direct)):
                r = p.get(key)
                if r and r.get("dateRapport"):
                    out.append(round((r["dateRapport"] - start) / 60000, 1))
                    kinds[(key, r.get("typePari"), r.get("typeRapport"))] += 1
    return {"reference_minutes_vs_start": quantiles(ref), "direct_minutes_vs_start": quantiles(direct),
            "share_reference_after_start_pct": pct(sum(x > 0 for x in ref), len(ref)),
            "kinds": {" / ".join(map(str, k)): v for k, v in kinds.items()}}


def payout_units():
    """Raw rapports: PourUneMiseDeBase == PourUnEuro x miseBase/100, base stake and categories over time."""
    bad_units, mise, refunded, cats_by_month = 0, Counter(), 0, defaultdict(set)
    for iso, reunion, course in bd.quinte_races():
        f = bd.RAW / "rapports" / iso / f"R{reunion['numOfficiel']}C{course['numOrdre']}.json"
        if not f.exists():
            continue
        for bet in json.loads(f.read_bytes()):
            if bet["typePari"] != "QUINTE_PLUS":
                continue
            mise[bet["miseBase"]] += 1
            refunded += bool(bet.get("rembourse"))
            for r in bet["rapports"]:
                if abs(r["dividendePourUneMiseDeBase"] - r["dividendePourUnEuro"] * bet["miseBase"] / 100) > 1:
                    bad_units += 1
                cats_by_month[iso[:7]].add(bd.bet_kind("QUINTE_PLUS", r["libelle"]))
    return {"base_stake_cents": dict(mise), "unit_mismatches": bad_units, "races_flagged_rembourse": refunded,
            "categories_by_month": {m: sorted(c) for m, c in sorted(cats_by_month.items())}}


def main():
    runners, past, payouts = load("runners.csv"), load("past_runs.csv"), load("payouts.csv")
    races = defaultdict(list)
    for r in runners:
        races[r["race_id"]].append(r)
    rep = {}

    # identifiers and duplicates
    keys = Counter((r["race_id"], r["numPmu"]) for r in runners)
    horse_keys = Counter((r["race_id"], r["idCheval"]) for r in runners)
    names_per_id, ids_per_name = defaultdict(set), defaultdict(set)
    for r in runners:
        names_per_id[r["idCheval"]].add(r["nom"])
        ids_per_name[r["nom"]].add(r["idCheval"])
    rep["identifiers"] = {
        "runner_rows": len(runners), "races": len(races),
        "duplicate_race_numPmu": sum(v > 1 for v in keys.values()),
        "duplicate_race_idCheval": sum(v > 1 for v in horse_keys.values()),
        "idCheval_with_several_names": sum(len(v) > 1 for v in names_per_id.values()),
        "names_with_several_idCheval": sum(len(v) > 1 for v in ids_per_name.values()),
        "distinct_horses": len(names_per_id),
    }

    # race composition, products, statuses
    by_spec = Counter(f[0]["specialite"] for f in races.values())
    rep["races"] = {
        "by_specialite": dict(by_spec),
        "by_country": dict(Counter(f[0]["pays"] for f in races.values())),
        "statut_course": dict(Counter(f[0]["statut_course"] for f in races.values())),
        "starters": quantiles([sum(x["statut"] == "PARTANT" for x in f) for f in races.values()]),
        "first_date": min(races)[:10], "last_date": max(races)[:10],
    }

    # scratches, non-finishers, disqualifications, dead heats
    starters = [r for r in runners if r["statut"] == "PARTANT"]
    np_ = [r for r in runners if r["statut"] != "PARTANT"]
    unplaced = [r for r in starters if not r["res_place"]]
    top5_sizes = Counter(sum(x["res_top5"] == "1" for x in f) for f in races.values())
    dead_heat_races = [rid for rid, f in races.items()
                       if len([x["res_place"] for x in f if x["res_place"]]) != len({x["res_place"] for x in f if x["res_place"]})]
    rep["outcomes"] = {
        "starters": len(starters), "non_runners": len(np_),
        "non_runners_with_reference_odds (late scratches)": sum(bool(r["cote_reference"]) for r in np_),
        "starters_without_place (non-finishers, DQ, falls)": len(unplaced),
        "starters_without_place_pct": pct(len(unplaced), len(starters)),
        "unplaced_by_specialite_pct": {s: pct(sum(1 for r in unplaced if r["specialite"] == s),
                                             sum(1 for r in starters if r["specialite"] == s)) for s in by_spec},
        "incident_codes": dict(Counter(r["res_incident"] or "(none)" for r in unplaced).most_common(12)),
        "races_by_number_of_top5_runners": dict(top5_sizes),
        "dead_heat_races": dead_heat_races,
    }

    # missing values among starters, by discipline
    cols = ["cote_reference", "cote_finale", "musique", "driver", "entraineur", "handicapPoids", "handicapValeur",
            "handicapDistance", "placeCorde", "deferre", "avisEntraineur", "terrain", "nombreCourses", "gainsAnneeEnCours"]
    rep["missing_pct_among_starters"] = {
        s: {c: pct(sum(1 for r in starters if r["specialite"] == s and r[c] == ""), sum(1 for r in starters if r["specialite"] == s))
            for c in cols} for s in by_spec}
    rep["races_with_reference_odds_for_every_starter_pct"] = pct(
        sum(all(x["cote_reference"] for x in f if x["statut"] == "PARTANT") for f in races.values()), len(races))

    # odds timing
    rep["odds_timing"] = odds_timestamps()

    # past runs: strictly before the race, duplicates
    pk = Counter((p["race_id"], p["numPmu"], p["past_date"], p["prix"]) for p in past)
    rep["past_runs"] = {
        "rows": len(past), "on_or_after_race_date": sum(int(p["jours_avant"]) <= 0 for p in past),
        "min_days_before": min(int(p["jours_avant"]) for p in past),
        "duplicate_rows": sum(v > 1 for v in pk.values()),
        "status_counts": dict(Counter(p["statusArrivee"] for p in past).most_common()),
    }
    with_hist = {(p["race_id"], p["numPmu"]) for p in past}
    rep["past_runs"]["starters_without_history"] = sum((r["race_id"], r["numPmu"]) not in with_hist for r in starters)

    # payouts: units, categories, consistency with the arrival
    q = [p for p in payouts if p["pari"] in QUINTE_CATS]
    by_race = defaultdict(list)
    for p in q:
        by_race[p["race_id"]].append(p)
    no_desordre = [rid for rid in races if not any(p["pari"] == "QUINTE_PLUS:DESORDRE" for p in by_race.get(rid, []))]
    zero_ordre = sorted({p["race_id"] for p in q if "ORDRE" in p["pari"] and "DES" not in p["pari"] and float(p["euros_pour_1_euro"]) == 0})
    mismatch = []
    for rid, f in races.items():
        arrival = [x["numPmu"] for x in sorted((x for x in f if x["res_place"] and int(x["res_place"]) <= 5),
                                                key=lambda x: int(x["res_place"]))]
        des = [p for p in by_race.get(rid, []) if p["pari"] == "QUINTE_PLUS:DESORDRE"]
        if des and rid not in dead_heat_races and set(des[0]["combinaison"].split("-")) != set(arrival):
            mismatch.append(rid)
    order_violations = 0
    for rid, rows in by_race.items():
        best = {c: max((float(p["euros_pour_1_euro"]) for p in rows if p["pari"] == c), default=None) for c in QUINTE_CATS}
        chain = [best["QUINTE_PLUS:DESORDRE"], best["QUINTE_PLUS:BONUS_4SUR5"], best["QUINTE_PLUS:BONUS_3"]]
        if all(chain) and not chain[0] >= chain[1] >= chain[2]:
            order_violations += 1
    rep["payouts"] = {
        **payout_units(),
        "rows_by_category": dict(Counter(p["pari"] for p in q)),
        "races_without_desordre_row": no_desordre,
        "races_with_zero_ordre_dividend (no winner, carried over)": zero_ordre,
        "desordre_combination_differs_from_runner_places": mismatch,
        "races_where_desordre>=bonus4sur5>=bonus3_fails": order_violations,
        "desordre_eur_per_1eur": quantiles([float(p["euros_pour_1_euro"]) for p in q if p["pari"] == "QUINTE_PLUS:DESORDRE"]),
        "bonus3_eur_per_1eur": quantiles([float(p["euros_pour_1_euro"]) for p in q if p["pari"] == "QUINTE_PLUS:BONUS_3"]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
