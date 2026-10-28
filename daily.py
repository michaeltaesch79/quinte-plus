"""
Phase 7: today's Quinté+ -> data/daily/YYYY-MM-DD.json (pipeline record) and app/data/today.json (for the app).

Each run:
  1. settles earlier days that have no result yet: downloads the final programme, runners and payouts
     (3 requests per day), saves them in data/raw/ like the bulk scripts, rebuilds the CSV tables, and
     records the arrival and what the 2 € prono ticket would have returned (for the scoreboard);
  2. downloads today's programme, runners and past performances (3 requests);
  3. refits the model on every finished Quinté in data/ and writes the tips with short explanations;
  4. converts the record to the app's format (app_export.py, schema in app/README.md).

Every percentage in the output is a model estimate and is labelled as such.

Usage (Python 3.9+, no extra packages):
    python daily.py               # today
    python daily.py 2026-10-28    # another day (for testing)
    python daily.py --refresh 2026-10-28   # same tips with the latest odds (no refit, one request)
    python daily.py --settle 2026-10-28    # record that day's result once the race is over

Personal use only: one request per second, no parallel calls.
"""
import csv
import json
import math
import random
import sys
import time
from datetime import date, datetime, timedelta

import app_export
import backtest as bt
import build_dataset as bd
import features as ft
from clock import paris_now
from fetch_programmes import HOSTS, PAUSE_S, fetch
from paths import DATA

RAW = DATA / "raw"
DAILY = DATA / "daily"

WARNING = ("Les pourcentages sont des estimations du modèle, pas des certitudes. Sur les Quintés de test, "
           "aucune stratégie (ni l'IA, ni les favoris) n'a été rentable : à jouer pour le plaisir.")

# Model settings (research/REPORT.md): L2 penalty per model and stage discounts (2nd-5th places less
# predictable than the win), chosen on the validation months. No feature group is dropped: the groups
# dropped on validation made the test months worse.
MODELS = {"sans_cote": {"l2": 20.0}, "avec_cote": {"l2": 10.0}}
DROPPED = set()

# Features used in explanations, with the direction that makes sense (+1: more is better).
# A feature is only explained when the fitted weight has that same sign.
EXPECTED = {"f_class": 1, "f_jockey_rides": 1, "f_jockey_top3": 1, "f_draw_rel": -1, "f_avis_pos": 1, "f_avis_neg": -1,
            "f_weight_rel": 1, "f_career_win": 1, "f_last_place": -1, "f_form_place": -1, "f_form_top3": 1,
            "f_form_win": 1, "f_going_top3": 1, "f_dist_top3": 1, "f_track_top3": 1, "f_trainer_top5": 1,
            "f_log_gains_year": 1, "f_days_since": -1, "f_unshod4": 1, "f_dist_handicap": -1}


def get(path):
    """One feed request (first host that answers with JSON), then the usual pause."""
    for host in HOSTS:
        _, body = fetch(host + path)
        time.sleep(PAUSE_S)
        if body:
            try:
                return json.loads(body), body
            except ValueError:
                continue
    return None, None


def find_quinte(prog):
    for reunion in prog["programme"]["reunions"]:
        for course in reunion["courses"]:
            if any(p.get("codePari") == "QUINTE_PLUS" for p in course.get("paris", [])):
                return reunion, course
    return None, None


def csv_row(d):
    """Same shape as a row read back from the CSV files: every value a string, None -> ''."""
    return {k: "" if v is None else str(v) for k, v in d.items()}


def load_csv(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- 1. settle earlier days

def saved_or_get(target, path):
    """The final file from data/raw/ if already downloaded, else fetch it and save it there.
    Returns (parsed JSON or None, True if a new file was saved)."""
    if target.exists():
        return json.loads(target.read_bytes()), False
    data, body = get(path)
    if not data:
        return None, False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    return data, True


def ticket_result(nums, arrival, rows):
    rank, eur = bt.ticket_return(nums, rows)
    return {"chevaux_trouves": len(set(arrival) & set(nums)), "rang": rank,
            "gain_euros": round(bt.STAKE * eur, 2), "mise_euros": bt.STAKE}


def settle(last):
    """Add the result to the daily files dated up to `last` (included) whose race is over.
    Returns True if new raw files were saved."""
    saved = False
    for f in sorted(DAILY.glob("????-??-??.json")):
        tip = json.loads(f.read_text(encoding="utf-8"))
        res = tip.get("resultat") or {}
        if res.get("annulee") or "bon_rapport" in res or not tip.get("course") or f.stem > last:
            continue
        iso, r, c = f.stem, tip["course"]["reunion"], tip["course"]["course"]
        key = f"{iso[8:10]}{iso[5:7]}{iso[:4]}"
        race_file = f"{iso}/R{r}C{c}.json"
        rap, new = saved_or_get(RAW / "rapports" / race_file, f"{key}/R{r}/C{c}/rapports-definitifs")
        if not rap:
            print(f"{iso}: payouts not available yet")
            continue
        saved |= new
        prog, new = saved_or_get(RAW / "programme" / f"{iso}.json", key)
        saved |= new
        if not prog:
            continue
        reunion, course = find_quinte(prog)
        if course is None:
            continue
        _, new = saved_or_get(RAW / "participants" / race_file, f"{key}/R{r}/C{c}/participants")
        saved |= new
        if course.get("statut") == "COURSE_ANNULEE":
            tip["resultat"] = {"annulee": True}
        else:
            rows = [(p["pari"], [int(x) for x in p["combinaison"].split("-")], p["euros_pour_1_euro"])
                    for p in bd.payout_rows(tip["course"]["race_id"], rap) if p["pari"] in bt.BET_RANKS]
            arrival = [n for group in course.get("ordreArrivee", []) for n in group][:5]
            tip["resultat"] = {"arrivee": arrival, **ticket_result(tip["prono"], arrival, rows),
                               "bon_rapport": (ticket_result(tip["bon_rapport"], arrival, rows)
                                               if tip.get("bon_rapport") else None)}
        f.write_text(json.dumps(tip, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{iso}: result recorded {tip['resultat']}")
    return saved


# ---------------------------------------------------------------- 2. today's race

def today_rows(iso):
    """Fetch and save today's files; return (race, runner rows, past rows) or None if no Quinté."""
    key = f"{iso[8:10]}{iso[5:7]}{iso[:4]}"
    snap = DAILY / "raw" / iso
    snap.mkdir(parents=True, exist_ok=True)
    prog, body = get(key)
    if not prog:
        sys.exit(f"{iso}: programme not available")
    (snap / "programme.json").write_bytes(body)
    reunion, course = find_quinte(prog)
    if course is None:
        return None
    race = bd.race_row(iso, reunion, course)
    r, c = race["reunion"], race["course"]
    part, body = get(f"{key}/R{r}/C{c}/participants")
    if not part:
        sys.exit(f"{iso}: runners not available")
    (snap / "participants.json").write_bytes(body)
    perf, body = get(f"{key}/R{r}/C{c}/performances-detaillees/pretty")
    if perf:  # history as of the race date: also the right file for the dataset
        (snap / "performances.json").write_bytes(body)
        target = RAW / "performances" / iso / f"R{r}C{c}.json"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
    ps = part["participants"]
    partants = sum(p["statut"] == "PARTANT" for p in ps)
    runners = [csv_row({**race, **bd.runner_row(p, partants)}) for p in ps]
    past = [csv_row(x) for x in bd.past_rows(race["race_id"], iso, perf)] if perf else []
    return race, runners, past


# ---------------------------------------------------------------- 3. model and tips

def explain(i, X, w, cols, feats, frow, runner):
    """Up to 3 strengths and 2 weaknesses: the features that move this horse most against the field."""
    out = []
    for j, c in enumerate(cols):
        name = feats[c]
        if name not in EXPECTED or w[j] * EXPECTED[name] <= 0:
            continue
        mean = sum(x[j] for x in X) / len(X)
        contrib = w[j] * (X[i][j] - mean)
        text = phrase(name, X[i][j] > mean, frow, runner)
        if text and abs(contrib) >= 0.08:
            out.append((contrib, {"facteur": name, "texte": text}))
    out.sort(key=lambda x: -x[0])
    return [t for v, t in out if v > 0][:3], [t for v, t in sorted(out, key=lambda x: x[0]) if v < 0][:2]


def phrase(name, high, f, r):
    num = ft.num
    rider = "Driver" if r["specialite"].startswith("TROT") else "Jockey"
    if name == "f_class":
        return "A déjà couru des courses mieux dotées" if high else "Monte de catégorie (courses précédentes moins dotées)"
    if name == "f_jockey_rides":
        return f"{rider} très demandé ces 3 derniers mois" if high else f"{rider} peu présent ces 3 derniers mois"
    if name == "f_jockey_top3":
        return f"{rider} en réussite ces 3 derniers mois" if high else f"{rider} en manque de réussite ces 3 derniers mois"
    if name == "f_draw_rel" and r["placeCorde"]:
        return f"Numéro de corde élevé ({r['placeCorde']})" if high else f"Bon numéro de corde ({r['placeCorde']})"
    if name == "f_avis_pos" and high:
        return "Avis positif de l'entraîneur"
    if name == "f_avis_neg" and high:
        return "Avis négatif de l'entraîneur"
    if name == "f_weight_rel" and r["handicapPoids"]:
        kg = f"{num(r['handicapPoids']) / 10:g} kg".replace(".", ",")
        return (f"Parmi les plus chargés ({kg}) : le handicapeur le juge parmi les meilleurs" if high
                else f"Parmi les moins chargés ({kg}) : moins bien jugé par le handicapeur")
    if name == "f_career_win":
        v, n = int(num(r["nombreVictoires"]) or 0), int(num(r["nombreCourses"]) or 0)
        rec = f"{v} victoire{'s' if v > 1 else ''} en {n} course{'s' if n > 1 else ''}"
        return f"Gagne souvent ({rec})" if high else f"Gagne peu ({rec})"
    if name == "f_last_place" and f["f_last_place"]:
        p = int(float(f["f_last_place"]))
        txt = "hors des 9 premiers ou non classé" if p >= 10 else ("1er" if p == 1 else f"{p}e")
        return f"Dernière course décevante ({txt})" if high else f"Bonne dernière course ({txt})"
    if name == "f_form_place":
        return "Forme moyenne sur ses 5 dernières courses" if high else "Bonne forme sur ses 5 dernières courses"
    if name == "f_form_top3":
        return "Souvent dans les 3 premiers récemment" if high else "Rarement dans les 3 premiers récemment"
    if name == "f_form_win":
        return "A gagné récemment" if high else None
    if name == "f_going_top3" and r["terrain"]:
        return (f"Bons résultats sur ce terrain ({r['terrain'].lower()})" if high
                else f"Résultats moyens sur ce terrain ({r['terrain'].lower()})")
    if name == "f_dist_top3":
        return "Bons résultats sur cette distance" if high else "Résultats moyens sur cette distance"
    if name == "f_track_top3":
        return "Bons résultats sur cet hippodrome" if high else "Résultats moyens sur cet hippodrome"
    if name == "f_trainer_top5":
        return "Entraîneur qui réussit dans les Quintés" if high else "Entraîneur peu en réussite dans les Quintés"
    if name == "f_log_gains_year":
        return "Beaucoup de gains cette année" if high else "Peu de gains cette année"
    if name == "f_days_since" and f["f_days_since"]:
        d = int(float(f["f_days_since"]))
        return f"Revient d'une coupure ({d} jours sans courir)" if high else f"A couru récemment (il y a {d} jours)"
    if name == "f_unshod4" and f["f_unshod4"] == "1":
        return "Déferré des 4 pieds"
    if name == "f_dist_handicap" and high and f["f_dist_handicap"]:
        return f"Part avec un recul de {round(float(f['f_dist_handicap']) * 25)} m"
    return None


def model_cols(name, feats):
    return [i for i, f in enumerate(feats) if f not in DROPPED and (name == "avec_cote" or f not in ft.ODDS)]


def unpack(fitted):
    """(weights, stage factors); records written before the stage discounts hold weights only."""
    return (fitted[0], fitted[1]) if fitted and isinstance(fitted[0], list) else (fitted, [1.0] * 5)


def ranking(nums, raw, feats, stats, weights, odds):
    """Prono, bon rapport and estimated chances of one race from fitted weights and the given odds
    (None where unknown). The market features are recomputed from these odds as in features.py."""
    mkt = feats.index("f_mkt_logp")
    has_odds = all(odds)
    inv = sum(1 / o for o in odds if o)
    raw = [row[:mkt] + [math.log((1 / o) / inv) if o else None] + row[mkt + 1:] for row, o in zip(raw, odds)]
    if "f_outsider" in feats:  # absent from records cached before it was added
        out = feats.index("f_outsider")
        raw = [row[:out] + [ft.outsider(o)] + row[out + 1:] for row, o in zip(raw, odds)]
    s, lam = {}, {}
    for name, fitted in weights.items():
        w, lam[name] = unpack(fitted)
        race = {"raw": raw}
        bt.design([race], model_cols(name, feats), stats)
        s[name] = bt.scores(race["X"], w)
    main = "avec_cote" if has_odds else "sans_cote"
    rng = random.Random(1)
    p5 = bt.p_top5(s[main], rng, lam[main])
    pwin = [math.exp(x - max(s[main])) for x in s[main]]  # the win stage has factor 1
    pwin = [x / sum(pwin) for x in pwin]
    p5_form = bt.p_top5(s["sans_cote"], rng, lam["sans_cote"])
    p5_mkt = bt.p_top5([-math.log(o) for o in odds], rng) if has_odds else None
    n = len(nums)
    order = sorted(range(n), key=lambda i: -p5[i])
    value = []
    if p5_mkt:  # same rule as the "Bon rapport" strategy of backtest.py
        shortlist = sorted(range(n), key=lambda i: -p5_form[i])[:8]
        picked = sorted(shortlist, key=lambda i: -p5_form[i] / max(p5_mkt[i], 1e-6))[:5]
        value = [nums[i] for i in sorted(picked, key=lambda i: -p5_form[i])]
    return {"main": main, "order": order, "prono": [nums[i] for i in order[:5]], "value": value,
            "p5": p5, "pwin": pwin, "p5_mkt": p5_mkt}


def chances(i, rk):
    return {"chance_top5_estimee": round(rk["p5"][i], 3), "chance_gagner_estimee": round(rk["pwin"][i], 3),
            "chance_top5_marche_estimee": round(rk["p5_mkt"][i], 3) if rk["p5_mkt"] else None}


def model_note(with_odds, at):
    return (f"Classement établi avec les cotes de {at} (elles bougent jusqu'au départ)." if with_odds else
            "Cotes pas encore disponibles : classement établi sans les cotes, uniquement sur la forme et le profil.")


def tips(race, runners, past):
    hist_runners = [r for r in load_csv("runners.csv") if r["race_id"] != race["race_id"]]
    hist_past = [p for p in load_csv("past_runs.csv") if p["race_id"] != race["race_id"]]
    rows = ft.compute(hist_runners + runners, hist_past + past)
    train, feats = bt.load_races([r for r in rows if r["race_id"] != race["race_id"]])
    today, _ = bt._races([r for r in rows if r["race_id"] == race["race_id"]], keep_unfinished=True)
    today = today[0]
    frows = {r["numPmu"]: r for r in rows if r["race_id"] == race["race_id"]}
    rows_by_num = {r["numPmu"]: r for r in runners}

    stats = bt.fit_stats(train, len(feats))
    weights = {}
    for name in ("sans_cote", "avec_cote"):  # both, so a later refresh can use the odds without refitting
        bt.design(train, model_cols(name, feats), stats)
        w, lam = bt.fit_pl(train, len(model_cols(name, feats)), l2=MODELS[name]["l2"], discount=True)
        weights[name] = [w, lam]
        print(f"model '{name}' fitted on {len(train)} races")

    nums = today["nums"]
    odds = [ft.num(rows_by_num[str(n)]["cote_reference"]) or ft.num(rows_by_num[str(n)]["cote_finale"]) for n in nums]
    rk = ranking(nums, today["raw"], feats, stats, weights, odds)
    form = {"raw": today["raw"]}
    cols = model_cols("sans_cote", feats)
    bt.design([form], cols, stats)
    horses = []
    for rank, i in enumerate(rk["order"], 1):
        num = nums[i]
        r = rows_by_num[str(num)]
        plus, minus = explain(i, form["X"], weights["sans_cote"][0], cols, feats, frows[str(num)], r)
        horses.append({
            **runner_info(r), "rang_ia": rank, "cote": odds[i], **chances(i, rk),
            "bon_rapport": num in rk["value"], "atouts": plus, "points_faibles": minus,
        })
    at = f"{paris_now():%Hh%M}"
    return {
        "prono": rk["prono"], "bon_rapport": rk["value"], "chevaux": horses,
        "cotes_maj": at if rk["main"] == "avec_cote" else None,
        "modele": {
            "avec_cotes": rk["main"] == "avec_cote",
            "courses_d_entrainement": len(train),
            "note": model_note(rk["main"] == "avec_cote", at),
        },
        "cache": {"feats": feats, "stats": stats, "weights": weights, "nums": nums, "raw": today["raw"]},
    }


def refresh(iso):
    """Re-rank the prono and the bon rapport of a daily record with the latest odds: one request, no refit."""
    f = DAILY / f"{iso}.json"
    rec = json.loads(f.read_text(encoding="utf-8"))
    c = rec.get("course")
    if not c or "cache" not in rec:
        sys.exit(f"{iso}: nothing to refresh")
    if f"{paris_now():%Y-%m-%d %H:%M}" >= c["heure_depart"]:
        print(f"{iso}: race started, odds no longer refreshed")
        return
    key = f"{iso[8:10]}{iso[5:7]}{iso[:4]}"
    part, body = get(f"{key}/R{c['reunion']}/C{c['course']}/participants")
    if not part:
        sys.exit(f"{iso}: runners not available")
    snap = DAILY / "raw" / iso  # timestamped odds snapshots, to study odds movement later
    snap.mkdir(parents=True, exist_ok=True)
    (snap / f"participants_{paris_now():%H%M}.json").write_bytes(body)
    live ={p["numPmu"]: p for p in part["participants"] if p.get("statut") == "PARTANT"}
    cache = rec["cache"]
    keep = [i for i, n in enumerate(cache["nums"]) if n in live]
    nums = [cache["nums"][i] for i in keep]
    odds = [(live[n].get("dernierRapportDirect") or live[n].get("dernierRapportReference") or {}).get("rapport")
            for n in nums]
    rk = ranking(nums, [cache["raw"][i] for i in keep], cache["feats"], cache["stats"], cache["weights"], odds)

    by_num = {h["num"]: h for h in rec["chevaux"]}
    for n in set(by_num) - set(nums):  # withdrawn since the morning
        h = by_num.pop(n)
        rec["non_partants"].append({k: h[k] for k in ("num", "nom", "jockey", "entraineur", "age", "sexe",
                                                      "musique", "poids_kg", "corde")})
    for rank, i in enumerate(rk["order"], 1):
        by_num[nums[i]].update({"rang_ia": rank, "cote": odds[i], **chances(i, rk)})
    if rk["value"]:  # keeps the last list if the odds disappear
        rec.setdefault("bon_rapport_initial", rec["bon_rapport"])
        rec["bon_rapport"] = rk["value"]
        for n, h in by_num.items():
            h["bon_rapport"] = n in rk["value"]
    at = f"{paris_now():%Hh%M}"
    rec.setdefault("prono_initial", rec["prono"])
    rec["prono"] = rk["prono"]
    rec["chevaux"] = sorted(by_num.values(), key=lambda h: h["rang_ia"])
    rec["course"]["partants"] = len(nums)
    if rk["main"] == "avec_cote":
        rec["cotes_maj"] = at
    rec["modele"].update({"avec_cotes": rk["main"] == "avec_cote", "note": model_note(rk["main"] == "avec_cote", at)})
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    app_export.write(iso)
    print(f"{iso}: odds of {at}, prono {'-'.join(map(str, rec['prono']))} "
          f"(was {'-'.join(map(str, rec['prono_initial']))})")


def runner_info(r):
    """Descriptive fields of a runner row, for the app."""
    num = ft.num
    weight = num(r["handicapPoids"]) or num(r["poidsConditionMonte"])
    return {"num": int(r["numPmu"]), "nom": r["nom"], "jockey": r["driver"], "entraineur": r["entraineur"],
            "age": int(num(r["age"]) or 0) or None, "sexe": r["sexe"], "musique": r["musique"],
            "poids_kg": weight / 10 if weight else None, "corde": int(num(r["placeCorde"]) or 0) or None}


def main():
    if sys.argv[1:2] == ["--refresh"]:
        return refresh(sys.argv[2] if len(sys.argv) > 2 else paris_now().date().isoformat())
    if sys.argv[1:2] == ["--settle"]:
        iso = sys.argv[2] if len(sys.argv) > 2 else paris_now().date().isoformat()
        if settle(iso):
            bd.main()
        return app_export.write(iso)
    iso = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    DAILY.mkdir(parents=True, exist_ok=True)
    if settle((date.fromisoformat(min(iso, date.today().isoformat())) - timedelta(days=1)).isoformat()):
        bd.main()
    got = today_rows(iso)
    out = {"genere_le": datetime.now().isoformat(timespec="seconds"), "date": iso, "avertissement": WARNING}
    if got is None:
        out.update({"course": None, "message": "Pas de Quinté+ trouvé au programme de ce jour."})
    else:
        race, runners, past = got
        out["course"] = {k: race[k] for k in ("race_id", "reunion", "course", "heure_depart", "hippodrome", "prix",
                                              "discipline", "specialite", "distance", "typePiste", "terrain",
                                              "categorieParticularite", "montantPrix", "meteo", "temperature")}
        out["course"]["partants"] = sum(r["statut"] == "PARTANT" for r in runners)
        out["non_partants"] = [runner_info(r) for r in runners if r["statut"] != "PARTANT"]
        out.update(tips(race, runners, past))
    (DAILY / f"{iso}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    target = app_export.write(iso)
    if out["course"]:
        c = out["course"]
        print(f"\n{iso} {c['hippodrome']} R{c['reunion']}C{c['course']} {c['heure_depart'][11:]} "
              f"{c['specialite']} {c['distance']} m, {c['partants']} partants")
        print(f"Prono: {'-'.join(map(str, out['prono']))}   Bon rapport: {'-'.join(map(str, out['bon_rapport'])) or '-'}   "
              f"({out['modele']['note']})")
    print(f"Written: {DAILY / (iso + '.json')} and {target}")


if __name__ == "__main__":
    main()
