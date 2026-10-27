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

Personal use only: one request per second, no parallel calls.
"""
import csv
import json
import math
import random
import sys
import time
from datetime import date, datetime

import app_export
import backtest as bt
import build_dataset as bd
import features as ft
from fetch_programmes import HOSTS, PAUSE_S, fetch
from paths import DATA

RAW = DATA / "raw"
DAILY = DATA / "daily"

WARNING = ("Les pourcentages sont des estimations du modèle, pas des certitudes. Sur les Quintés de test, "
           "aucune stratégie (ni l'IA, ni les favoris) n'a été rentable : à jouer pour le plaisir.")

# Features used in explanations, with the direction that makes sense (+1: more is better).
# A feature is only explained when the fitted weight has that same sign.
EXPECTED = {"f_class": 1, "f_jockey_rides": 1, "f_jockey_top3": 1, "f_draw_rel": -1, "f_avis": 1,
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


def settle(today):
    """Add the result to earlier daily files. Returns True if new raw files were saved."""
    saved = False
    for f in sorted(DAILY.glob("????-??-??.json")):
        tip = json.loads(f.read_text(encoding="utf-8"))
        if tip.get("resultat") or not tip.get("course") or f.stem >= today:
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
            rank, eur = bt.ticket_return(tip["prono"], rows)
            tip["resultat"] = {"arrivee": arrival, "chevaux_trouves": len(set(arrival) & set(tip["prono"])),
                               "rang": rank, "gain_euros": round(bt.STAKE * eur, 2), "mise_euros": bt.STAKE}
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
    if name == "f_avis" and r["avisEntraineur"] in ("POSITIF", "NEGATIF"):
        return "Avis positif de l'entraîneur" if r["avisEntraineur"] == "POSITIF" else "Avis négatif de l'entraîneur"
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
    mkt = feats.index("f_mkt_logp")
    sets = {"sans_cote": [i for i in range(len(feats)) if i != mkt]}
    if today["has_odds"]:
        sets["avec_cote"] = list(range(len(feats)))
    fitted = {}
    for name, cols in sets.items():
        bt.design(train + [today], cols, stats)
        w = bt.fit(train, len(cols))
        fitted[name] = (cols, w, [x[:] for x in today["X"]], bt.scores(today["X"], w))
        print(f"model '{name}' fitted on {len(train)} races")

    rng = random.Random(1)
    main_model = "avec_cote" if "avec_cote" in fitted else "sans_cote"
    s = fitted[main_model][3]
    p5 = bt.p_top5(s, rng)
    pwin = [math.exp(x - max(s)) for x in s]
    pwin = [x / sum(pwin) for x in pwin]
    p5_form = bt.p_top5(fitted["sans_cote"][3], rng)
    p5_mkt = bt.p_top5([-math.log(o) for o in today["odds"]], rng) if today["has_odds"] else None
    n = len(today["nums"])
    order = sorted(range(n), key=lambda i: -p5[i])
    prono = [today["nums"][i] for i in order[:5]]
    value = []
    if p5_mkt:  # same rule as the "Bon rapport" strategy of backtest.py
        shortlist = sorted(range(n), key=lambda i: -p5_form[i])[:8]
        picked = sorted(shortlist, key=lambda i: -p5_form[i] / max(p5_mkt[i], 1e-6))[:5]
        value = [today["nums"][i] for i in sorted(picked, key=lambda i: -p5_form[i])]

    cols, w, X, _ = fitted["sans_cote"]
    horses = []
    for rank, i in enumerate(order, 1):
        num = today["nums"][i]
        r = rows_by_num[str(num)]
        plus, minus = explain(i, X, w, cols, feats, frows[str(num)], r)
        cote = ft.num(r["cote_finale"]) or ft.num(r["cote_reference"])
        horses.append({
            **runner_info(r), "rang_ia": rank,
            "cote": cote,
            "chance_top5_estimee": round(p5[i], 3),
            "chance_gagner_estimee": round(pwin[i], 3),
            "chance_top5_marche_estimee": round(p5_mkt[i], 3) if p5_mkt else None,
            "bon_rapport": num in value,
            "atouts": plus, "points_faibles": minus,
        })
    return {
        "prono": prono, "bon_rapport": value, "chevaux": horses,
        "modele": {
            "avec_cotes": main_model == "avec_cote",
            "courses_d_entrainement": len(train),
            "note": ("Classement établi avec les cotes disponibles au moment du calcul (elles bougent jusqu'au départ)."
                     if main_model == "avec_cote" else
                     "Cotes pas encore disponibles : classement établi sans les cotes, uniquement sur la forme et le profil."),
        },
    }


def runner_info(r):
    """Descriptive fields of a runner row, for the app."""
    num = ft.num
    weight = num(r["handicapPoids"]) or num(r["poidsConditionMonte"])
    return {"num": int(r["numPmu"]), "nom": r["nom"], "jockey": r["driver"], "entraineur": r["entraineur"],
            "age": int(num(r["age"]) or 0) or None, "sexe": r["sexe"], "musique": r["musique"],
            "poids_kg": weight / 10 if weight else None, "corde": int(num(r["placeCorde"]) or 0) or None}


def main():
    iso = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    DAILY.mkdir(parents=True, exist_ok=True)
    if settle(min(iso, date.today().isoformat())):
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
