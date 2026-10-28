"""
Phase 5: flatten the raw Quinté files into three CSV tables (no downloads).

    data/runners.csv    one row per runner per Quinté race
    data/past_runs.csv  one row per runner per past race (from performances-detaillees)
    data/payouts.csv    one row per payout line of each Quinté race
    data/all_runs.csv   one row per runner of every French race whose runners file was downloaded
                        (fetch_participants.py --france): jockey and trainer form across all races

Columns starting with res_ are only known after the race: never use them as model inputs.
Runner fields are a pre-race snapshot (checked: musique and career counts do not include
the race itself). cote_finale is taken just after the start, cote_reference 30 min before.

Usage (Python 3.9+, no extra packages):
    python build_dataset.py
"""
import csv
import json
from datetime import date, datetime, timezone

from paths import DATA as ROOT

RAW = ROOT / "raw"


def local_time(ms, offset_ms):
    """Feed timestamps are UTC ms epoch with a separate timezone offset in ms."""
    return datetime.fromtimestamp((ms + (offset_ms or 0)) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def going_value(pen):
    """penetrometre.valeurMesure is a French decimal string such as "4,4"."""
    try:
        return float(pen["valeurMesure"].replace(",", "."))
    except (KeyError, AttributeError, ValueError):
        return None


def quinte_races():
    """Yield (iso_date, reunion, course) for the Quinté+ race of each saved programme."""
    for f in sorted((RAW / "programme").glob("*.json")):
        prog = json.loads(f.read_bytes())["programme"]
        for reunion in prog["reunions"]:
            for course in reunion["courses"]:
                if any(p.get("codePari") == "QUINTE_PLUS" for p in course.get("paris", [])):
                    yield f.stem, reunion, course


def race_row(iso, reunion, course):
    pen = course.get("penetrometre") or {}
    meteo = reunion.get("meteo") or {}
    return {
        "race_id": f"{iso}_R{reunion['numOfficiel']}C{course['numOrdre']}",
        "date": iso,
        "reunion": reunion["numOfficiel"],
        "course": course["numOrdre"],
        "heure_depart": local_time(course["heureDepart"], course.get("timezoneOffset")),
        "hippodrome": reunion["hippodrome"]["libelleCourt"],
        "hippodrome_code": reunion["hippodrome"]["code"],
        "pays": reunion["pays"]["code"],
        "prix": course.get("libelle"),
        "discipline": course.get("discipline"),
        "specialite": course.get("specialite"),
        "distance": course.get("distance"),
        "corde": course.get("corde"),
        "typePiste": course.get("typePiste"),
        "terrain": pen.get("intitule"),
        "terrain_valeur": going_value(pen),
        "categorieParticularite": course.get("categorieParticularite"),
        "conditionAge": course.get("conditionAge"),
        "conditionSexe": course.get("conditionSexe"),
        "montantPrix": course.get("montantPrix"),
        "nombreDeclaresPartants": course.get("nombreDeclaresPartants"),
        "meteo": meteo.get("nebulositeLibelleCourt"),
        "temperature": meteo.get("temperature"),
        "forceVent": meteo.get("forceVent"),
        "directionVent": meteo.get("directionVent"),
        "statut_course": course.get("statut"),
    }


def runner_row(p, partants):
    gains = p.get("gainsParticipant") or {}
    ecart = p.get("distanceChevalPrecedent") or {}
    place = p.get("ordreArrivee")
    return {
        "numPmu": p["numPmu"],
        "nom": p["nom"],
        "idCheval": p.get("idCheval"),  # missing in some older files
        "statut": p["statut"],
        "partants": partants,
        "age": p.get("age"),
        "sexe": p.get("sexe"),
        "race_cheval": p.get("race"),
        "musique": p.get("musique"),
        "driver": p.get("driver"),
        "driverChange": p.get("driverChange"),
        "entraineur": p.get("entraineur"),
        "proprietaire": p.get("proprietaire"),
        "eleveur": p.get("eleveur"),
        "nomPere": p.get("nomPere"),
        "nomMere": p.get("nomMere"),
        "handicapPoids": p.get("handicapPoids"),
        "poidsConditionMonte": p.get("poidsConditionMonte"),
        "handicapValeur": p.get("handicapValeur"),
        "handicapDistance": p.get("handicapDistance"),
        "placeCorde": p.get("placeCorde"),
        "deferre": p.get("deferre"),
        "oeilleres": p.get("oeilleres"),
        "avisEntraineur": p.get("avisEntraineur"),
        "indicateurInedit": p.get("indicateurInedit"),
        "nombreCourses": p.get("nombreCourses"),
        "nombreVictoires": p.get("nombreVictoires"),
        "nombrePlaces": p.get("nombrePlaces"),
        "nombrePlacesSecond": p.get("nombrePlacesSecond"),
        "nombrePlacesTroisieme": p.get("nombrePlacesTroisieme"),
        "gainsCarriere": gains.get("gainsCarriere"),
        "gainsVictoires": gains.get("gainsVictoires"),
        "gainsPlace": gains.get("gainsPlace"),
        "gainsAnneeEnCours": gains.get("gainsAnneeEnCours"),
        "gainsAnneePrecedente": gains.get("gainsAnneePrecedente"),
        "cote_reference": (p.get("dernierRapportReference") or {}).get("rapport"),
        "cote_finale": (p.get("dernierRapportDirect") or {}).get("rapport"),
        "res_place": place,
        "res_top5": int(place is not None and place <= 5) if p["statut"] == "PARTANT" else None,
        "res_incident": p.get("incident"),
        "res_ecart": ecart.get("libelleCourt"),
        "res_reductionKilometrique": p.get("reductionKilometrique"),
        "res_tempsObtenu": p.get("tempsObtenu"),
    }


def past_rows(race_id, iso, perf):
    race_day = date.fromisoformat(iso)
    for horse in perf["participants"]:
        for c in horse.get("coursesCourues") or []:
            me = next((r for r in c["participants"] if r.get("itsHim")), None)
            if me is None:  # 3 past races in the feed lack the horse's own line
                continue
            place = me.get("place") or {}
            past_day = datetime.fromtimestamp((c["date"] + (c.get("timezoneOffset") or 0)) / 1000, tz=timezone.utc).date()
            yield {
                "race_id": race_id,
                "numPmu": horse["numPmu"],
                "nom": horse["nomCheval"],
                "past_date": past_day.isoformat(),
                "jours_avant": (race_day - past_day).days,
                "hippodrome": c.get("hippodrome"),
                "prix": c.get("nomPrix"),
                "discipline": c.get("discipline"),
                "distance": c.get("distance"),
                "allocation": c.get("allocation"),
                "nbParticipants": c.get("nbParticipants"),
                "tempsDuPremier": c.get("tempsDuPremier"),
                "terrain": c.get("etatTerrain"),
                "place": place.get("place"),
                "statusArrivee": place.get("statusArrivee"),
                "jockey": me.get("nomJockey"),
                "poidsJockey": me.get("poidsJockey"),
                "corde": me.get("corde"),
                "ecart": (me.get("distanceAvecPrecedent") or {}).get("rawValue"),
                "reductionKilometrique": me.get("reductionKilometrique"),
                "oeillere": me.get("oeillere"),
            }


def bet_kind(type_pari, libelle):
    """Normalise the varying libelle wording ("Quinté+ Ordre" or just "Ordre"...) to one code."""
    if "NP" in libelle.split():
        return f"{type_pari}:NP"
    for word, code in (("Tirelire", "ORDRE_TIRELIRE"), ("Désordre", "DESORDRE"), ("Ordre", "ORDRE"),
                       ("4sur5", "BONUS_4SUR5"), ("Bonus 4", "BONUS_4"), ("Bonus 3", "BONUS_3"), ("Bonus", "BONUS")):
        if word in libelle:
            return f"{type_pari}:{code}"
    return type_pari


def payout_rows(race_id, rapports):
    for bet in rapports:
        for r in bet["rapports"]:
            yield {
                "race_id": race_id,
                "typePari": bet["typePari"],
                "pari": bet_kind(bet["typePari"], r["libelle"]),
                "libelle": r["libelle"],
                "combinaison": r["combinaison"],
                "euros_pour_1_euro": r["dividendePourUnEuro"] / 100,
                "mise_de_base_euros": bet["miseBase"] / 100,
                "nombreGagnants": r.get("nombreGagnants"),
                "rembourse": bet.get("rembourse"),
            }


def all_runs():
    """Runners of every French race with a saved runners file (Quinté races included)."""
    for f in sorted((RAW / "programme").glob("*.json")):
        prog = json.loads(f.read_bytes())["programme"]
        for reunion in prog["reunions"]:
            if reunion["pays"]["code"] != "FRA":
                continue
            for course in reunion["courses"]:
                path = RAW / "participants" / f.stem / f"R{reunion['numOfficiel']}C{course['numOrdre']}.json"
                if not path.exists():
                    continue
                ps = json.loads(path.read_bytes()).get("participants") or []
                partants = sum(p.get("statut") == "PARTANT" for p in ps)
                for p in ps:
                    if p.get("statut") != "PARTANT":
                        continue
                    yield {"date": f.stem, "race_id": f"{f.stem}_R{reunion['numOfficiel']}C{course['numOrdre']}",
                           "specialite": course.get("specialite"), "partants": partants, "numPmu": p["numPmu"],
                           "driver": p.get("driver"), "entraineur": p.get("entraineur"), "place": p.get("ordreArrivee")}


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{path.relative_to(ROOT.parent)}: {len(rows)} rows")


def main():
    runners, past, payouts, missing = [], [], [], []
    for iso, reunion, course in quinte_races():
        race = race_row(iso, reunion, course)
        name = f"R{race['reunion']}C{race['course']}.json"
        files = {k: RAW / k / iso / name for k in ("participants", "performances", "rapports")}
        missing += [str(p) for p in files.values() if not p.exists()]
        if files["participants"].exists():
            ps = json.loads(files["participants"].read_bytes())["participants"]
            partants = sum(p["statut"] == "PARTANT" for p in ps)
            runners += [{**race, **runner_row(p, partants)} for p in ps]
        if files["performances"].exists():
            past += past_rows(race["race_id"], iso, json.loads(files["performances"].read_bytes()))
        if files["rapports"].exists():
            payouts += payout_rows(race["race_id"], json.loads(files["rapports"].read_bytes()))
    write_csv(ROOT / "runners.csv", runners)
    write_csv(ROOT / "past_runs.csv", past)
    write_csv(ROOT / "payouts.csv", payouts)
    write_csv(ROOT / "all_runs.csv", all_runs())
    if missing:
        print(f"{len(missing)} raw files missing, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
