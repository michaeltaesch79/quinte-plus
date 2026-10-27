"""
Convert the pipeline's daily record (data/daily/YYYY-MM-DD.json, written by daily.py) into the
app's data file app/data/today.json, schema_version 1 (format in app/README.md).

Also uses data/backtest_summary.json (test results), data/payouts.csv + data/runners.csv
(usual payout ranges) and the earlier daily records (yesterday's result and the season score).

Usage (Python 3.9+, no extra packages):
    python app_export.py 2026-10-28     # normally called by daily.py
"""
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone

from paths import APP_JSON, DATA

DAILY = DATA / "daily"
OUT = APP_JSON
AI_NAME = "L'IA qui parle aux chevaux"
STAKE = 2

DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
          "octobre", "novembre", "décembre"]
SMALL_WORDS = {"de", "du", "des", "la", "le", "les", "et", "en", "sur", "à", "au", "aux"}
SEX = {"MALES": "Mâle", "FEMELLES": "Femelle", "HONGRES": "Hongre"}
FACTOR_TITLE = {
    "f_class": "Niveau", "f_jockey_rides": "{rider}", "f_jockey_top3": "{rider}", "f_draw_rel": "Corde",
    "f_avis": "Entraîneur", "f_trainer_top5": "Entraîneur", "f_weight_rel": "Poids", "f_career_win": "Palmarès",
    "f_log_gains_year": "Gains", "f_last_place": "Forme", "f_form_place": "Forme", "f_form_top3": "Forme",
    "f_form_win": "Forme", "f_going_top3": "Terrain", "f_dist_top3": "Distance", "f_track_top3": "Hippodrome",
    "f_days_since": "Fraîcheur", "f_unshod4": "Ferrure", "f_dist_handicap": "Recul",
}
BET_LABEL = [("QUINTE_PLUS:ORDRE", "Quinté+ dans l'ordre"), ("QUINTE_PLUS:DESORDRE", "Quinté+ dans le désordre"),
             ("QUINTE_PLUS:BONUS_4SUR5", "Bonus 4 sur 5"), ("QUINTE_PLUS:BONUS_3", "Bonus 3")]
STRATEGY = {"avec": "IA avec cotes", "sans": "IA sans cotes", "fav": "Favoris (cotes)", "value": "Bon rapport"}


def date_label(iso):
    d = date.fromisoformat(iso)
    return f"{DAYS[d.weekday()]} {d.day}{'er' if d.day == 1 else ''} {MONTHS[d.month - 1]}"


def nice(text):
    """'PRIX DE LA ROUTE DES PRINCES' -> 'Prix de la Route des Princes'; mixed case ('ParisLongchamp') is kept."""
    if not text or text != text.upper():
        return text or ""
    words = []
    for i, w in enumerate(text.lower().split(" ")):
        parts = re.split(r"([-'/])", w)
        w = "".join(p if p in "-'/" or (i and j == 0 and p in SMALL_WORDS) else p.capitalize()
                    for j, p in enumerate(parts))
        words.append(re.sub(r"^(D|L)'", lambda m: m.group(1).lower() + "'", w) if i else w)
    return " ".join(words).replace("/Mer", "-sur-Mer")


def person(name):
    """'S.PASQUIER' or 'B. ROCHARD' -> 'S. Pasquier', 'J.PH. DUBOIS' -> 'J.Ph. Dubois'."""
    m = re.match(r"^((?:[A-Z]{1,2}\.)+)\s*(.+)$", name or "")
    if not m:
        return nice(name)
    initials = ".".join(p.capitalize() for p in m.group(1).split(".") if p) + "."
    return f"{initials} {nice(m.group(2))}"


def musique(text):
    """'3p1p(25)2p' -> '3p 1p (25) 2p' (the app splits on spaces)."""
    return " ".join(re.findall(r"\(\d+\)|\d+[a-zA-Z]|[A-Z][a-z]", text or ""))


def surface(c):
    spec, disc = c["specialite"], c["discipline"]
    if spec.startswith("TROT"):
        return "trot monté" if spec == "TROT_MONTE" else "trot attelé"
    if spec == "OBSTACLE":
        return {"HAIE": "haies", "STEEPLECHASE": "steeple-chase", "CROSS": "cross"}.get(disc, "obstacle")
    return "sable" if (c.get("terrain") or "").upper().startswith("PSF") else "herbe"


def race_type(cat):
    cat = cat or ""
    if "HANDICAP" in cat:
        return "Handicap"
    if cat.startswith("GROUPE_"):
        return "Groupe " + cat.split("_", 1)[1]
    labels = {"EUROPEENNE": "Course européenne", "INTERNATIONALE": "Course internationale",
              "NATIONALE": "Course nationale", "AUTOSTART": "Autostart", "COURSE_A_CONDITIONS": "Course à conditions"}
    base = labels.get(cat.replace("_AUTOSTART", ""), "Course")
    return base + (" (autostart)" if cat.endswith("_AUTOSTART") else "")


def euros(v):
    v = round(v) if v >= 10 else round(v, 1)
    return f"{v:,}".replace(",", " ").replace(".", ",")


def payout_ranges(specialite):
    """Middle half (25th-75th percentile) of each Quinté+ payout over the year, for this type of race."""
    with (DATA / "runners.csv").open(encoding="utf-8") as f:
        kind = {r["race_id"]: r["specialite"] for r in csv.DictReader(f)}
    vals = defaultdict(list)
    with (DATA / "payouts.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if kind.get(r["race_id"]) == specialite:
                vals[r["pari"]].append(STAKE * float(r["euros_pour_1_euro"]))
    rows = []
    for pari, label in BET_LABEL:
        v = sorted(set(vals[pari]))
        if len(v) >= 8:
            rows.append({"label": label, "range": f"{euros(v[len(v) // 4])} à {euros(v[3 * len(v) // 4])} €"})
    races = len({rid for rid, s in kind.items() if s == specialite})
    return {"title": "Si ça passe (estimation)", "rows": rows,
            "note": f"Fourchette habituelle pour un ticket de {STAKE} € : la moitié des rapports des {races} Quintés "
                    f"de ce type sur les 12 derniers mois. Le vrai rapport dépend du nombre de gagnants."}


def factors(h, rider, has_odds):
    out = [{"sign": "plus", "title": FACTOR_TITLE.get(a["facteur"], "Atout").format(rider=rider), "text": a["texte"] + "."}
           for a in h["atouts"]]
    out += [{"sign": "minus", "title": FACTOR_TITLE.get(a["facteur"], "Point faible").format(rider=rider),
             "text": a["texte"] + "."} for a in h["points_faibles"]]
    if has_odds and h["cote"]:
        out.append({"sign": "eq", "title": "Cote", "text": f"Cote {euros(h['cote'])} contre 1 : les parieurs lui donnent "
                    f"environ {round(100 * h['chance_top5_marche_estimee'])} % de chances d'être dans les 5 (estimation)."})
    else:
        out.append({"sign": "q", "title": "Cote", "text": "Pas encore de cote publiée au moment du calcul."})
    if not h["atouts"] and not h["points_faibles"]:
        out.insert(0, {"sign": "eq", "title": "Profil", "text": "Rien ne le distingue nettement des autres partants."})
    return out


def verdict(h):
    lead = ("Le cheval que l'IA voit le plus souvent dans les 5." if h["rang_ia"] == 1 else
            f"Dans les 5 de l'IA ({h['rang_ia']}e)." if h["rang_ia"] <= 5 else
            f"Pas dans la sélection de l'IA ({h['rang_ia']}e).")
    if h["rang_ia"] <= 5 and h["atouts"]:
        return f"{lead} Atout : {h['atouts'][0]['texte'][0].lower()}{h['atouts'][0]['texte'][1:]}."
    if h["points_faibles"]:
        return f"{lead} Frein : {h['points_faibles'][0]['texte'][0].lower()}{h['points_faibles'][0]['texte'][1:]}."
    return lead


def sex_age(h):
    return " ".join(x for x in (SEX.get(h["sexe"], ""), f"{h['age']} ans" if h["age"] else "") if x)


def runner(h, rider, has_odds, non_runner=False):
    return {
        "num": h["num"], "name": nice(h["nom"]), "jockey": person(h["jockey"]),
        "weight_kg": h["poids_kg"], "sex_age": sex_age(h), "draw": h["corde"], "musique": musique(h["musique"]),
        "non_runner": non_runner,
        "ai_rank": None if non_runner else h["rang_ia"],
        "win_chance_pct": None if non_runner else max(1, round(100 * h["chance_gagner_estimee"])),
        "verdict": "Non-partant." if non_runner else verdict(h),
        "factors": [] if non_runner else factors(h, rider, has_odds),
    }


def summary():
    f = DATA / "backtest_summary.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def test_line(s, key):
    x = s["strategies"][STRATEGY[key]]
    return f"{round(x['rendu_euros'])} € rendus pour {round(x['mise_euros'])} € misés"


def ticket(nums, title, note=None):
    lines = [{"bet": "Quinté+ unitaire", "nums": nums, "price_eur": STAKE}] if len(nums) == 5 else []
    t = {"title": title, "lines": lines, "total_eur": STAKE if lines else 0}
    if note:
        t["note"] = note
    return t


def modes(rec, s, payouts):
    with_odds = rec["modele"]["avec_cotes"]
    test = (f"Sur {s['courses_testees']} Quintés passés (test), ce ticket : {test_line(s, 'avec' if with_odds else 'sans')}. "
            f"Les 5 favoris : {test_line(s, 'fav')}." if s else None)
    prono = {
        "title": "Mes 5 chevaux pour l'arrivée",
        "intro": rec["modele"]["note"],
        "picks": rec["prono"],
        "ticket": ticket(rec["prono"], "Mon ticket conseillé", "Dans l'ordre de préférence de l'IA."),
        "payouts": payouts,
        "notices": [x for x in (test,) if x],
    }
    if rec["bon_rapport"]:
        value = {
            "title": "Le bon rapport",
            "intro": "Des chevaux que l'IA juge meilleurs que ne le disent leurs cotes : plus risqué, mieux payé si ça passe.",
            "picks": rec["bon_rapport"],
            "ticket": ticket(rec["bon_rapport"], "Ticket bon rapport"),
            "payouts": payouts,
            "notices": [f"Attention : sur nos tests, cette méthode a fait moins bien que les favoris "
                        f"({test_line(s, 'value')})."] if s else [],
        }
    else:
        value = {
            "title": "Le bon rapport",
            "intro": "Il compare l'avis de l'IA aux cotes des parieurs.",
            "picks": [],
            "ticket": ticket([], "Ticket bon rapport", "Pas de ticket tant que les cotes ne sont pas connues."),
            "notices": ["Les cotes ne sont pas encore publiées : le bon rapport sera calculé au prochain passage."],
        }
    return {"prono": prono, "value": value}


def settled(before):
    """Earlier daily records that have a result, oldest first."""
    out = []
    for f in sorted(DAILY.glob("????-??-??.json")):
        if f.stem < before:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("resultat"):
                out.append(rec)
    return out


def yesterday(history):
    done = [r for r in history if not r["resultat"].get("annulee")]
    if not done:
        return {"label": "Pas encore de résultat", "arrival": [], "sources": []}
    r = done[-1]
    return {"label": f"{date_label(r['date'])} · {nice(r['course']['hippodrome'])}",
            "arrival": r["resultat"]["arrivee"], "sources": [{"source": AI_NAME, "picks": r["prono"]}]}


def season(history):
    played = [r for r in history if not r["resultat"].get("annulee")]
    bet = sum(r["resultat"]["mise_euros"] for r in played)
    won = round(sum(r["resultat"]["gain_euros"] for r in played), 2)
    since = f"depuis le {date_label(played[0]['date']).split(' ', 1)[1]}" if played else "pas encore commencé"
    return {"players": [{"name": AI_NAME, "bet_eur": bet, "won_eur": won}],
            "note": f"Un ticket Quinté+ à {STAKE} € par jour sur le prono de l'IA, {since}. "
                    f"Les gains comptent les vrais rapports du PMU."}


def build(iso):
    rec = json.loads((DAILY / f"{iso}.json").read_text(encoding="utf-8"))
    if not rec.get("course"):
        raise SystemExit(f"{iso}: no Quinté in the record, app file not written")
    c = rec["course"]
    rider = "Driver" if c["specialite"].startswith("TROT") else "Jockey"
    has_odds = rec["modele"]["avec_cotes"]
    history = settled(iso)
    runners = [runner(h, rider, has_odds) for h in rec["chevaux"]]
    runners += [runner(h, rider, has_odds, non_runner=True) for h in rec.get("non_partants", [])]
    runners.sort(key=lambda r: r["num"])
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": f"Plackett-Luce {'avec' if has_odds else 'sans'} cotes, "
                        f"{rec['modele']['courses_d_entrainement']} Quintés (daily.py)",
        "race": {
            "date": iso, "date_label": date_label(iso),
            "reunion": f"R{c['reunion']}", "course": f"C{c['course']}",
            "hippodrome": nice(c["hippodrome"]), "prix": nice(c["prix"]),
            "start_time": c["heure_depart"][11:].replace(":", "h"), "distance_m": c["distance"],
            "surface": surface(c), "race_type": race_type(c["categorieParticularite"]),
            "runners_count": c["partants"],
        },
        "runners": runners,
        "modes": modes(rec, summary(), payout_ranges(c["specialite"])),
        "other_tips": {"sources": [], "note": "Les pronos de la presse ne sont pas encore récupérés automatiquement."},
        "yesterday": yesterday(history),
        "season": season(history),
    }


def write(iso):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(iso), ensure_ascii=False, indent=1), encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(write(sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()))
