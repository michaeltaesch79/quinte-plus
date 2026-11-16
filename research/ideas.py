"""
Candidate features for the morning prono (benchmark.py candidates), declared in PREREGISTRATION.md
("Morning model, 2026-11-16") before any run on the test periods.

Each idea adds columns known before the race, with or without odds:
  memory   the horse's previous Quinté+ starts in the last 365 days: how the bettors rated it then
           (reference odds of those races, relative to an average runner), how often it ran and placed;
  field    recent form measured against the field: finishing position as a share of the runners, and
           the level of the races in which it finished well;
  musique  the form line of the runner file, up to 10 runs (the past-performance file stops at 5 in trot);
  speed    trot times per km of the last 5 trot runs (the two features of speed_eval.py, unchanged).
Every idea, and the four together ("morning pack"), is scored on the production model without odds (the
morning prono) and with odds.

    <venv>/python research/benchmark.py --with research/ideas.py --tag morning

Adopted on 2026-11-16: the columns now come from features.py (MORNING, model without odds only), checked
equal to this file on every runner. To rerun the test itself, check out the pre-registration commit
(0f71e08): on later code these columns are already in the production model.
"""
import math
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

import benchmark as bm
import features as ft
import speed_eval

IDEAS = {
    "memory": ["f_prev_mkt", "f_prev_runs", "f_prev_top5"],
    "field": ["f_form_pct", "f_form_pct_best", "f_class_perf"],
    "musique": ["f_mus_place", "f_mus_top3", "f_mus_fail", "f_mus_win"],
    "speed": speed_eval.SPEED,
}
NEW = [c for cols in IDEAS.values() for c in cols]


# ---------------------------------------------------------------- features (plain Python, CSV rows in)

def market_memory(runners):
    """(race_id, numPmu) -> the three memory features. A previous start counts when it is dated before
    the race day; its odds are the reference odds of that earlier race."""
    by_race = defaultdict(list)
    for r in runners:
        if r["statut"] == "PARTANT":
            by_race[r["race_id"]].append(r)
    starts = defaultdict(list)  # horse name -> [(day, log(market win chance x runners), top 5 or None)]
    for field in by_race.values():
        odds = [ft.num(r["cote_reference"]) for r in field]
        inv = sum(1 / o for o in odds if o)
        for r, o in zip(field, odds):
            rel = math.log((1 / o) / inv * len(field)) if o and all(odds) else None
            top5 = (r["res_top5"] == "1") if r["res_top5"] != "" else None
            starts[r["nom"]].append((date.fromisoformat(r["date"]), rel, top5))
    out = {}
    for field in by_race.values():
        for r in field:
            day = date.fromisoformat(r["date"])
            prev = sorted((s for s in starts[r["nom"]] if day - timedelta(days=365) <= s[0] < day),
                          key=lambda s: s[0], reverse=True)
            rels = [rel for _, rel, _ in prev if rel is not None][:3]
            done = [t for _, _, t in prev if t is not None]
            out[(r["race_id"], r["numPmu"])] = {
                "f_prev_mkt": sum(rels) / len(rels) if rels else None,
                "f_prev_runs": math.log1p(len(prev)),
                "f_prev_top5": ft.rate(sum(done), len(done), 0.3)}
    return out


def field_form(runners, past):
    """(race_id, numPmu) -> the three field features, from the last 5 past runs. Position as a share of the
    field: 0 = won, 1 = last; unplaced without a position = 1. Runs not finished (disqualified, fell,
    pulled up) are left out: f_form_failed and f_dq_rate already count them."""
    prize = {r["race_id"]: ft.num(r["montantPrix"]) for r in runners}
    runs = defaultdict(list)
    for p in past:
        runs[(p["race_id"], p["numPmu"])].append(p)
    out = {}
    for key, rows in runs.items():
        pcts, weighted, weight = [], 0.0, 0.0
        for p in sorted(rows, key=lambda p: p["past_date"], reverse=True)[:5]:
            n, place = ft.num(p["nbParticipants"]), ft.num(p["place"])
            if p["statusArrivee"] == "PLACE" and place and n and n >= 2:
                pct = min(max((place - 1) / (n - 1), 0.0), 1.0)
            elif p["statusArrivee"] == "NON_PLACE":
                pct = 1.0
            else:
                continue
            pcts.append(pct)
            alloc = ft.num(p["allocation"])
            if alloc and prize.get(key[0]):
                weighted += (1 - pct) * math.log(alloc / prize[key[0]])
                weight += 1 - pct
        out[key] = {"f_form_pct": sum(pcts) / len(pcts) if pcts else None,
                    "f_form_pct_best": min(pcts) if pcts else None,
                    "f_class_perf": weighted / weight if weight else None}
    return out


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


def musique_form(runners):
    """(race_id, numPmu) -> the four musique features over the last 10 runs of the form line."""
    out = {}
    for r in runners:
        t = musique_tokens(r["musique"])[:10]
        places = [x if isinstance(x, int) and x else 10 for x in t]  # unplaced or not finished = 10
        fails = sum(not isinstance(x, int) for x in t)
        out[(r["race_id"], r["numPmu"])] = {
            "f_mus_place": sum(places) / len(places) if t else None,
            "f_mus_top3": ft.rate(sum(1 <= p <= 3 for p in places), len(t), 0.3) if t else None,
            "f_mus_fail": ft.rate(fails, len(t), 0.1) if t else None,
            "f_mus_win": ft.rate(sum(p == 1 for p in places), len(t), 0.1) if t else None}
    return out


# ---------------------------------------------------------------- benchmark candidates

def prepare(races, feats, data):
    """Append every new column to the races once; each candidate then picks its own columns by name."""
    tables = [market_memory(data["runners"]), field_form(data["runners"], data["past"]), musique_form(data["runners"])]
    names = IDEAS["memory"] + IDEAS["field"] + IDEAS["musique"]
    for r in races:
        rows = []
        for n in r["nums"]:
            f = {}
            for t in tables:
                f.update(t.get((r["id"], str(n)), {}))
            rows.append([np.nan if f.get(c) is None else f[c] for c in names])
        r["raw"] = np.hstack([r["raw"], np.array(rows, float)])
    speed_eval.add_speed(races, [], data["past"])
    return names + IDEAS["speed"]


def cfg(idea_cols, odds):
    base = bm.baseline_models  # the production columns and penalties, whatever they are when this runs

    def make(feats):
        prod = base([c for c in feats if c not in NEW])[4 if odds else 3]["cfg"]
        return {**prod, "cols": prod["cols"] + idea_cols}
    return make


def model(label, cols, odds):
    return {"name": f"{label}, {'with' if odds else 'no'} odds", "versus": bm.WITH_ODDS if odds else bm.NO_ODDS,
            "cfg": cfg(cols, odds)}


MODELS = [model(label, cols, odds) for odds in (False, True) for label, cols in [*IDEAS.items(), ("morning pack", NEW)]]
MODELS[0]["prepare"] = prepare
STRATEGIES = [{"name": "top 5 " + m["name"], "versus": "top 5 " + m["versus"], "pick": bm.top5(m["name"])} for m in MODELS]
