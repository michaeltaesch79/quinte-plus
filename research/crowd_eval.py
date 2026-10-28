"""
Avoid the crowd? The Quinté+ stakes (fetch_combinaisons.py, online pool only, FINAL stakes) list the 12
most-played 5-horse sets of each race. A winning set the crowd played pays a small Désordre.

1. Désordre dividend when the winning set was / was not among the 12 most played (all races).
2. One 2 € ticket per race, production model with odds (current features.py, L2 10, stage discounts):
   - top 5: the model's five strongest horses (the app's prono);
   - avoid the crowd: the same ticket, unless its set is among the 12 most played; then the most likely
     set (exact Plackett-Luce set probability) of 5 among the model's top 7 that is not.
Rule, declared before running: adopt only if the race-by-race difference in returns (avoid - top 5) is
positive with a 95% week-block interval above 0, in BOTH periods (hold-out 2024-11..2025-10, 2026-02..10).
Caveat: final stakes are optimistic; at betting time only part of the pool is known.

    <venv>/python research/crowd_eval.py
Writes research/results/crowd_eval.json.
"""
import glob
import itertools
import json
import statistics
from collections import defaultdict

import numpy as np

import evaluate as ev
import features as ft
import plmodel as plm
from paths import DATA

PERIODS = {"hold-out 2024-25": ("2024-11", "2025-10"), "2026": ("2026-02", "2026-10")}


def load_crowd():
    """race_id -> {frozenset of 5 numbers: stake in euros} for the 12 most-played Quinté+ sets."""
    out = {}
    for f in glob.glob(str(DATA / "raw" / "combinaisons" / "*" / "R*.json")):
        iso, rc = f.replace("\\", "/").split("/")[-2:]
        d = json.load(open(f, encoding="utf-8"))
        for c in d.get("combinaisons", []):
            if c["pariType"] == "E_QUINTE_PLUS":
                out[f"{iso}_{rc[:-5]}"] = {frozenset(x["combinaison"]): x["totalEnjeu"] / 100 for x in c["listeCombinaisons"]}
    return out


def desordre_by_crowd(races, crowd):
    pay = defaultdict(list)
    for p in ev.load_csv(DATA / "payouts.csv"):
        if p["pari"] == "QUINTE_PLUS:DESORDRE":
            pay[p["race_id"]].append(float(p["euros_pour_1_euro"]))
    split = {"in the 12 most played": [], "not among them": []}
    for r in races:
        if r["id"] in crowd and pay.get(r["id"]):
            win = frozenset(r["nums"][i] for i in r["order"])
            split["in the 12 most played" if win in crowd[r["id"]] else "not among them"].append(max(pay[r["id"]]))
    return {k: {"races": len(v), "median_eur_per_1": round(statistics.median(v), 1)} for k, v in split.items() if v}


def main():
    runners, past = ev.load_csv(DATA / "runners.csv"), ev.load_csv(DATA / "past_runs.csv")
    odds_ref = {(r["race_id"], r["numPmu"]): float(r["cote_reference"]) for r in runners if r["cote_reference"]}
    races, feats = ev.build_races(ft.compute(runners, past), odds_ref)
    crowd = load_crowd()
    rep = {"desordre_dividend": desordre_by_crowd(races, crowd)}
    print(rep["desordre_dividend"])
    months = {p: ev.months_between(races, *span) for p, span in PERIODS.items()}
    preds = ev.walk_forward(races, feats, {"cols": feats, "l2": 10.0, "discount": True},
                            sorted({m for ms in months.values() for m in ms}))
    payouts = ev.payout_table()
    for period, ms in months.items():
        rows = {"top 5": [], "avoid the crowd": []}
        diff, in_crowd = [], 0
        for r in races:
            if r["month"] not in ms or r["id"] not in crowd or np.isnan(r["odds"]).any():
                continue
            s, lam = preds[r["id"]]
            top7 = list(np.argsort(-s)[:7])
            base = top7[:5]
            ticket = base
            if frozenset(r["nums"][i] for i in base) in crowd[r["id"]]:
                in_crowd += 1
                free = [c for c in itertools.combinations(top7, 5) if frozenset(r["nums"][i] for i in c) not in crowd[r["id"]]]
                ticket = list(max(free, key=lambda c: plm.set_prob(s, lam, c)))
            got = {}
            for name, idx in (("top 5", base), ("avoid the crowd", ticket)):
                rank, ret = ev.settle(r, idx, payouts)
                got[name] = ret
                rows[name].append({"date": r["date"], "family": r["family"], "month": r["month"], "bet": True,
                                   "ret": ret, "rank": rank})
            diff.append((r["date"], got["avoid the crowd"] - got["top 5"]))
        rep[period] = {"races": len(diff), "top5_ticket_among_12_most_played": in_crowd,
                       **{name: ev.bet_summary(x) for name, x in rows.items()},
                       "difference_eur_per_race": round(float(np.mean([d for _, d in diff])), 3),
                       "difference_ci95": ev.block_bootstrap(diff, np.mean)}
        print(period, json.dumps({k: rep[period][k] for k in ("races", "top5_ticket_among_12_most_played",
                                                               "difference_eur_per_race", "difference_ci95")}))
        for name in rows:
            b = rep[period][name]
            print(f"   {name:<16} back {b['returned']:.0f} on {b['staked']:.0f}, ROI {b['roi']:+.2f}, {b['hits']}")
    rep["adopt"] = all(rep[p]["difference_ci95"][0] > 0 for p in PERIODS)
    print("-> adopt:", rep["adopt"])
    (ev.OUT / "crowd_eval.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
