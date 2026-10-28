"""
Do trot speed figures improve the production model? f_trot_gap (seconds per km behind the winner, last 5
trot runs) and f_trot_best_rel (best time per km of the last 5 trot runs vs the field's median best),
trot races only. Production settings with and without them (current features.py, stage discounts,
L2 20 without odds, 10 with odds), walk-forward.
Rule, declared before running: adopt only if the mean log-likelihood gain is positive in ALL three
periods (validation 2026-02..05, test 2026-06..10, hold-out year 2024-11..2025-10) for BOTH models.
Gains are also shown on trot races alone (the only races where the features exist).

    <venv>/python research/speed_eval.py
Writes research/results/speed_eval.json.
"""
import json
from collections import defaultdict

import numpy as np

import evaluate as ev
import features as ft
from paths import DATA

SPEED = ["f_trot_gap", "f_trot_best_rel"]
PERIODS = {"validation": ev.VALID, "test": ev.TEST, "hold-out": ("2024-11", "2025-10")}


def production_cols(feats, odds):
    return [c for c in feats if odds or c not in ft.ODDS]


def add_speed(races, feats, past):
    """Append the two speed columns (trot races only). Times are in centiseconds: the horse's time per
    km (reductionKilometrique) and the winner's total time (tempsDuPremier) over the race distance."""
    runs = defaultdict(list)
    for p in past:
        if ft.family(p["discipline"]) not in ("ATTELE", "MONTE"):
            continue
        rk, t1, d = ft.num(p["reductionKilometrique"]), ft.num(p["tempsDuPremier"]), ft.num(p["distance"])
        rk = rk / 100 if rk else None
        gap = min(max(rk - t1 * 10 / d, 0.0), 4.0) if rk and t1 and d else None
        runs[(p["race_id"], p["numPmu"])].append((p["past_date"], rk, gap))
    for r in races:
        cols = np.full((len(r["nums"]), 2), np.nan)
        if r["family"] == "TROT":
            last5 = [sorted(runs[(r["id"], str(n))], key=lambda x: x[0], reverse=True)[:5] for n in r["nums"]]
            best = [min((rk for _, rk, _ in x if rk), default=None) for x in last5]
            known = sorted(b for b in best if b)
            median_best = known[len(known) // 2] if known else None
            for i, x in enumerate(last5):
                gaps = [g for *_, g in x if g is not None]
                if gaps:
                    cols[i, 0] = sum(gaps) / len(gaps)  # seconds per km behind the winner
                if best[i]:
                    cols[i, 1] = min(max(best[i] - median_best, -5.0), 5.0)  # a few records are tens of seconds off
        r["raw"] = np.hstack([r["raw"], cols])
    return feats + SPEED


def main():
    runners, past = ev.load_csv(DATA / "runners.csv"), ev.load_csv(DATA / "past_runs.csv")
    odds_ref = {(r["race_id"], r["numPmu"]): float(r["cote_reference"]) for r in runners if r["cote_reference"]}
    races, feats = ev.build_races(ft.compute(runners, past), odds_ref)
    feats = add_speed(races, feats, past)
    months = {p: ev.months_between(races, *span) for p, span in PERIODS.items()}
    every = sorted({m for ms in months.values() for m in ms})
    rep, adopt = {}, True
    for label, odds, l2 in (("no odds", False, 20.0), ("with odds", True, 10.0)):
        base = [c for c in production_cols(feats, odds) if c not in SPEED]
        a = ev.walk_forward(races, feats, {"cols": base, "l2": l2, "discount": True}, every)
        b = ev.walk_forward(races, feats, {"cols": base + SPEED, "l2": l2, "discount": True}, every)
        rep[label] = {}
        for period, ms in months.items():
            d = {r["id"]: (r["date"], ev.pl_loglik(*b[r["id"]], r["order"]) - ev.pl_loglik(*a[r["id"]], r["order"]))
                 for r in races if r["month"] in ms}
            trot = [d[r["id"]] for r in races if r["id"] in d and r["family"] == "TROT"]
            gain = float(np.mean([x for _, x in d.values()]))
            rep[label][period] = {"mean_ll_gain": round(gain, 4), "ci95": ev.block_bootstrap(list(d.values()), np.mean),
                                  "races": len(d), "trot_only_gain": round(float(np.mean([x for _, x in trot])), 4),
                                  "trot_ci95": ev.block_bootstrap(trot, np.mean), "trot_races": len(trot)}
            adopt &= gain > 0
        print(label, json.dumps(rep[label]))
    rep["adopt_in_production"] = adopt
    print("-> adopt:", adopt)
    (ev.OUT / "speed_eval.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
