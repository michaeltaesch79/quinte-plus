"""
Do jockey/trainer form features from ALL French races (data/all_runs.csv) improve the model?

Production settings with and without the 5 form features, same walk-forward protocol as
evaluate.py (validation 2026-02..05, test 2026-06..10). The form features exist only where 30 days of
all-race history precede the race (all_runs covers 2025-10-27 onwards); before that they are missing,
which the model treats as "no information".
Pre-declared rule: adopt in production only if the mean log-likelihood gain is positive on BOTH the
validation and the test months, for the with-odds model AND the no-odds model.

    <venv>/python research/form_eval.py
Writes research/results/form_eval.json.
"""
import json

import numpy as np

import evaluate as ev
import features as ft
from paths import DATA

FORM = ["f_jockey_form90", "f_jockey_runs90", "f_trainer_form90", "f_trainer_runs90", "f_trainer_form30"]


def main():
    runners, past = ev.load_csv(DATA / "runners.csv"), ev.load_csv(DATA / "past_runs.csv")
    all_runs = ev.load_csv(DATA / "all_runs.csv")
    odds_ref = {(r["race_id"], r["numPmu"]): float(r["cote_reference"]) for r in runners if r["cote_reference"]}
    races, feats = ev.build_races(ft.compute(runners, past, all_runs), odds_ref)
    mv = ev.months_between(races, *ev.VALID)
    mt = ev.months_between(races, *ev.TEST)
    base_no = [c for c in feats if c != "f_mkt_logp" and c not in FORM]
    base_with = base_no + ["f_mkt_logp"]
    cfgs = {
        "no odds": ({"cols": base_no, "l2": 20.0, "discount": True}, {"cols": base_no + FORM, "l2": 20.0, "discount": True}),
        "with odds": ({"cols": base_with, "l2": 10.0, "discount": True}, {"cols": base_with + FORM, "l2": 10.0, "discount": True}),
    }
    has_form = {r["id"]: not np.isnan(r["raw"][:, [feats.index(c) for c in FORM]]).all() for r in races}
    rep = {"races_with_form_features": {p: sum(has_form[r["id"]] for r in races if r["month"] in ms)
                                        for p, ms in (("validation", mv), ("test", mt))}}
    adopt = True
    for label, (without, with_form) in cfgs.items():
        a = ev.walk_forward(races, feats, without, mv + mt)
        b = ev.walk_forward(races, feats, with_form, mv + mt)
        rep[label] = {}
        for period, ms in (("validation", mv), ("test", mt)):
            d = [(r["date"], ev.pl_loglik(*b[r["id"]], r["order"]) - ev.pl_loglik(*a[r["id"]], r["order"]))
                 for r in races if r["month"] in ms]
            gain = float(np.mean([x for _, x in d]))
            rep[label][period] = {"mean_ll_gain": round(gain, 4), "ci95": ev.block_bootstrap(d, np.mean), "races": len(d)}
            adopt &= gain > 0
        print(label, json.dumps(rep[label]))
    rep["adopt_in_production"] = adopt
    print("races with form features:", rep["races_with_form_features"], "-> adopt:", adopt)
    (ev.OUT / "form_eval.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
