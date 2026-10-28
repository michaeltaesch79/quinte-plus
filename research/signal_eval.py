"""
Two fixes suggested by the factor scan (the market and the model both overrate horses with a negative
trainer opinion and rank outsiders):
  - trainer opinion as two indicators (negative, positive) instead of one -1/0/+1 scale;
  - outsider correction for the with-odds model: max(0, log(odds / 30)), 0 below odds 30.
Production settings otherwise, same walk-forward protocol as evaluate.py (validation 2026-02..05,
test 2026-06..10). Rule, declared before running: adopt a fix only if the mean log-likelihood gain is
positive on BOTH the validation and the test months for each model it applies to.
Caveat: the scan that suggested these fixes looked at all 3 years, these months included, so a pass
here is weaker evidence than the pre-registered hold-out. The clean test is the months to come.

    <venv>/python research/signal_eval.py
Writes research/results/signal_eval.json.
"""
import json

import numpy as np

import evaluate as ev

FORM = ["f_jockey_form90", "f_jockey_runs90", "f_trainer_form90", "f_trainer_runs90", "f_trainer_form30"]


def add_columns(races, feats):
    a = feats.index("f_avis")
    for r in races:
        avis = r["raw"][:, a]
        odds = r["odds"]
        with np.errstate(invalid="ignore"):
            extra = np.column_stack([np.where(np.isnan(avis), np.nan, (avis == -1).astype(float)),
                                     np.where(np.isnan(avis), np.nan, (avis == 1).astype(float)),
                                     np.maximum(0.0, np.log(odds / 30.0))])
        r["raw"] = np.hstack([r["raw"], extra])
    return feats + ["f_avis_neg", "f_avis_pos", "f_outsider"]


def main():
    races, feats = ev.load_versions()["v2"]
    feats = add_columns(races, feats)
    mv, mt = ev.months_between(races, *ev.VALID), ev.months_between(races, *ev.TEST)
    no = [c for c in feats if c not in FORM and c not in ("f_mkt_logp", "f_avis_neg", "f_avis_pos", "f_outsider")]
    with_ = no + ["f_mkt_logp"]
    split = lambda cols: [c for c in cols if c != "f_avis"] + ["f_avis_neg", "f_avis_pos"]
    tests = {
        "avis split, no odds": ({"cols": no, "l2": 20.0}, {"cols": split(no), "l2": 20.0}),
        "avis split, with odds": ({"cols": with_, "l2": 10.0}, {"cols": split(with_), "l2": 10.0}),
        "outsider correction, with odds": ({"cols": with_, "l2": 10.0}, {"cols": with_ + ["f_outsider"], "l2": 10.0}),
        "both, with odds": ({"cols": with_, "l2": 10.0}, {"cols": split(with_) + ["f_outsider"], "l2": 10.0}),
    }
    rep = {}
    for label, (a_cfg, b_cfg) in tests.items():
        a = ev.walk_forward(races, feats, {**a_cfg, "discount": True}, mv + mt)
        b = ev.walk_forward(races, feats, {**b_cfg, "discount": True}, mv + mt)
        rep[label] = {}
        for period, ms in (("validation", mv), ("test", mt)):
            d = [(r["date"], ev.pl_loglik(*b[r["id"]], r["order"]) - ev.pl_loglik(*a[r["id"]], r["order"]))
                 for r in races if r["month"] in ms]
            rep[label][period] = {"mean_ll_gain": round(float(np.mean([x for _, x in d])), 4),
                                  "ci95": ev.block_bootstrap(d, np.mean), "races": len(d)}
        rep[label]["passes_rule"] = all(rep[label][p]["mean_ll_gain"] > 0 for p in ("validation", "test"))
        print(label, json.dumps(rep[label]))
    (ev.OUT / "signal_eval.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
