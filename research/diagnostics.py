"""
Post-hoc diagnostics (they look at the test months, so nothing here is used as a clean result):
1. decomposition: which change moves the log-likelihood (leak fix + new features, settings, dropped groups);
2. paired confidence intervals new vs old model;
3. expected-value model: predicted vs observed hit counts per payout category, and predicted vs
   actual dividends when a category is hit (is the error in the probabilities or the payouts?).

    <venv>/python research/diagnostics.py
    <venv>/python research/diagnostics.py --production   # betting results of the production settings
Writes research/results/diagnostics.json.
"""
import json
import sys

import numpy as np

import evaluate as ev
import plmodel as plm


def ll_by_race(races, preds):
    return {r["id"]: ev.pl_loglik(*preds[r["id"]], r["order"]) for r in races if r["id"] in preds}


def main():
    data = ev.load_versions()
    races1, f1 = data["v1"]
    races2, f2 = data["v2"]
    mv = ev.months_between(races2, *ev.VALID)
    mt = ev.months_between(races2, *ev.TEST)
    both = mv + mt
    rep = json.loads((ev.OUT / "evaluation.json").read_text(encoding="utf-8"))
    dropped = rep["choices"]["morning_features_dropped"]
    no_mkt2 = [c for c in f2 if c != "f_mkt_logp"]
    kept = [c for c in no_mkt2 if c not in dropped]
    variants = {
        "a) v1 features (leaky), L2 5, plain PL": (races1, f1, {"cols": [c for c in f1 if c != "f_mkt_logp"], "l2": 5.0}),
        "b) v2 features (leak fixed + new), L2 5, plain PL": (races2, f2, {"cols": no_mkt2, "l2": 5.0}),
        "c) b + L2 20 + stage discounts": (races2, f2, {"cols": no_mkt2, "l2": 20.0, "discount": True}),
        "d) c + groups dropped on validation (= v2)": (races2, f2, {"cols": kept, "l2": 20.0, "discount": True}),
    }
    variants_odds = {
        "a) v1 features (leaky), L2 5, plain PL": (races1, f1, {"cols": f1, "l2": 5.0}),
        "b) v2 features, L2 5, plain PL": (races2, f2, {"cols": f2, "l2": 5.0}),
        "c) b + L2 10 + stage discounts": (races2, f2, {"cols": f2, "l2": 10.0, "discount": True}),
        "d) c + groups dropped on validation (= v2)": (races2, f2, {"cols": kept + ["f_mkt_logp"], "l2": 10.0, "discount": True}),
        "market fitted": (races2, f2, {"cols": ["f_mkt_logp"], "l2": 0.01, "discount": True}),
    }
    out = {}
    for label, vs in (("morning (no odds)", variants), ("T-30 (with odds)", variants_odds)):
        out[label] = {}
        lls = {}
        for name, (rs, feats, cfg) in vs.items():
            lls[name] = ll_by_race(rs, ev.walk_forward(rs, feats, cfg, both))
            out[label][name] = {p: round(float(np.mean([v for k, v in lls[name].items() if k[:7] in ms])), 4)
                                for p, ms in (("validation", mv), ("test", mt))}
        first, last = list(vs)[0], [k for k in vs if k.startswith("d)")][0]
        diffs = [(k[:10], lls[last][k] - lls[first][k]) for k in lls[last] if k[:7] in mt]
        out[label]["v2 minus v1, test, mean and week-block CI"] = [round(float(np.mean([d for _, d in diffs])), 4),
                                                                   ev.block_bootstrap(diffs, np.mean)]
        diffs = [(k[:10], lls[last][k] - lls[first][k]) for k in lls[last]]
        out[label]["v2 minus v1, validation+test, mean and week-block CI"] = [round(float(np.mean([d for _, d in diffs])), 4),
                                                                              ev.block_bootstrap(diffs, np.mean)]
        print(label, json.dumps(out[label], indent=1))

    # expected-value model: where does it go wrong?
    payouts = ev.payout_table()
    cfg = {"cols": kept + ["f_mkt_logp"], "l2": 10.0, "discount": True}
    preds = ev.walk_forward(races2, f2, cfg, both)
    rng = np.random.default_rng(5)
    acc = {c: {"expected_hits": 0.0, "observed_hits": 0} for c in ("set (Ordre/Désordre)", "Bonus 4sur5", "Bonus 3")}
    pay_ratio = {"DESORDRE": [], "BONUS_4SUR5": [], "BONUS_3": []}
    ev_cat = {c: {"predicted_eur": 0.0, "realised_eur": 0.0} for c in ("ORDRE", "DESORDRE", "BONUS_4SUR5", "BONUS_3")}
    pms = {}
    for r in (x for x in races2 if x["month"] in both):
        s, lam = preds[r["id"]]
        orders = plm.sample_orders(s, lam, 20000, rng)
        t = tuple(np.argsort(-s)[:5])
        T = np.zeros(len(s), bool)
        T[list(t)] = True
        k5 = T[orders].sum(axis=1)
        k3 = T[orders[:, :3]].sum(axis=1)
        p_set = plm.set_prob(s, lam, t)
        acc["set (Ordre/Désordre)"]["expected_hits"] += p_set
        acc["Bonus 4sur5"]["expected_hits"] += float((k5 == 4).mean())
        acc["Bonus 3"]["expected_hits"] += float(((k5 == 3) & (k3 == 3)).mean())
        rank, ret = ev.settle(r, list(t), payouts)
        if rank:
            ev_cat[rank]["realised_eur"] += ret
        acc["set (Ordre/Désordre)"]["observed_hits"] += rank in ("ORDRE", "DESORDRE")
        acc["Bonus 4sur5"]["observed_hits"] += rank == "BONUS_4SUR5"
        acc["Bonus 3"]["observed_hits"] += rank == "BONUS_3"
        if r["month"] not in pms:
            pms[r["month"]] = ev.PayoutModel([x for x in races2 if x["month"] < r["month"]], payouts)
        lq = ev.logq(r)
        pm = pms[r["month"]]
        p_ord = plm.order_prob(s, lam, t)
        pop_t = lq[list(t)].sum()
        ev_cat["ORDRE"]["predicted_eur"] += ev.STAKE * p_ord * float(pm.mean("ORDRE", r, pop_t))
        ev_cat["DESORDRE"]["predicted_eur"] += ev.STAKE * (p_set - p_ord) * float(pm.mean("DESORDRE", r, pop_t))
        ev_cat["BONUS_4SUR5"]["predicted_eur"] += ev.STAKE * float(((k5 == 4) * pm.mean("BONUS_4SUR5", r, lq[orders].sum(axis=1))).mean())
        ev_cat["BONUS_3"]["predicted_eur"] += ev.STAKE * float((((k5 == 3) & (k3 == 3)) * pm.mean("BONUS_3", r, lq[orders[:, :3]].sum(axis=1))).mean())
        for cat, depth in (("DESORDRE", 5), ("BONUS_4SUR5", 5), ("BONUS_3", 3)):
            vals = [e for p, c, e in payouts.get(r["id"], []) if p == "QUINTE_PLUS:" + cat and e > 0]
            if vals:
                pred = float(pms[r["month"]].mean(cat, r, lq[r["order"][:depth]].sum()))
                pay_ratio[cat].append(np.mean(vals) / pred)
    out["ev_model"] = {
        "hits of the model's own top-5 ticket, validation+test": {k: {"expected": round(v["expected_hits"], 1),
                                                                     "observed": v["observed_hits"]} for k, v in acc.items()},
        "actual / predicted dividend for the real arrival (median, mean)": {
            k: [round(float(np.median(v)), 3), round(float(np.mean(v)), 3)] for k, v in pay_ratio.items()},
        "euros returned by category, predicted vs realised (2 € tickets)": {
            k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in ev_cat.items()},
        "races": len([x for x in races2 if x["month"] in both]),
    }
    print(json.dumps(out["ev_model"], indent=1, ensure_ascii=False))
    (ev.OUT / "diagnostics.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__" and "--production" not in sys.argv:
    main()


def production_betting():
    """Betting results of the configuration used in production (chosen after seeing the test: labelled post-hoc)."""
    data = ev.load_versions()
    races2, f2 = data["v2"]
    mv = ev.months_between(races2, *ev.VALID)
    mt = ev.months_between(races2, *ev.TEST)
    both = mv + mt
    payouts = ev.payout_table()
    no_mkt = [c for c in f2 if c != "f_mkt_logp"]
    rng = np.random.default_rng(11)
    form = {x["id"]: x for x in ev.evaluate_preds(races2, ev.walk_forward(races2, f2, {"cols": no_mkt, "l2": 20.0, "discount": True}, both), rng)}
    odds = {x["id"]: x for x in ev.evaluate_preds(races2, ev.walk_forward(races2, f2, {"cols": f2, "l2": 10.0, "discount": True}, both), rng)}
    mkt = {x["id"]: x for x in ev.evaluate_preds(races2, ev.market_preds(races2, both), rng)}
    rows = {"top5 production with odds": [], "top5 production no odds": [], "bon rapport production": []}
    for r in (x for x in races2 if x["month"] in both):
        base = {"date": r["date"], "family": r["family"], "month": r["month"], "period": "validation" if r["month"] in mv else "test"}
        pf, mk = form[r["id"]]["p5"], mkt[r["id"]]["p5"]
        short = list(np.argsort(-pf)[:8])
        value = sorted(sorted(short, key=lambda i: -pf[i] / max(mk[i], 1e-6))[:5], key=lambda i: -pf[i])
        for name, idx in (("top5 production with odds", list(np.argsort(-odds[r["id"]]["p5"])[:5])),
                          ("top5 production no odds", list(np.argsort(-pf)[:5])), ("bon rapport production", value)):
            rank, ret = ev.settle(r, idx, payouts)
            rows[name].append({**base, "bet": True, "ret": ret, "rank": rank})
    out = {p: {n: ev.bet_summary([x for x in rs if x["period"] == p]) for n, rs in rows.items()} for p in ("validation", "test")}
    out["found_per_race_test"] = {"with odds": round(float(np.mean([odds[k]["found"] for k in odds if k[:7] in mt])), 2),
                                  "no odds": round(float(np.mean([form[k]["found"] for k in form if k[:7] in mt])), 2)}
    cal = {"with odds": ev.summary_metrics([x for x in odds.values() if x["id"][:7] in mt]),
           "no odds": ev.summary_metrics([x for x in form.values() if x["id"][:7] in mt])}
    out["prediction_test"] = {k: {m: v[m] for m in ("ll_top5_order", "win_logloss", "win_brier", "top5_logloss", "top5_brier", "found")}
                              | {"ece_win": v["win_calibration"]["ece"], "ece_top5": v["top5_calibration"]["ece"],
                                 "top5_calibration": v["top5_calibration"]["bins"]} for k, v in cal.items()}
    for p in ("validation", "test"):
        for n, b in out[p].items():
            print(f"{p:<11}{n:<28}{b['bets']:>5}{b['returned']:>9.1f}{b['roi']:>7.2f} {b.get('roi_ci95_week_blocks')} -top1 {b['roi_without_top1']} {b['hits']}")
    print(json.dumps(out["prediction_test"], indent=1))
    d = json.loads((ev.OUT / "diagnostics.json").read_text(encoding="utf-8"))
    d["production_config_post_hoc"] = out
    (ev.OUT / "diagnostics.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__" and "--production" in sys.argv:
    production_betting()
