"""
Pre-registered hold-out test (research/PREREGISTRATION.md): frozen models and strategies, no tuning.

    <venv>/python research/holdout.py 2024-11 2025-10     # the pre-registered hold-out year
    <venv>/python research/holdout.py 2026-06 2026-10     # dry run on months already seen

Writes research/results/holdout_<from>_<to>.json and prints the tables and the pre-declared verdicts.
"""
import itertools
import json
import sys
import zlib

import numpy as np

import evaluate as ev
import plmodel as plm

EV_THRESHOLD = 2.0  # chosen on the 2026 validation months, frozen


def paired(a, b, months):
    """Mean per-race log-likelihood difference a - b over the months, with a week-block 95% CI."""
    ib = {x["id"]: x["ll_top5_order"] for x in b}
    d = [(x["date"], x["ll_top5_order"] - ib[x["id"]]) for x in a if x["id"][:7] in months and x["id"] in ib]
    return {"mean": round(float(np.mean([v for _, v in d])), 4), "ci95": ev.block_bootstrap(d, np.mean), "races": len(d)}


def main(lo, hi):
    data = ev.load_versions()
    races1, f1 = data["v1"]
    races2, f2 = data["v2"]
    months = ev.months_between(races2, lo, hi)
    payouts = ev.payout_table()
    rng = np.random.default_rng(2024)
    complete = {r["id"]: not np.isnan(r["odds"]).any() for r in races2}
    wf = ev.walk_forward
    models = {
        "uniform": {r["id"]: (np.zeros(len(r["nums"])), np.ones(5)) for r in races2 if r["month"] in months},
        "market raw (T-30)": {k: v for k, v in ev.market_preds(races2, months).items() if complete[k]},
        "market fitted (T-30)": wf(races2, f2, {"cols": ["f_mkt_logp"], "l2": 0.01, "discount": True}, months),
        "before, no odds": wf(races1, f1, {"cols": [c for c in f1 if c != "f_mkt_logp"], "l2": 5.0}, months),
        "before, with odds": wf(races1, f1, {"cols": f1, "l2": 5.0}, months),
        "after, no odds": wf(races2, f2, {"cols": [c for c in f2 if c != "f_mkt_logp"], "l2": 20.0, "discount": True}, months),
        "after, with odds": wf(races2, f2, {"cols": f2, "l2": 10.0, "discount": True}, months),
    }
    evals = {}
    for name, preds in models.items():
        rs = races1 if name.startswith("before") else races2
        main_model = name == "after, with odds"
        evals[name] = ev.evaluate_preds(rs, preds, rng, sims=ev.SIMS if main_model else 10000, keep_orders=main_model)
    rule_of = {r["id"]: ev.race_rule(r, payouts) for r in races2}
    rep = {"period": [lo, hi], "months": months, "races": len(evals["uniform"]),
           "races_with_incomplete_odds": sum(not complete[x["id"]] for x in evals["uniform"]),
           "races_by_rule": {k: sum(rule_of[x["id"]] == k for x in evals["uniform"]) for k in ("BONUS_4", "BONUS_4SUR5")},
           "races_by_family": {f: sum(x["family"] == f for x in evals["uniform"]) for f in ("PLAT", "TROT", "OBSTACLE")}}
    rep["prediction"] = {name: ev.summary_metrics(e) for name, e in evals.items()}
    rep["prediction_by_family"] = {name: {f: round(float(np.mean([x["ll_top5_order"] for x in e if x["family"] == f])), 4)
                                          for f in ("PLAT", "TROT", "OBSTACLE") if any(x["family"] == f for x in e)}
                                   for name, e in evals.items()}
    rep["comparisons"] = {
        "after minus before, no odds": paired(evals["after, no odds"], evals["before, no odds"], months),
        "after minus before, with odds": paired(evals["after, with odds"], evals["before, with odds"], months),
        "after with odds minus market fitted": paired(evals["after, with odds"], evals["market fitted (T-30)"], months),
        "before with odds minus market fitted": paired(evals["before, with odds"], evals["market fitted (T-30)"], months),
    }

    # paper betting, one 2 € ticket per race, official dividends
    by = {name: {x["id"]: x for x in e} for name, e in evals.items()}
    rows = {k: [] for k in ("favourites (T-30)", "random", "top 5 before, no odds", "top 5 before, with odds",
                            "top 5 after, no odds", "top 5 after, with odds", "bon rapport (after, T-30)",
                            f"expected value > {EV_THRESHOLD:g} € (after, T-30)")}
    pms = {}
    by_id2 = {r["id"]: r for r in races2}
    for r in (by_id2[x["id"]] for x in evals["uniform"]):
        base = {"date": r["date"], "family": r["family"], "month": r["month"], "rule": rule_of[r["id"]]}

        def add(name, idx):
            if idx is None:  # no bet
                rows[name].append({**base, "bet": False, "ret": 0.0, "rank": None})
                return
            rank, ret = ev.settle(r, list(idx), payouts)
            rows[name].append({**base, "bet": True, "ret": ret, "rank": rank})

        n = len(r["nums"])
        ok = complete[r["id"]]
        add("favourites (T-30)", np.argsort(r["odds"])[:5] if ok else None)
        add("random", np.random.default_rng(zlib.crc32(r["id"].encode())).permutation(n)[:5])
        for tag, name in (("before, no odds", "before, no odds"), ("before, with odds", "before, with odds"),
                          ("after, no odds", "after, no odds"), ("after, with odds", "after, with odds")):
            add(f"top 5 {tag}", np.argsort(-by[name][r["id"]]["p5"])[:5])
        if ok:
            pf, mk = by["after, no odds"][r["id"]]["p5"], by["market raw (T-30)"][r["id"]]["p5"]
            short = list(np.argsort(-pf)[:8])
            add("bon rapport (after, T-30)", sorted(sorted(short, key=lambda i: -pf[i] / max(mk[i], 1e-6))[:5], key=lambda i: -pf[i]))
        else:
            add("bon rapport (after, T-30)", None)
        pick = None
        if ok:
            if r["month"] not in pms:
                pms[r["month"]] = ev.PayoutModel([x for x in races2 if x["month"] < r["month"]], payouts)
            s, lam = models["after, with odds"][r["id"]]
            top8 = list(np.argsort(-s)[:8])
            cands = [tuple(sorted(c, key=lambda i: -s[i])) for c in itertools.combinations(top8, 5)]
            try:
                best = max(ev.ticket_evs(r, s, lam, by["after, with odds"][r["id"]]["orders"], pms[r["month"]], cands,
                                         rule_of[r["id"]]), key=lambda e: e["ev_net"])
                pick = best["ticket"] if best["ev_net"] > EV_THRESHOLD else None
            except KeyError:  # payout category without enough earlier races to estimate it: no bet
                pick = None
        add(f"expected value > {EV_THRESHOLD:g} € (after, T-30)", pick)

    rep["betting"] = {name: ev.bet_summary(rs) for name, rs in rows.items()}
    rep["betting_by_rule"] = {rule: {name: ev.bet_summary([x for x in rs if x["rule"] == rule]) for name, rs in rows.items()}
                              for rule in ("BONUS_4", "BONUS_4SUR5") if any(x["rule"] == rule for x in rows["random"])}

    # pre-declared verdicts (PREREGISTRATION.md)
    c = rep["comparisons"]
    rep["verdicts"] = {
        "after improves on before (no odds)": c["after minus before, no odds"]["mean"] > 0 and c["after minus before, no odds"]["ci95"][0] > 0,
        "after improves on before (with odds)": c["after minus before, with odds"]["mean"] > 0 and c["after minus before, with odds"]["ci95"][0] > 0,
        "model adds information beyond the odds": c["after with odds minus market fitted"]["ci95"][0] > 0,
        "strategies with a demonstrated edge": [n for n, b in rep["betting"].items()
                                                if b.get("roi_ci95_week_blocks") and b["roi_ci95_week_blocks"][0] > 0
                                                and (b["roi_without_top1"] or -1) > 0],
    }
    ev.OUT.mkdir(parents=True, exist_ok=True)
    (ev.OUT / f"holdout_{lo}_{hi}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    show(rep)


def show(rep):
    print(f"\nHold-out {rep['period'][0]} -> {rep['period'][1]}: {rep['races']} races, by rule {rep['races_by_rule']}, "
          f"by family {rep['races_by_family']}, incomplete odds {rep['races_with_incomplete_odds']}")
    print(f"{'model':<24}{'ll top5':>9}{'win LL':>8}{'top5 Br':>9}{'found':>7}{'ECE t5':>8}")
    for name, m in rep["prediction"].items():
        print(f"{name:<24}{m['ll_top5_order']:>9.3f}{m['win_logloss']:>8.3f}{m['top5_brier']:>9.4f}{m['found']:>7.2f}"
              f"{m['top5_calibration']['ece']:>8.3f}")
    for k, v in rep["comparisons"].items():
        print(f"  {k:<40}{v['mean']:+.4f}  {v['ci95']}")
    print(f"\n{'strategy':<34}{'bets':>5}{'staked':>8}{'back':>9}{'ROI':>7}{'CI95':>20}{'-top1':>7}{'maxDD':>7}  hits")
    for name, b in rep["betting"].items():
        roi = b["roi"] if b["roi"] is not None else float("nan")
        print(f"{name:<34}{b['bets']:>5}{b['staked']:>8.0f}{b['returned']:>9.1f}{roi:>7.2f}{str(b.get('roi_ci95_week_blocks', '-')):>20}"
              f"{(b['roi_without_top1'] if b['roi_without_top1'] is not None else float('nan')):>7.2f}{b['max_drawdown']:>7.0f}  {b['hits']}")
    print("\nPre-declared verdicts:", json.dumps(rep["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
