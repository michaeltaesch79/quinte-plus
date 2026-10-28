"""
Paper betting, one 2 € Quinté+ ticket per race, official dividends: the model before ("prod") vs after
("fix") the trainer-opinion/outsider change of signal_eval.py, on the hold-out year 2024-11..2025-10 and
on 2026-02..10. Strategies: top 5 without odds, top 5 with odds, bon rapport (production rule).

    <venv>/python research/signal_bets.py      (~40 min)
"""
import numpy as np

import evaluate as ev
import plmodel as plm
import signal_eval as se

races, feats = ev.load_versions()["v2"]
feats = se.add_columns(races, feats)
no = [c for c in feats if c not in se.FORM and c not in ("f_mkt_logp", "f_avis_neg", "f_avis_pos", "f_outsider")]
w = no + ["f_mkt_logp"]
split = lambda cols: [c for c in cols if c != "f_avis"] + ["f_avis_neg", "f_avis_pos"]
months = ev.months_between(races, "2024-11", "2025-10") + ev.months_between(races, "2026-02", "2026-10")
cfgs = {"prod no": {"cols": no, "l2": 20.0}, "fix no": {"cols": split(no), "l2": 20.0},
        "prod with": {"cols": w, "l2": 10.0}, "fix with": {"cols": split(w) + ["f_outsider"], "l2": 10.0}}
P = {k: ev.walk_forward(races, feats, {**c, "discount": True}, months) for k, c in cfgs.items()}
payouts = ev.payout_table()
rng = np.random.default_rng(7)
p5 = lambda s, lam, n: plm.top5_probs(plm.sample_orders(s, np.asarray(lam), 20000, rng), n)
rows = {}
for r in races:
    if r["month"] not in months or np.isnan(r["odds"]).any():
        continue
    n = len(r["nums"])
    period = "hold-out 2024-25" if r["month"] <= "2025-10" else "2026"
    mk = p5(np.log(1 / r["odds"]), np.ones(5), n)
    q = {k: p5(*P[k][r["id"]], n) for k in P}
    picks = {}
    for tag in ("prod", "fix"):
        picks[f"top 5 no odds, {tag}"] = np.argsort(-q[f"{tag} no"])[:5]
        picks[f"top 5 with odds, {tag}"] = np.argsort(-q[f"{tag} with"])[:5]
        pf = q[f"{tag} no"]
        short = list(np.argsort(-pf)[:8])
        picks[f"bon rapport, {tag}"] = sorted(short, key=lambda i: -pf[i] / max(mk[i], 1e-6))[:5]
    for name, idx in picks.items():
        rank, ret = ev.settle(r, list(idx), payouts)
        for p in (period, "both periods"):
            rows.setdefault((p, name), []).append({"date": r["date"], "family": r["family"], "month": r["month"],
                                                   "bet": True, "ret": ret, "rank": rank})
for (p, name), rs in sorted(rows.items()):
    b = ev.bet_summary(rs)
    print(f"{p:<18}{name:<26} bets {b['bets']:>4} back {b['returned']:>8.0f} ROI {b['roi']:+.2f} CI {b.get('roi_ci95_week_blocks')} "
          f"w/o top1 {b['roi_without_top1']:+.2f}  {b['hits']}")
