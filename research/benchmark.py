"""
One benchmark for every idea: the current production model, the market and the reference tickets,
walk-forward over every Quinté+ from 2024-11 to the latest race in data/. A candidate model or
ticket rule is scored on exactly the same races, with the same random numbers, next to the baseline.

    <venv>/python research/benchmark.py                       # baseline -> results/baseline_<last race>.json
    <venv>/python research/benchmark.py --with research/ref_v2.py --tag ref_v2
                                                              # + candidates -> results/benchmark_ref_v2_<last race>.json

Protocol (same as evaluate.py / holdout.py): refit every month on all earlier races; cutoffs MORNING
(no odds) and T-30 (reference odds); post-start odds never used; one 2 € Quinté+ ticket per race, settled
with the official dividends; 95% intervals by resampling whole weeks. The production settings are read
from daily.py and features.py, so the baseline follows production.

Periods (race days, included). Everything up to 2026-10-28 was already seen by earlier research;
only "fresh" is new. "all" also holds 2025-11..2026-01, which belong to no named period.

A candidate file is a Python module with MODELS and/or STRATEGIES (lists of dicts):
  model:    name, versus (a model to compare with), and either
              cfg      a walk_forward config {"cols", "l2", "discount"} or a function feats -> config
              predict  a function (races, feats, months) -> {race_id: (strengths, stage factors)}
            optional prepare(races, feats, data) -> names of the columns it appended to every r["raw"]
            (data = {"runners", "past"}: the CSV rows; use only what is known before the race).
  strategy: name, versus (a strategy to compare with), pick(race, ctx) -> 5 runner indices in ticket
            order, or None for no bet. ctx: "ev" {model: probabilities and sampled orders}, "preds"
            {model: (strengths, stage factors)}, "complete" (every starter has reference odds), "rule"
            (bonus rule of the race), "payout_model"() (fitted on earlier months only) and "crowd" (the
            12 most-played sets: FINAL stakes, not known before the race, for diagnostics only).
"""
import argparse
import csv
import importlib.util
import itertools
import json
import zlib
from collections import defaultdict
from datetime import datetime

import numpy as np

import evaluate as ev
import features as ft
import crowd_eval
import daily
from paths import DATA

START = "2024-11"  # first test month: the first fit then has 12 months of races, as in production
PERIODS = {
    "hold-out 2024-11..2025-10": ("2024-11-01", "2025-10-31"),
    "validation 2026-02..05": ("2026-02-01", "2026-05-31"),
    "test 2026-06..10": ("2026-06-01", "2026-10-28"),
    "fresh 2026-10-29..": ("2026-10-29", "9999"),
    "all": ("2024-11-01", "9999"),
}
SIMS = 20000
NO_ODDS, WITH_ODDS, MARKET = "prod, no odds", "prod, with odds", "market fitted (T-30)"
EV_THRESHOLD = 2.0  # euros, chosen on the 2026 validation months (PREREGISTRATION.md)
FAMILIES = ("PLAT", "TROT", "OBSTACLE")


# ---------------------------------------------------------------- data and baseline models

def load_data():
    runners, past = ev.load_csv(DATA / "runners.csv"), ev.load_csv(DATA / "past_runs.csv")
    odds_ref = {(r["race_id"], r["numPmu"]): float(r["cote_reference"]) for r in runners if r["cote_reference"]}
    races, feats = ev.build_races(ft.compute(runners, past), odds_ref)  # as daily.py: no all-race form
    assert max(len(r["nums"]) for r in races) <= ev.N_PAD, "a field is larger than evaluate.N_PAD"
    return races, feats, {"runners": runners, "past": past}


def baseline_models(feats):
    cols = [c for c in feats if c not in daily.DROPPED]
    return [
        {"name": "uniform", "predict": lambda races, feats, months: {
            r["id"]: (np.zeros(len(r["nums"])), np.ones(5)) for r in races if r["month"] in months}},
        {"name": "market raw (T-30)", "predict": lambda races, feats, months: {
            k: v for k, v in ev.market_preds(races, months).items() if not np.isnan(v[0]).any()}},
        {"name": MARKET, "cfg": {"cols": ["f_mkt_logp"], "l2": 0.01, "discount": True}},
        {"name": NO_ODDS, "cfg": {"cols": [c for c in cols if c not in ft.ODDS],
                                  "l2": daily.MODELS["sans_cote"]["l2"], "discount": True}},
        {"name": WITH_ODDS, "versus": MARKET, "cfg": {"cols": cols, "l2": daily.MODELS["avec_cote"]["l2"], "discount": True}},
    ]


# ---------------------------------------------------------------- ticket rules

def top5(model):
    """The five strongest runners of a model, strongest first (the app's prono). P(top 5) rises with the
    strength, so this is the top 5 by P(top 5) without the simulation noise that flips near-ties."""
    return lambda r, c: list(np.argsort(-c["preds"][model][0], kind="stable")[:5]) if model in c["ev"] else None


def favourites(r, c):
    return list(np.argsort(r["odds"])[:5]) if c["complete"] else None


def random5(r, c):
    return list(np.random.default_rng(zlib.crc32(r["id"].encode())).permutation(len(r["nums"]))[:5])


def bon_rapport(model):
    """Production rule: among the model's top 8, the 5 with the highest model / raw market P(top 5)."""
    def pick(r, c):
        if not c["complete"] or model not in c["ev"]:
            return None
        pf, mk = c["ev"][model]["p5"], c["ev"]["market raw (T-30)"]["p5"]
        short = list(np.argsort(-pf)[:8])
        return sorted(sorted(short, key=lambda i: -pf[i] / max(mk[i], 1e-6))[:5], key=lambda i: -pf[i])
    return pick


def expected_value(model, threshold=EV_THRESHOLD):
    """Best expected-value ticket among the 5-horse sets of the model's top 8; no bet under the threshold."""
    def pick(r, c):
        if not c["complete"] or model not in c["ev"]:
            return None
        s, lam = c["preds"][model]
        top8 = list(np.argsort(-s)[:8])
        cands = [tuple(sorted(t, key=lambda i: -s[i])) for t in itertools.combinations(top8, 5)]
        try:
            best = max(ev.ticket_evs(r, s, lam, c["ev"][model]["orders"], c["payout_model"](), cands, c["rule"]),
                       key=lambda e: e["ev_net"])
        except KeyError:  # payout category without enough earlier races to estimate it: no bet
            return None
        return list(best["ticket"]) if best["ev_net"] > threshold else None
    return pick


def baseline_strategies():
    return [
        {"name": "random", "pick": random5},
        {"name": "favourites (T-30)", "pick": favourites},
        {"name": "top 5 " + NO_ODDS, "pick": top5(NO_ODDS)},
        {"name": "top 5 " + WITH_ODDS, "versus": "favourites (T-30)", "pick": top5(WITH_ODDS)},
        {"name": "bon rapport (prod rule)", "pick": bon_rapport(NO_ODDS)},
        {"name": f"expected value > {EV_THRESHOLD:g} € ({WITH_ODDS})", "pick": expected_value(WITH_ODDS)},
    ]


# ---------------------------------------------------------------- run

def run(models=(), strategies=()):
    """-> (report, per-race rows). models / strategies: the candidates, see the module docstring."""
    races, feats, data = load_data()
    feats = list(feats)
    all_models = baseline_models(feats) + list(models)
    all_strategies = baseline_strategies() + list(strategies)
    for m in models:
        if m.get("prepare"):
            feats += m["prepare"](races, feats, data)
    months = sorted({r["month"] for r in races if r["month"] >= START})
    preds = {}
    for m in all_models:
        if "predict" in m:
            preds[m["name"]] = m["predict"](races, feats, months)
        else:
            preds[m["name"]] = ev.walk_forward(races, feats, m["cfg"](feats) if callable(m["cfg"]) else m["cfg"], months)
        print(f"  predicted: {m['name']}")

    payouts, crowd = ev.payout_table(), crowd_eval.load_crowd()
    pms, evals, bets, per_race = {}, defaultdict(list), defaultdict(list), []
    tested = [r for r in races if r["month"] >= START]
    for r in tested:
        seed = zlib.crc32(r["id"].encode())  # same random numbers for every model: paired, and stable when races are added
        e = {name: ev.evaluate_preds([r], p, np.random.default_rng(seed), SIMS, keep_orders=True)[0]
             for name, p in preds.items() if r["id"] in p}

        def payout_model(month=r["month"]):
            if month not in pms:
                pms[month] = ev.PayoutModel([x for x in races if x["month"] < month], payouts)
            return pms[month]

        ctx = {"ev": e, "preds": {name: preds[name][r["id"]] for name in e}, "complete": not np.isnan(r["odds"]).any(),
               "rule": ev.race_rule(r, payouts), "payout_model": payout_model, "crowd": crowd.get(r["id"])}
        line = {"race_id": r["id"], "family": r["family"], **{"ll " + name: round(x["ll_top5_order"], 4) for name, x in e.items()}}
        for st in all_strategies:
            idx = st["pick"](r, ctx)
            row = {"date": r["date"], "family": r["family"], "month": r["month"], "bet": idx is not None, "ret": 0.0, "rank": None}
            if idx is not None:
                idx = [int(i) for i in idx]
                assert len(set(idx)) == 5, f"{st['name']}: a ticket is 5 different runners"
                row["rank"], row["ret"] = ev.settle(r, idx, payouts)
                row["found"] = int(r["top5"][idx].sum())
                if ctx["crowd"] is not None:
                    row["crowd"] = frozenset(r["nums"][i] for i in idx) in ctx["crowd"]
                line["ticket " + st["name"]] = "-".join(str(r["nums"][i]) for i in idx)
            line["return " + st["name"]] = row["ret"]
            bets[st["name"]].append(row)
        for name, x in e.items():
            x["orders"] = None
            evals[name].append(x)
        per_race.append(line)

    comparisons = [(m["name"], m["versus"]) for m in all_models if m.get("versus")] + [(NO_ODDS, MARKET)]
    pairs = [(s["name"], s["versus"]) for s in all_strategies if s.get("versus")]
    rep = {"generated": datetime.now().isoformat(timespec="seconds"),
           "data": {"races_in_data": len(races), "first": races[0]["date"], "last": races[-1]["date"],
                    "races_tested": len(tested), "tested_from": tested[0]["date"]},
           "settings": {"sims": SIMS, "stake": ev.STAKE, "production": daily.MODELS, "features": len(baseline_models(feats)[-1]["cfg"]["cols"])},
           "periods": {}}
    for period, (lo, hi) in PERIODS.items():
        ids = {r["id"] for r in tested if lo <= r["date"] <= hi}
        if not ids:
            continue
        sub = {name: [x for x in v if x["id"] in ids] for name, v in evals.items()}
        out = {"races": len(ids), "from": min(ids)[:10], "to": max(ids)[:10],
               "races_by_family": {f: sum(x["family"] == f for x in sub["uniform"]) for f in FAMILIES},
               "prediction": {name: metrics(v) for name, v in sub.items() if v},
               "prediction_by_family": {f: {name: metrics([x for x in v if x["family"] == f]) for name, v in sub.items()
                                            if any(x["family"] == f for x in v)} for f in FAMILIES},
               "comparisons": {f"{a} minus {b}": paired(sub[a], sub[b]) for a, b in comparisons},
               "betting": {}, "betting_differences": {}}
        for name, rows in bets.items():
            rows = [x for x in rows if lo <= x["date"] <= hi]
            b = ev.bet_summary(rows)
            played = [x for x in rows if x["bet"]]
            b["found_per_ticket"] = round(float(np.mean([x["found"] for x in played])), 2) if played else None
            known = [x["crowd"] for x in played if "crowd" in x]
            b["among_12_most_played"] = f"{sum(known)}/{len(known)}"
            out["betting"][name] = b
        for a, b in pairs:
            d = [(x["date"], x["ret"] - x["bet"] * ev.STAKE - (y["ret"] - y["bet"] * ev.STAKE))
                 for x, y in zip(bets[a], bets[b]) if lo <= x["date"] <= hi]
            out["betting_differences"][f"{a} minus {b}"] = {
                "net_eur_per_race": round(float(np.mean([v for _, v in d])), 3), "ci95": ev.block_bootstrap(d, np.mean)}
        rep["periods"][period] = out
    return rep, per_race


def metrics(evs):
    m = ev.summary_metrics(evs)
    return {"races": m["races"], "ll_top5_order": m["ll_top5_order"], "win_logloss": m["win_logloss"],
            "top5_brier": m["top5_brier"], "found": round(m["found"], 2), "ece_top5": m["top5_calibration"]["ece"],
            "ece_win": m["win_calibration"]["ece"]}


def paired(a, b):
    """Mean per-race log-likelihood difference a - b, with a week-block 95% interval."""
    ib = {x["id"]: x["ll_top5_order"] for x in b}
    d = [(x["date"], x["ll_top5_order"] - ib[x["id"]]) for x in a if x["id"] in ib]
    return {"mean": round(float(np.mean([v for _, v in d])), 4), "ci95": ev.block_bootstrap(d, np.mean), "races": len(d)}


# ---------------------------------------------------------------- output

def show(rep):
    d = rep["data"]
    print(f"\n{d['races_in_data']} races in data ({d['first']} -> {d['last']}), {d['races_tested']} tested from {d['tested_from']}")
    for period, p in rep["periods"].items():
        print(f"\n=== {period}: {p['races']} races ({p['from']} -> {p['to']}), {p['races_by_family']}")
        print(f"{'model':<26}{'ll top5':>9}{'win LL':>8}{'top5 Br':>9}{'found':>7}{'ECE t5':>8}" + "".join(f"{f[:5]:>9}" for f in FAMILIES))
        for name, m in p["prediction"].items():
            fam = "".join(f"{p['prediction_by_family'][f][name]['ll_top5_order']:>9.3f}" if name in p["prediction_by_family"][f]
                          else f"{'-':>9}" for f in FAMILIES)
            print(f"{name:<26}{m['ll_top5_order']:>9.3f}{m['win_logloss']:>8.3f}{m['top5_brier']:>9.4f}{m['found']:>7.2f}{m['ece_top5']:>8.3f}{fam}")
        for k, v in p["comparisons"].items():
            print(f"  {k:<50}{v['mean']:+.4f}  {v['ci95']}")
        print(f"{'ticket':<42}{'bets':>5}{'staked':>8}{'back':>8}{'ROI':>7}{'CI95':>20}{'-top1':>7}{'found':>6}{'crowd':>9}  hits")
        for name, b in p["betting"].items():
            nan = float("nan")
            print(f"{name:<42}{b['bets']:>5}{b['staked']:>8.0f}{b['returned']:>8.1f}{(b['roi'] if b['roi'] is not None else nan):>7.2f}"
                  f"{str(b.get('roi_ci95_week_blocks', '-')):>20}{(b['roi_without_top1'] if b['roi_without_top1'] is not None else nan):>7.2f}"
                  f"{(b['found_per_ticket'] if b['found_per_ticket'] is not None else nan):>6.2f}{b['among_12_most_played']:>9}  {b['hits']}")
        for k, v in p["betting_differences"].items():
            print(f"  {k:<60}{v['net_eur_per_race']:+.3f} € per race  {v['ci95']}")


def load_module(path):
    spec = importlib.util.spec_from_file_location("candidates_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--with", dest="files", action="append", default=[], help="candidate file (MODELS / STRATEGIES)")
    ap.add_argument("--tag", help="name of the result file when candidates are given")
    args = ap.parse_args()
    mods = [load_module(f) for f in args.files]
    models = [m for mod in mods for m in getattr(mod, "MODELS", [])]
    strategies = [s for mod in mods for s in getattr(mod, "STRATEGIES", [])]
    rep, per_race = run(models, strategies)
    rep["candidates"] = {"files": args.files, "models": [m["name"] for m in models], "strategies": [s["name"] for s in strategies]}
    stem = f"benchmark_{args.tag or 'candidates'}_{rep['data']['last']}" if mods else f"baseline_{rep['data']['last']}"
    ev.OUT.mkdir(parents=True, exist_ok=True)
    (ev.OUT / f"{stem}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    keys = list(dict.fromkeys(k for line in per_race for k in line))
    with (ev.OUT / f"{stem}_races.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_race)
    show(rep)
    print(f"\nWritten: research/results/{stem}.json and {stem}_races.csv")


if __name__ == "__main__":
    main()
