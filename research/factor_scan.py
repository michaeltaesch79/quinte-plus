"""
Where is the market wrong? For groups of runners (odds level, odds movement, weather, ground, course,
draw, equipment, trainer opinion, freshness, last-race comment), the actual top-5 and win rates vs what
the reference odds implied. Explore half: races before 2025-05-01; confirm half: from 2025-05-01.
Exploratory (many groups, approximate standard errors): a pattern counts only if it holds in both halves.

    <venv>/python research/factor_scan.py            # baseline: market fitted with stage discounts
    <venv>/python research/factor_scan.py --raw      # baseline: raw odds (overrates favourites' places)
    <venv>/python research/factor_scan.py --model    # baseline: the model with odds, before signal_eval.py
"""
import csv, glob, json, re, sys
from collections import defaultdict
import numpy as np

import evaluate as ev
import plmodel as plm
from paths import DATA

rd = lambda n: list(csv.DictReader(open(DATA / n, encoding="utf-8")))
runners = {(r["race_id"], r["numPmu"]): r for r in rd("runners.csv")}
feats = rd("features.csv")
past = defaultdict(list)
for p in rd("past_runs.csv"):
    past[(p["race_id"], p["numPmu"])].append(p)

# odds trend at T-30 and comments of every downloaded race (for "what happened last time")
trend, comments = {}, defaultdict(list)
for f in glob.glob(str(DATA / "raw" / "participants" / "*" / "R*.json")):
    iso, rc = f.replace("\\", "/").split("/")[-2:]
    try:
        d = json.load(open(f, encoding="utf-8"))
    except ValueError:
        continue
    for p in d.get("participants", []):
        ref = p.get("dernierRapportReference") or {}
        trend[(f"{iso}_{rc[:-5]}", str(p["numPmu"]))] = ref.get("nombreIndicateurTendance")
        t = (p.get("commentaireApresCourse") or {}).get("texte")
        if t:
            comments[p["nom"]].append((iso, t.lower()))
TROUBLE = re.compile(r"g[êe]n[ée]|bouscul|enferm|manqu[ée] de place|ouverture|barr[ée]|heurt|coinc|serr[ée]|dans le dos|sans espace|n'a pu|pas pu")
FAULT = re.compile(r"galop|fautif|faute|s'est mis")

by_race = defaultdict(list)
for f in feats:
    by_race[f["race_id"]].append(f)
rng = np.random.default_rng(0)
RAW = "--raw" in sys.argv
if not RAW:  # market fitted with stage discounts, walk-forward (corrects the favourite/top-5 distortion of raw odds)
    races2, f2 = ev.load_versions()["v2"]
    cfg = {"cols": f2, "l2": 10.0, "discount": True} if "--model" in sys.argv else {"cols": ["f_mkt_logp"], "l2": 0.01, "discount": True}
    mkt = ev.walk_forward(races2, f2, cfg,
                          sorted({r["month"] for r in races2})[1:])
    nums_of = {r["id"]: list(r["nums"]) for r in races2}
recs = []
for rid, fs in by_race.items():
    if any(not f["cote_reference"] for f in fs) or any(f["res_top5"] == "" for f in fs):
        continue
    odds = np.array([float(f["cote_reference"]) for f in fs])
    if RAW:
        s, lam = np.log(1 / odds), np.ones(5)
    else:
        if rid not in mkt:
            continue
        sm, lam = mkt[rid]
        pos = {int(n): j for j, n in enumerate(nums_of[rid])}
        s = np.array([sm[pos[int(f["numPmu"])]] for f in fs])
    pw = plm.win_probs(s, lam[0])
    p5 = plm.top5_probs(plm.sample_orders(s, lam, 20000, rng), len(fs))
    for i, f in enumerate(fs):
        r = runners[(rid, f["numPmu"])]
        x = {k: (float(v) if v not in ("", "nan") else np.nan) for k, v in f.items() if k.startswith("f_")}
        prev = sorted(past[(rid, f["numPmu"])], key=lambda p: p["past_date"])
        last_blink = prev[-1]["oeillere"] if prev else ""
        com = [t for d, t in comments.get(f["nom"], []) if d < f["date"]]
        last_com = max(((d, t) for d, t in comments.get(f["nom"], []) if d < f["date"]), default=None)
        recs.append(dict(rid=rid, half="explore" if f["date"] < "2025-05-01" else "confirm", spec=f["specialite"],
                         odds=odds[i], pw=pw[i], p5=p5[i], top5=int(f["res_top5"]), win=int(f["res_place"] == "1"),
                         meteo=r["meteo"].lower(), temp=float(r["temperature"] or "nan"), wind=float(r["forceVent"] or "nan"),
                         terrain=r["terrain"].lower(), trend=trend.get((rid, f["numPmu"])), x=x,
                         blink_now=r["oeilleres"] != "SANS_OEILLERES", blink_before=bool(last_blink) and last_blink != "SANS_OEILLERES",
                         has_prev=bool(prev), last_com=last_com[1] if last_com else None,
                         last_com_days=None if not last_com else (np.datetime64(f["date"]) - np.datetime64(last_com[0])).astype(int)))

def row(name, sel):
    out = [f"{name:<44}"]
    for h in ("explore", "confirm"):
        g = [r for r in recs if r["half"] == h and sel(r)]
        if len(g) < 40:
            out.append(f"{'n<40':>34}")
            continue
        a, e = np.mean([r["top5"] for r in g]), np.mean([r["p5"] for r in g])
        se = np.sqrt(sum(r["p5"] * (1 - r["p5"]) for r in g)) / len(g)
        aw, ew = np.mean([r["win"] for r in g]), np.mean([r["pw"] for r in g])
        out.append(f"{len(g):>6} top5 {a:.3f} vs {e:.3f} ({(a - e) / se:+.1f}σ) win {aw:.3f}/{ew:.3f}")
    print("  ".join(out))

X = lambda k: (lambda r: r["x"].get(k, np.nan))
print(f"{len(recs)} runners, {len({r['rid'] for r in recs})} races. Columns: actual vs odds-implied; σ = deviation in standard errors\n")
print("== Odds level (favourite-longshot bias)")
for lo, hi in ((0, 4), (4, 7), (7, 12), (12, 20), (20, 35), (35, 60), (60, 999)):
    row(f"odds {lo}-{hi}", lambda r, lo=lo, hi=hi: lo <= r["odds"] < hi)
print("== Odds movement before T-30 (negative = odds shortening)")
for lo, hi in ((-999, -15), (-15, -5), (-5, 5), (5, 15), (15, 999)):
    row(f"trend {lo}..{hi}%", lambda r, lo=lo, hi=hi: r["trend"] is not None and lo <= r["trend"] < hi)
print("== Weather")
row("rain (pluie/averses/orage)", lambda r: bool(re.search("pluie|averse|orage", r["meteo"])))
row("sun", lambda r: bool(re.search("soleil|ensoleill", r["meteo"])))
row("cold (< 8°C)", lambda r: r["temp"] < 8)
row("hot (>= 25°C)", lambda r: r["temp"] >= 25)
row("windy (>= 25 km/h)", lambda r: r["wind"] >= 25)
print("== Ground, horse's record on today's going (flat/jumps)")
heavy = lambda r: bool(re.search("lourd|tr[èe]s souple|collant", r["terrain"]))
row("heavy ground, horse never ran on this going", lambda r: heavy(r) and X("f_going_n")(r) == 0)
row("heavy ground, top3 rate on going >= 50%", lambda r: heavy(r) and X("f_going_n")(r) >= 2 and X("f_going_top3")(r) >= .5)
row("good ground, top3 rate on going >= 50%", lambda r: not heavy(r) and r["terrain"] and X("f_going_n")(r) >= 2 and X("f_going_top3")(r) >= .5)
print("== Course and distance specialists")
row("ran here >= 2x, top3 rate >= 50%", lambda r: X("f_track_n")(r) >= 2 and X("f_track_top3")(r) >= .5)
row("never ran on this course", lambda r: X("f_track_n")(r) == 0)
row("distance: top3 rate >= 50% (n>=2)", lambda r: X("f_dist_n")(r) >= 2 and X("f_dist_top3")(r) >= .5)
print("== Draw (flat)")
for lo, hi in ((0, .25), (.25, .75), (.75, 1.01)):
    row(f"flat, relative draw {lo}-{hi}", lambda r, lo=lo, hi=hi: r["spec"] == "PLAT" and lo <= X("f_draw_rel")(r) < hi)
print("== Equipment, signals")
row("blinkers first time", lambda r: r["blink_now"] and r["has_prev"] and not r["blink_before"])
row("blinkers removed", lambda r: not r["blink_now"] and r["blink_before"])
row("trot, barefoot all four", lambda r: X("f_unshod4")(r) == 1)
row("trainer opinion positive", lambda r: X("f_avis_pos")(r) == 1)
row("trainer opinion negative", lambda r: X("f_avis_neg")(r) == 1)
row("jockey/driver change", lambda r: X("f_driver_change")(r) == 1)
print("== Freshness")
row("back after > 90 days", lambda r: X("f_days_since")(r) > 90)
row("ran within 14 days", lambda r: X("f_days_since")(r) <= 14)
row("3+ runs in 60 days", lambda r: X("f_runs_60d")(r) >= 3)
print("== Last race comment (what happened), only where the previous race was downloaded")
recent = lambda r: r["last_com"] is not None and r["last_com_days"] <= 60
row("has a last-race comment (<= 60 days)", recent)
row("  trouble in running last time", lambda r: recent(r) and bool(TROUBLE.search(r["last_com"])))
row("  broke / fault last time (trot)", lambda r: recent(r) and bool(FAULT.search(r["last_com"])))
row("  no trouble mentioned", lambda r: recent(r) and not TROUBLE.search(r["last_com"]) and not FAULT.search(r["last_com"]))
