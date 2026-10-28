"""
Before/after evaluation of the Quinté+ prediction model and of paper-betting strategies.

Protocol (chronological, every runner of a race in the same split):
- refit every month on all earlier races (walk-forward);
- VALIDATION months 2026-02..2026-05: every choice (penalty, stage discounts, discipline
  interactions, feature groups, gradient boosting settings, betting threshold) is made here;
- TEST months 2026-06..2026-10: reported once with the frozen choices. Never used to choose.
Two prediction cutoffs: MORNING (06:00, no odds: the history has no morning odds) and T-30
(reference odds taken 30 minutes before the start). Odds taken after the start are never used.

Needs numpy + scipy (+ lightgbm for the boosting comparison), e.g. in a separate venv:
    python -m venv .venv && .venv/Scripts/pip install numpy scipy lightgbm
    .venv/Scripts/python research/evaluate.py
Writes research/results/evaluation.json and prints the tables. Paper betting only.
"""
import csv
import importlib.util
import itertools
import json
import math
import sys
import warnings
import zlib
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import backtest as bt  # noqa: E402
import features as ft  # noqa: E402
import plmodel as plm  # noqa: E402
from paths import DATA  # noqa: E402

OUT = HERE / "results"
VALID = ("2026-02", "2026-05")
TEST = ("2026-06", "2026-10")
STAKE = 2.0
SIMS = 20000
N_PAD = 20
L2_GRID = (2.0, 5.0, 10.0, 20.0, 40.0)
GROUPS = {
    "form": ["f_form_place", "f_form_top3", "f_form_win", "f_last_place", "f_form_failed", "f_dq_rate", "f_last_failed"],
    "freshness": ["f_days_since", "f_runs_60d", "f_long_break"],
    "suitability": ["f_going_n", "f_going_top3", "f_dist_n", "f_dist_top3", "f_track_n", "f_track_top3", "f_same_family_share"],
    "class_earnings": ["f_class", "f_log_gains_per_run", "f_log_gains_year", "f_career_win", "f_career_place"],
    "weight_handicap_draw": ["f_weight_rel", "f_handicap_rel", "f_dist_handicap", "f_draw_rel"],
    "jockey_trainer": ["f_jockey_top3", "f_jockey_rides", "f_trainer_top5", "f_driver_change", "f_avis"],
    "profile": ["f_age_rel", "f_female", "f_entire", "f_blinkers", "f_unshod4", "f_n_past"],
}
SIMPLE_FORM = ["f_form_place", "f_last_place", "f_form_top3", "f_career_win"]


# ---------------------------------------------------------------- data

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_races(rows, odds_ref):
    feats = [k for k in rows[0] if k.startswith("f_")]
    by = defaultdict(list)
    for r in rows:
        by[r["race_id"]].append(r)
    races = []
    for rid, field in sorted(by.items()):
        field.sort(key=lambda r: int(r["numPmu"]))
        placed = sorted((int(r["res_place"]), int(r["numPmu"]), i) for i, r in enumerate(field) if r["res_place"])
        if not placed:  # cancelled race, stakes refunded
            continue
        spec = field[0]["specialite"]
        races.append({
            "id": rid, "date": rid[:10], "month": rid[:7], "spec": spec,
            "family": "TROT" if spec.startswith("TROT") else ("OBSTACLE" if spec == "OBSTACLE" else "PLAT"),
            "nums": [int(r["numPmu"]) for r in field],
            "raw": np.array([[float(r[k]) if r[k] != "" else np.nan for k in feats] for r in field]),
            "order": [i for *_, i in placed[:5]],
            "top5": np.array([r["res_top5"] == "1" for r in field]),
            "odds": np.array([odds_ref.get((rid, r["numPmu"]), np.nan) for r in field]),
        })
    return races, feats


def load_versions():
    odds_ref = {(r["race_id"], r["numPmu"]): float(r["cote_reference"])
                for r in load_csv(DATA / "runners.csv") if r["cote_reference"]}
    runners, past = load_csv(DATA / "runners.csv"), load_csv(DATA / "past_runs.csv")
    code = DATA / "research" / "features_v1.py"  # the pre-fix feature code, kept to rebuild the "before" rows
    if code.exists():
        spec = importlib.util.spec_from_file_location("features_v1", code)
        v1code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v1code)
        v1rows = v1code.compute(runners, past)
    else:
        v1rows = load_csv(DATA / "research" / "features_v1.csv")
    v1, f1 = build_races(v1rows, odds_ref)
    v2, f2 = build_races(ft.compute(runners, past), odds_ref)
    return {"v1": (v1, f1), "v2": frozen_v2(v2, f2)}


def frozen_v2(races, feats):
    """The v2 features as frozen for the protocol: the trainer-opinion split and the outsider feature
    were added to production later (signal_eval.py adds them back), so rebuild the single -1/0/+1
    f_avis in place of f_avis_neg and drop f_avis_pos and f_outsider."""
    neg, pos = feats.index("f_avis_neg"), feats.index("f_avis_pos")
    keep = [i for i, c in enumerate(feats) if c not in ("f_avis_pos", "f_outsider")]
    for r in races:
        raw = r["raw"].copy()
        raw[:, neg] = raw[:, pos] - raw[:, neg]
        r["raw"] = raw[:, keep]
    return races, ["f_avis" if c == "f_avis_neg" else c for c in (feats[i] for i in keep)]


def design(races, cols, stats, trot=False):
    """Standardise with training stats; missing -> race average (no effect within a race)."""
    R = len(races)
    k = len(cols) * (2 if trot else 1)
    X = np.zeros((R, N_PAD, k))
    mask = np.zeros((R, N_PAD), bool)
    order = np.full((R, 5), -1)
    for i, r in enumerate(races):
        Z = (r["raw"][:, cols] - stats[0][cols]) / stats[1][cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-missing column -> 0
            col_mean = np.nan_to_num(np.nanmean(Z, axis=0))
        Z = np.where(np.isnan(Z), col_mean, Z)
        if trot:
            Z = np.concatenate([Z, Z * (r["family"] == "TROT")], axis=1)
        n = len(r["nums"])
        X[i, :n] = Z
        mask[i, :n] = True
        order[i, :len(r["order"])] = r["order"]
    return X, mask, order


def fit_stats(races):
    A = np.concatenate([r["raw"] for r in races])
    m = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0)
    m = np.where(np.isnan(m), 0.0, m)
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    return m, sd


def months_between(races, lo, hi):
    return sorted({r["month"] for r in races if lo <= r["month"] <= hi})


# ---------------------------------------------------------------- models (walk-forward)

def walk_forward(races, feats, cfg, months):
    """cfg: cols (feature names), l2, discount, trot. Returns {race_id: (s, lam)}."""
    cols = [feats.index(c) for c in cfg["cols"]]
    preds = {}
    for m in months:
        train = [r for r in races if r["month"] < m]
        test = [r for r in races if r["month"] == m]
        stats = fit_stats(train)
        Xtr, mtr, otr = design(train, cols, stats, cfg.get("trot", False))
        pen = np.full(Xtr.shape[2], cfg["l2"])
        if cfg.get("trot"):
            pen[len(cols):] *= 2.0
        model = plm.PL(pen, cfg.get("discount", False)).fit(Xtr, mtr, otr)
        Xte, mte, _ = design(test, cols, stats, cfg.get("trot", False))
        for i, r in enumerate(test):
            preds[r["id"]] = (model.strengths(Xte[i])[:len(r["nums"])], model.lam)
    return preds


def market_preds(races, months):
    """Raw market: strength = log(1 / reference odds), plain PL."""
    sel = set(months)
    return {r["id"]: (np.log(1 / r["odds"]), np.ones(5)) for r in races if r["month"] in sel}


def gbm_preds(races, feats, cols_names, months, params):
    """LightGBM lambdarank grouped by race; scores turned into PL strengths by a temperature
    fitted on the last two training months with a model that did not see them."""
    import lightgbm as lgb
    cols = [feats.index(c) for c in cols_names]

    def demeaned(raw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return raw - np.nanmean(raw, axis=0)

    def xy(rs):
        X = np.concatenate([demeaned(r["raw"][:, cols]) for r in rs])
        X = np.concatenate([X, np.concatenate([r["raw"][:, cols] for r in rs])], axis=1)
        rel = np.concatenate([[5 - r["order"].index(i) if i in r["order"] else 0 for i in range(len(r["nums"]))] for r in rs])
        return X, rel, [len(r["nums"]) for r in rs]

    def train(rs):
        X, y, g = xy(rs)
        p = {"objective": "lambdarank", "verbosity": -1, "num_leaves": params["num_leaves"],
             "min_data_in_leaf": params["min_data_in_leaf"], "learning_rate": params["learning_rate"], "seed": 1}
        return lgb.train(p, lgb.Dataset(X, y, group=g), num_boost_round=params["rounds"])

    preds = {}
    for m in months:
        tr = [r for r in races if r["month"] < m]
        inner_cut = sorted({r["month"] for r in tr})[-2]
        early, late = [r for r in tr if r["month"] < inner_cut], [r for r in tr if r["month"] >= inner_cut]
        g_early = train(early)
        sc = [g_early.predict(xy([r])[0]) for r in late]
        temps = np.linspace(0.1, 3.0, 59)
        lls = [sum(pl_loglik(t * s, np.ones(5), r["order"]) for s, r in zip(sc, late)) for t in temps]
        t = temps[int(np.argmax(lls))]
        g = train(tr)
        for r in (x for x in races if x["month"] == m):
            preds[r["id"]] = (t * g.predict(xy([r])[0]), np.ones(5))
    return preds


# ---------------------------------------------------------------- metrics

def pl_loglik(s, lam, order):
    left = np.ones(len(s), bool)
    ll = 0.0
    for j, i in enumerate(order):
        z = lam[j] * s[left]
        m = z.max()
        ll += lam[j] * s[i] - m - math.log(np.exp(z - m).sum())
        left[i] = False
    return ll


def evaluate_preds(races, preds, rng, sims=10000, keep_orders=False):
    """Per-race metrics plus arrays for calibration (and the sampled orders if asked)."""
    out = []
    for r in races:
        if r["id"] not in preds:
            continue
        s, lam = preds[r["id"]]
        n = len(s)
        pw = plm.win_probs(s, lam[0])
        orders = plm.sample_orders(s, lam, sims, rng)
        p5 = plm.top5_probs(orders, n)
        y5 = r["top5"].astype(float)
        win = r["order"][0]
        yw = np.zeros(n)
        yw[win] = 1
        pred5 = set(np.argsort(-p5)[:5])
        pc = np.clip(p5, 1e-4, 1 - 1e-4)
        out.append({
            "id": r["id"], "date": r["date"], "family": r["family"],
            "ll_top5_order": pl_loglik(s, lam, r["order"]),
            "win_logloss": -math.log(max(pw[win], 1e-12)),
            "win_brier": float(((pw - yw) ** 2).sum()),
            "top5_logloss": float(-(y5 * np.log(pc) + (1 - y5) * np.log(1 - pc)).mean()),
            "top5_brier": float(((p5 - y5) ** 2).mean()),
            "found": len(pred5 & set(np.flatnonzero(r["top5"]))),
            "pw": pw, "yw": yw, "p5": p5, "y5": y5, "orders": orders if keep_orders else None,
        })
    return out


def calibration(p, y, edges):
    p, y = np.concatenate(p), np.concatenate(y)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi)
        if sel.sum():
            rows.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": int(sel.sum()), "mean_pred": round(float(p[sel].mean()), 3),
                         "observed": round(float(y[sel].mean()), 3)})
            ece += sel.sum() / len(p) * abs(p[sel].mean() - y[sel].mean())
    return {"bins": rows, "ece": round(float(ece), 4)}


def week(d):
    y, w, _ = date.fromisoformat(d).isocalendar()
    return f"{y}-{w:02d}"


def block_bootstrap(values_by_race, stat, n_boot=4000, seed=7):
    """CI by resampling whole ISO weeks (consecutive races share conditions and the market)."""
    weeks = defaultdict(list)
    for d, v in values_by_race:
        weeks[week(d)].append(v)
    keys = list(weeks)
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        res.append(stat([v for k in pick for v in weeks[keys[k]]]))
    return [round(float(np.percentile(res, 2.5)), 4), round(float(np.percentile(res, 97.5)), 4)]


def summary_metrics(ev):
    k = len(ev)
    agg = {m: round(sum(x[m] for x in ev) / k, 4) for m in
           ("ll_top5_order", "win_logloss", "win_brier", "top5_logloss", "top5_brier", "found")}
    agg["races"] = k
    agg["win_calibration"] = calibration([x["pw"] for x in ev], [x["yw"] for x in ev], [0, .05, .1, .15, .2, .3, .5, 1.01])
    agg["top5_calibration"] = calibration([x["p5"] for x in ev], [x["y5"] for x in ev], [0, .1, .2, .3, .4, .5, .6, .8, 1.01])
    return agg


# ---------------------------------------------------------------- payouts and tickets

def payout_table():
    pay = bt.load_payouts()  # Ordre (no Tirelire), Désordre, Bonus 4sur5, Bonus 3, euros per 1 €
    return pay


def logq(r):
    q = 1 / r["odds"]
    return np.log(q / q.sum())


class PayoutModel:
    """log(dividend) ~ market popularity of the winning combination (sum of log implied win
    probabilities of its horses) + field size + discipline. Fitted on earlier races only;
    lognormal mean exp(mu + sigma^2/2). Its uncertainty is reported, not hidden."""
    CATS = {"ORDRE": 5, "DESORDRE": 5, "BONUS_4SUR5": 5, "BONUS_4": 4, "BONUS_3": 3}  # BONUS_4: older rule

    def __init__(self, races, payouts):
        self.coef, self.sigma, self.r2 = {}, {}, {}
        for cat, depth in self.CATS.items():
            X, y = [], []
            for r in races:
                vals = [e for p, c, e in payouts.get(r["id"], []) if p == "QUINTE_PLUS:" + cat and e > 0]
                if not vals or np.isnan(r["odds"]).any():
                    continue
                X.append(self.x(r, logq(r)[r["order"][:depth]].sum()))
                y.append(math.log(np.mean(vals)))
            if len(y) < 20:  # category not in force (or too rare) in the training period
                continue
            X, y = np.array(X), np.array(y)
            beta = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
            res = y - X @ beta
            self.coef[cat], self.sigma[cat] = beta, float(res.std())
            self.r2[cat] = float(1 - res.var() / y.var())

    @staticmethod
    def x(r, pop):
        return np.array([1.0, pop, len(r["nums"]), r["family"] == "TROT", r["family"] == "OBSTACLE"], float)

    def mean(self, cat, r, pop):
        """Expected euros per 1 € given the winning combination's popularity (array or scalar)."""
        b = self.coef[cat]
        base = b[0] + b[2] * len(r["nums"]) + b[3] * (r["family"] == "TROT") + b[4] * (r["family"] == "OBSTACLE")
        return np.exp(base + b[1] * np.asarray(pop) + self.sigma[cat] ** 2 / 2)


def race_rule(r, payouts):
    """Bonus rule in force for a race: the older "first four" Bonus 4 or the current Bonus 4sur5."""
    return "BONUS_4" if any(p == "QUINTE_PLUS:BONUS_4" for p, _, _ in payouts.get(r["id"], [])) else "BONUS_4SUR5"


def ticket_evs(r, s, lam, orders, pm, candidates, rule="BONUS_4SUR5"):
    """Expected net euros of each candidate ticket (ordered tuple of runner indices) for 2 €.
    Best category only: Ordre > Désordre > Bonus 4 or 4sur5 > Bonus 3."""
    lq = logq(r)
    n = len(s)
    rows = np.arange(len(orders))[:, None]
    T = np.zeros((len(candidates), n), int)
    for j, t in enumerate(candidates):
        T[j, list(t)] = 1

    def hits(depth):
        m = np.zeros((len(orders), n), int)
        m[rows, orders[:, :depth]] = 1
        return m @ T.T

    k5, k4, k3 = hits(5), hits(4), hits(3)
    if rule == "BONUS_4":  # the first four, any order
        b4 = (k4 == 4) & (k5 < 5)
        e_b4 = pm.mean("BONUS_4", r, lq[orders[:, :4]].sum(axis=1))[:, None]
    else:  # any 4 of the first five
        b4 = k5 == 4
        e_b4 = pm.mean("BONUS_4SUR5", r, lq[orders].sum(axis=1))[:, None]
    b3 = (k3 == 3) & ~b4 & (k5 < 5)
    ev_b45 = (b4 * e_b4).mean(axis=0)
    ev_b3 = (b3 * pm.mean("BONUS_3", r, lq[orders[:, :3]].sum(axis=1))[:, None]).mean(axis=0)
    perms = np.array([list(p) for t in candidates for p in itertools.permutations(t)])
    pp = plm.perm_probs(s, lam, perms).reshape(len(candidates), 120)
    out = []
    for j, t in enumerate(candidates):
        p_ord, p_set = pp[j, 0], pp[j].sum()  # the first permutation is the ticket's own order
        pop = lq[list(t)].sum()
        per_euro = p_ord * pm.mean("ORDRE", r, pop) + (p_set - p_ord) * pm.mean("DESORDRE", r, pop) + ev_b45[j] + ev_b3[j]
        out.append({"ticket": t, "ev_net": STAKE * (float(per_euro) - 1), "p_set": p_set, "p_b4": float(b4[:, j].mean())})
    return out


def settle(r, ticket_idx, payouts):
    nums = [r["nums"][i] for i in ticket_idx]
    rank, eur = bt.ticket_return(nums, payouts.get(r["id"], []))
    return rank, STAKE * eur


def bet_summary(rows):
    """rows: dicts with date, family, month, bet (bool), ret (euros returned), rank."""
    bets = [x for x in rows if x["bet"]]
    staked = STAKE * len(bets)
    ret = sum(x["ret"] for x in bets)
    path, peak, mdd = 0.0, 0.0, 0.0
    for x in sorted(bets, key=lambda x: x["date"]):
        path += x["ret"] - STAKE
        peak = max(peak, path)
        mdd = min(mdd, path - peak)
    gains = sorted((x["ret"] for x in bets), reverse=True)
    roi = lambda xs: (sum(xs) - STAKE * len(xs)) / (STAKE * len(xs)) if xs else float("nan")
    out = {
        "races": len(rows), "bets": len(bets), "staked": staked, "returned": round(ret, 2), "net": round(ret - staked, 2),
        "roi": round(roi([x["ret"] for x in bets]), 3) if bets else None,
        "hits": dict(Counter(x["rank"] for x in bets if x["rank"])),
        "max_drawdown": round(mdd, 2),
        "roi_without_top1": round(roi(gains[1:]), 3) if len(gains) > 1 else None,
        "roi_without_top3": round(roi(gains[3:]), 3) if len(gains) > 3 else None,
        "largest_returns": [round(g, 2) for g in gains[:3]],
    }
    if len(bets) >= 10:
        out["roi_ci95_week_blocks"] = block_bootstrap([(x["date"], x["ret"]) for x in bets], roi)
    out["by_family"] = {f: {"bets": len(b), "roi": round(roi([x["ret"] for x in b]), 3)}
                        for f in ("PLAT", "TROT", "OBSTACLE") if (b := [x for x in bets if x["family"] == f])}
    out["by_month"] = {m: round(sum(x["ret"] for x in bets if x["month"] == m) - STAKE * sum(x["month"] == m for x in bets), 2)
                       for m in sorted({x["month"] for x in bets})}
    return out


# ---------------------------------------------------------------- experiment

def select(races, feats, base_cols, months_valid, label):
    """Grid on validation: penalty x stage discount x trot interactions, by mean PL log-likelihood."""
    best, table = None, []
    for l2, disc, trot in itertools.product(L2_GRID, (False, True), (False, True)):
        cfg = {"cols": base_cols, "l2": l2, "discount": disc, "trot": trot}
        preds = walk_forward(races, feats, cfg, months_valid)
        ll = np.mean([pl_loglik(*preds[r["id"]], r["order"]) for r in races if r["id"] in preds])
        table.append({"l2": l2, "discount": disc, "trot": trot, "valid_ll": round(float(ll), 4)})
        if best is None or ll > best[0]:
            best = (ll, cfg)
    print(f"  {label}: best on validation l2={best[1]['l2']}, discount={best[1]['discount']}, "
          f"trot={best[1]['trot']}, ll={best[0]:.4f}")
    return best[1], table


def ablate(races, feats, cfg, months_valid):
    """Drop each feature group in turn (validation only); keep the groups whose removal hurts."""
    preds = walk_forward(races, feats, cfg, months_valid)
    base = np.mean([pl_loglik(*preds[r["id"]], r["order"]) for r in races if r["id"] in preds])
    rep, keep = {}, list(cfg["cols"])
    for g, cols in GROUPS.items():
        trial = {**cfg, "cols": [c for c in cfg["cols"] if c not in cols]}
        preds = walk_forward(races, feats, trial, months_valid)
        ll = np.mean([pl_loglik(*preds[r["id"]], r["order"]) for r in races if r["id"] in preds])
        rep[g] = round(float(ll - base), 4)  # > 0: the model is better WITHOUT this group
        if ll > base:
            keep = [c for c in keep if c not in cols]
    return rep, keep


def main():
    rng = np.random.default_rng(1)
    data = load_versions()
    payouts = payout_table()
    races2, f2 = data["v2"]
    races1, f1 = data["v1"]
    mv = months_between(races2, *VALID)
    mt = months_between(races2, *TEST)
    both = mv + mt
    no_mkt2 = [c for c in f2 if c != "f_mkt_logp"]
    report = {"protocol": {"validation_months": mv, "test_months": mt,
                           "validation_races": sum(r["month"] in mv for r in races2),
                           "test_races": sum(r["month"] in mt for r in races2)}}
    print(report["protocol"])

    # --- choices on validation only
    print("Selecting on validation...")
    cfg_sans, grid_sans = select(races2, f2, no_mkt2, mv, "v2 morning (no odds)")
    abl_sans, keep_sans = ablate(races2, f2, cfg_sans, mv)
    cfg_sans_final = {**cfg_sans, "cols": keep_sans}
    cfg_avec, grid_avec = select(races2, f2, keep_sans + ["f_mkt_logp"], mv, "v2 T-30 (with odds)")
    gbm_choice = None
    try:
        import lightgbm  # noqa: F401
        best = None
        for params in ({"rounds": 100, "num_leaves": 7, "min_data_in_leaf": 40, "learning_rate": 0.05},
                       {"rounds": 300, "num_leaves": 7, "min_data_in_leaf": 40, "learning_rate": 0.03},
                       {"rounds": 200, "num_leaves": 15, "min_data_in_leaf": 60, "learning_rate": 0.05}):
            p = gbm_preds(races2, f2, no_mkt2, mv, params)
            ll = np.mean([pl_loglik(*p[r["id"]], r["order"]) for r in races2 if r["id"] in p])
            if best is None or ll > best[0]:
                best = (ll, params)
        gbm_choice = best[1]
        print(f"  LightGBM: best on validation {gbm_choice} ll={best[0]:.4f}")
    except ImportError:
        print("  LightGBM not installed: boosting comparison skipped")
    report["choices"] = {"morning": {k: v for k, v in cfg_sans_final.items() if k != "cols"},
                         "morning_features_dropped": [c for c in no_mkt2 if c not in keep_sans],
                         "ablation_valid_ll_gain_when_dropped": abl_sans, "t30": {k: v for k, v in cfg_avec.items() if k != "cols"},
                         "grid_morning": grid_sans, "grid_t30": grid_avec, "lightgbm": gbm_choice}

    # --- all models on validation + test (walk-forward)
    print("Walk-forward predictions...")
    models = {
        "uniform": {r["id"]: (np.zeros(len(r["nums"])), np.ones(5)) for r in races2 if r["month"] in both},
        "market_raw (T-30)": market_preds(races2, both),
        "market_fitted (T-30)": walk_forward(races2, f2, {"cols": ["f_mkt_logp"], "l2": 0.01, "discount": True}, both),
        "simple_form (morning)": walk_forward(races2, f2, {"cols": SIMPLE_FORM, "l2": 5.0}, both),
        "v1_existing_no_odds (morning)": walk_forward(races1, f1, {"cols": [c for c in f1 if c != "f_mkt_logp"], "l2": 5.0}, both),
        "v1_existing_with_odds (T-30)": walk_forward(races1, f1, {"cols": f1, "l2": 5.0}, both),
        "v2_no_odds (morning)": walk_forward(races2, f2, cfg_sans_final, both),
        "v2_with_odds (T-30)": walk_forward(races2, f2, cfg_avec, both),
    }
    if gbm_choice:
        models["lightgbm_no_odds (morning)"] = gbm_preds(races2, f2, no_mkt2, both, gbm_choice)
    evals = {}
    for name, preds in models.items():
        rs = races1 if name.startswith("v1") else races2
        evals[name] = evaluate_preds(rs, preds, rng, sims=SIMS if name == "v2_with_odds (T-30)" else 10000,
                                     keep_orders=name == "v2_with_odds (T-30)")
    by_id2 = {r["id"]: r for r in races2}
    assert all(r["order"] == by_id2[r["id"]]["order"] and r["nums"] == by_id2[r["id"]]["nums"] for r in races1),         "v1 and v2 tables must describe the same races and runners"
    report["prediction"] = {}
    for period, ms in (("validation", mv), ("test", mt)):
        report["prediction"][period] = {}
        for name, ev in evals.items():
            sub = [x for x in ev if x["id"][:7] in ms]
            report["prediction"][period][name] = summary_metrics(sub)
    # market vs model: paired difference in log-likelihood on the test months, week-block CI
    ids = {x["id"]: x for x in evals["market_fitted (T-30)"]}
    report["beyond_market_test"] = {}
    for name in ("v2_with_odds (T-30)", "v1_existing_with_odds (T-30)", "v2_no_odds (morning)"):
        diffs = [(x["date"], x["ll_top5_order"] - ids[x["id"]]["ll_top5_order"]) for x in evals[name] if x["id"][:7] in mt]
        report["beyond_market_test"][name] = {"mean_ll_gain_vs_fitted_market": round(float(np.mean([d for _, d in diffs])), 4),
                                              "ci95": block_bootstrap(diffs, np.mean)}
    # per discipline, test
    report["prediction_test_by_family"] = {
        name: {f: round(float(np.mean([x["ll_top5_order"] for x in ev if x["id"][:7] in mt and x["family"] == f])), 4)
               for f in ("PLAT", "TROT", "OBSTACLE")} for name, ev in evals.items()}

    # --- betting (paper): settle with official dividends
    print("Paper betting...")
    ev_by = {name: {x["id"]: x for x in ev} for name, ev in evals.items()}
    rows = defaultdict(list)
    ev_check, own_ev, pms = [], [], {}
    for period, ms in (("validation", mv), ("test", mt)):
        for r in (x for x in races2 if x["month"] in ms):
            base = {"date": r["date"], "family": r["family"], "month": r["month"], "period": period}

            def add(name, idx, bet=True, extra=None):
                rank, ret = settle(r, idx, payouts) if bet else (None, 0.0)
                rows[name].append({**base, "bet": bet, "ret": ret, "rank": rank, **(extra or {})})

            n = len(r["nums"])
            add("favourites (T-30)", list(np.argsort(r["odds"])[:5]))
            add("random", list(np.random.default_rng(zlib.crc32(r["id"].encode())).permutation(n)[:5]))
            for tag, name in (("v1 no odds", "v1_existing_no_odds (morning)"), ("v1 with odds", "v1_existing_with_odds (T-30)"),
                              ("v2 no odds", "v2_no_odds (morning)"), ("v2 with odds", "v2_with_odds (T-30)")):
                x = ev_by[name][r["id"]]
                add(f"top5 {tag}", list(np.argsort(-x["p5"])[:5]))
            mk = ev_by["market_raw (T-30)"][r["id"]]["p5"]
            for tag, name in (("v1", "v1_existing_no_odds (morning)"), ("v2", "v2_no_odds (morning)")):
                pf = ev_by[name][r["id"]]["p5"]
                short = list(np.argsort(-pf)[:8])
                value = sorted(sorted(short, key=lambda i: -pf[i] / max(mk[i], 1e-6))[:5], key=lambda i: -pf[i])
                add(f"bon rapport {tag} (T-30)", value)
            # expected-value tickets with a no-bet option (T-30: needs the odds for the payout model)
            if r["month"] not in pms:
                pms[r["month"]] = PayoutModel([x for x in races2 if x["month"] < r["month"]], payouts)
            pm = pms[r["month"]]
            s, lam = models["v2_with_odds (T-30)"][r["id"]]
            orders = ev_by["v2_with_odds (T-30)"][r["id"]]["orders"]
            top8 = list(np.argsort(-s)[:8])
            cands = [tuple(sorted(c, key=lambda i: -s[i])) for c in itertools.combinations(top8, 5)]
            evs = sorted(ticket_evs(r, s, lam, orders, pm, cands), key=lambda e: -e["ev_net"])
            best = evs[0]
            rank, ret = settle(r, list(best["ticket"]), payouts)
            ev_check.append({**base, "ev_net": best["ev_net"], "ret": ret, "rank": rank})
            # the model's own most likely ticket, for comparing predicted and realised returns
            own = ticket_evs(r, s, lam, orders, pm, [tuple(np.argsort(-s)[:5])])[0]
            rk2, rt2 = settle(r, list(own["ticket"]), payouts)
            own_ev.append({**base, "ret": rt2, "ev": own["ev_net"]})
    thresholds = (0.0, 0.5, 1.0, 2.0)
    valid_ev = [x for x in ev_check if x["period"] == "validation"]
    tau_table = {}
    for tau in thresholds:
        b = [x for x in valid_ev if x["ev_net"] > tau]
        tau_table[tau] = {"bets": len(b), "net": round(sum(x["ret"] - STAKE for x in b), 2)}
    tau = max(thresholds, key=lambda t: (tau_table[t]["net"], -t)) if any(v["bets"] for v in tau_table.values()) else 0.0
    report["ev_threshold_validation"] = {"table": tau_table, "chosen": tau}
    for x in ev_check:
        rows[f"best expected value, bet only if EV > {tau} € (T-30)"].append({**x, "bet": x["ev_net"] > tau})
    rows_ev0 = [{**x, "bet": x["ev_net"] > 0} for x in ev_check]
    report["betting"] = {}
    for period in ("validation", "test"):
        report["betting"][period] = {name: bet_summary([x for x in rs if x["period"] == period]) for name, rs in rows.items()}
        report["betting"][period]["best expected value, EV > 0 (T-30, diagnostic)"] = bet_summary(
            [x for x in rows_ev0 if x["period"] == period])
    pred = own_ev
    report["ev_calibration"] = {p: {"predicted_net_per_ticket": round(float(np.mean([x["ev"] for x in pred if x["period"] == p])), 3),
                                    "realised_net_per_ticket": round(float(np.mean([x["ret"] - STAKE for x in pred if x["period"] == p])), 3)}
                                for p in ("validation", "test")}
    report["ev_max_by_race_test"] = {"share_of_races_with_positive_best_ev": round(
        float(np.mean([x["ev_net"] > 0 for x in ev_check if x["period"] == "test"])), 3)}
    pm_all = PayoutModel([x for x in races2 if x["month"] < TEST[0]], payouts)
    report["payout_model_fit_before_test"] = {c: {"r2": round(pm_all.r2[c], 3), "residual_sd_log": round(pm_all.sigma[c], 3),
                                                  "popularity_coef": round(float(pm_all.coef[c][1]), 3)} for c in pm_all.CATS}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print_report(report)


def print_report(rep):
    for period in ("validation", "test"):
        print(f"\n=== Prediction, {period} ({rep['protocol'][period + '_races']} races) ===")
        print(f"{'model':<34}{'ll top5':>9}{'win LL':>8}{'win Br':>8}{'top5 LL':>9}{'top5 Br':>9}{'found':>7}{'ECE win':>8}{'ECE t5':>8}")
        for name, m in rep["prediction"][period].items():
            print(f"{name:<34}{m['ll_top5_order']:>9.3f}{m['win_logloss']:>8.3f}{m['win_brier']:>8.4f}{m['top5_logloss']:>9.4f}"
                  f"{m['top5_brier']:>9.4f}{m['found']:>7.2f}{m['win_calibration']['ece']:>8.3f}{m['top5_calibration']['ece']:>8.3f}")
    print("\nLog-likelihood gain over the fitted market, test, week-block 95% CI:")
    for k, v in rep["beyond_market_test"].items():
        print(f"  {k:<34}{v['mean_ll_gain_vs_fitted_market']:+.4f}  {v['ci95']}")
    for period in ("validation", "test"):
        print(f"\n=== Paper betting, {period} (2 € tickets, official dividends) ===")
        print(f"{'strategy':<52}{'bets':>5}{'staked':>8}{'back':>9}{'ROI':>7}{'CI95':>18}{'ROI-top1':>9}{'maxDD':>8}  hits")
        for name, b in rep["betting"][period].items():
            ci = b.get("roi_ci95_week_blocks")
            print(f"{name:<52}{b['bets']:>5}{b['staked']:>8.0f}{b['returned']:>9.1f}{(b['roi'] if b['roi'] is not None else float('nan')):>7.2f}"
                  f"{(str(ci) if ci else '-'):>18}{(b['roi_without_top1'] if b['roi_without_top1'] is not None else float('nan')):>9.2f}"
                  f"{b['max_drawdown']:>8.0f}  {b['hits']}")
    print("\nEV threshold chosen on validation:", rep["ev_threshold_validation"])
    print("Predicted vs realised net per ticket (model's own top 5, T-30):", rep["ev_calibration"])
    print("Payout model (fit on races before the test):", rep["payout_model_fit_before_test"])
    print("Choices:", {k: v for k, v in rep["choices"].items() if not k.startswith("grid")})


if __name__ == "__main__":
    main()
