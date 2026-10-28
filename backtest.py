"""
Phase 6b: fit the ranking model and backtest the Quinté strategies, walk-forward.

Model: Plackett-Luce on the first 5 finishers (softmax within each race, applied 5 times),
linear in the f_* features of data/features.csv, L2-penalised, plain Python.
Test: for each month from TEST_FROM on, refit on every earlier race, then predict that month.
Each strategy plays one Quinté+ ticket of 5 horses at 2 € per race, paid with the real
payouts of data/payouts.csv (best matching rank: Ordre, Désordre, Bonus 4sur5, Bonus 3).
Writes data/backtest_summary.json, which daily.py copies into today.json.

Usage (Python 3.9+, no extra packages):
    python backtest.py
"""
import csv
import json
import math
import random
from collections import defaultdict

from paths import DATA

TEST_FROM = "2026-03"   # first test month (YYYY-MM); data starts 2025-10-27
L2 = 5.0                # penalty on the weights, fixed in advance (not tuned on the test races)
ITERS = 300
SIMS = 4000             # Monte Carlo draws to turn strengths into P(top 5)
STAKE = 2.0
BET_RANKS = ("QUINTE_PLUS:ORDRE", "QUINTE_PLUS:DESORDRE", "QUINTE_PLUS:BONUS_4SUR5", "QUINTE_PLUS:BONUS_3")


def load_races(rows=None):
    """Races with a result, from feature rows (default: data/features.csv). keep_unfinished keeps today's race."""
    return _races(rows, keep_unfinished=False)


def _races(rows, keep_unfinished):
    if rows is None:
        with (DATA / "features.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    feats = [k for k in rows[0] if k.startswith("f_")]
    races = defaultdict(list)
    for r in rows:
        races[r["race_id"]].append(r)
    out = []
    for race_id, field in sorted(races.items()):
        field.sort(key=lambda r: int(r["numPmu"]))
        placed = sorted((int(r["res_place"]), i) for i, r in enumerate(field) if r["res_place"])
        if not placed and not keep_unfinished:  # cancelled race, stakes refunded
            continue
        out.append({
            "id": race_id, "month": race_id[:7], "nums": [int(r["numPmu"]) for r in field],
            "type": {"TROT_ATTELE": "Trot", "TROT_MONTE": "Trot"}.get(field[0]["specialite"], field[0]["specialite"].title()),
            "names": [r["nom"] for r in field], "odds": [float(r["cote_reference"] or 999) for r in field],
            "has_odds": all(r["cote_reference"] for r in field),
            "raw": [[float(r[k]) if r[k] != "" else None for k in feats] for r in field],
            "order": [i for _, i in placed[:5]],
        })
    return out, feats


def design(races, cols, stats):
    """Standardise with train stats; a missing value becomes the race average (no effect in a softmax)."""
    for race in races:
        X = []
        for row in race["raw"]:
            X.append([None if row[c] is None else (row[c] - stats[c][0]) / stats[c][1] for c in cols])
        for j in range(len(cols)):
            vals = [x[j] for x in X if x[j] is not None]
            fill = sum(vals) / len(vals) if vals else 0.0
            for x in X:
                if x[j] is None:
                    x[j] = fill
        race["X"] = X


def fit_stats(races, n_feats):
    stats = []
    for c in range(n_feats):
        v = [row[c] for race in races for row in race["raw"] if row[c] is not None]
        m = sum(v) / len(v) if v else 0.0
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) if v else 1.0
        stats.append((m, sd or 1.0))
    return stats


def scores(X, w):
    return [sum(a * b for a, b in zip(x, w)) for x in X]


def pl_loglik(s, order, lam=None):
    """Log-probability of the observed top-5 order under Plackett-Luce with strengths s
    (stage j uses lam[j] * s when stage discounts are given)."""
    left, ll = set(range(len(s))), 0.0
    for st, i in enumerate(order):
        lj = lam[st] if lam else 1.0
        mx = max(lj * s[j] for j in left)
        ll += lj * s[i] - mx - math.log(sum(math.exp(lj * s[j] - mx) for j in left))
        left.discard(i)
    return ll


def fit(races, k):
    """Adam on the penalised Plackett-Luce log-likelihood of the top-5 order."""
    return fit_pl(races, k)[0]


def fit_pl(races, k, l2=L2, discount=False, iters=ITERS):
    """Adam on the penalised Plackett-Luce log-likelihood of the top-5 order -> (w, lam).
    With discount, stages 2-5 get their own factor lam_j (strengths lam_j * s, lam_1 = 1): later
    places are less predictable than the win. log lam_j has a weak pull towards 0."""
    n_par = k + (4 if discount else 0)
    theta, m, v = [0.0] * n_par, [0.0] * n_par, [0.0] * n_par
    lr, b1, b2 = 0.05, 0.9, 0.999
    for t in range(1, iters + 1):
        w = theta[:k]
        lam = [1.0] + ([math.exp(x) for x in theta[k:]] if discount else [1.0] * 4)
        g = [-l2 * wi for wi in w] + [-x for x in theta[k:]]
        for race in races:
            X, s = race["X"], scores(race["X"], w)
            coef = [0.0] * len(s)
            left = set(range(len(s)))
            for st, i in enumerate(race["order"]):
                lj = lam[st]
                mx = max(lj * s[j] for j in left)
                e = {j: math.exp(lj * s[j] - mx) for j in left}
                z = sum(e.values())
                if discount and st:
                    g[k + st - 1] += lj * (s[i] - sum(ej * s[j] for j, ej in e.items()) / z)
                for j, ej in e.items():
                    coef[j] -= lj * ej / z
                coef[i] += lj
                left.discard(i)
            for j, c in enumerate(coef):
                if c:
                    xj = X[j]
                    for f in range(k):
                        g[f] += c * xj[f]
        for f in range(n_par):  # gradient ascent with Adam
            m[f] = b1 * m[f] + (1 - b1) * g[f]
            v[f] = b2 * v[f] + (1 - b2) * g[f] ** 2
            theta[f] += lr * (m[f] / (1 - b1 ** t)) / (math.sqrt(v[f] / (1 - b2 ** t)) + 1e-8)
    return theta[:k], [1.0] + ([math.exp(x) for x in theta[k:]] if discount else [1.0] * 4)


def p_top5(s, rng, lam=None):
    """P(finish in the first 5) for each runner, by sampling Plackett-Luce with the Gumbel trick
    (one draw per stage when stage discounts are given)."""
    hits = [0] * len(s)
    for _ in range(SIMS):
        if lam is None:
            noisy = [x - math.log(-math.log(rng.random() or 1e-12)) for x in s]
            picked = sorted(range(len(s)), key=noisy.__getitem__, reverse=True)[:5]
        else:
            picked, left = [], set(range(len(s)))
            for lj in lam[:min(5, len(s))]:
                i = max(left, key=lambda j: lj * s[j] - math.log(-math.log(rng.random() or 1e-12)))
                picked.append(i)
                left.discard(i)
        for i in picked:
            hits[i] += 1
    return [h / SIMS for h in hits]


def load_payouts():
    pay = defaultdict(list)
    with (DATA / "payouts.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["pari"] in BET_RANKS:
                pay[r["race_id"]].append((r["pari"], [int(x) for x in r["combinaison"].split("-")],
                                          float(r["euros_pour_1_euro"])))
    return pay


def ticket_return(ticket, rows):
    """Best matching rank for an ordered ticket of 5 horse numbers -> (rank, euros per 1 €)."""
    best = (None, 0.0)
    for pari, comb, eur in rows:
        ok = {"QUINTE_PLUS:ORDRE": ticket == comb,
              "QUINTE_PLUS:DESORDRE": set(ticket) == set(comb),
              "QUINTE_PLUS:BONUS_4SUR5": set(comb) <= set(ticket),
              "QUINTE_PLUS:BONUS_3": set(comb) <= set(ticket)}[pari]
        if ok and eur > best[1]:
            best = (pari.split(":")[1], eur)
    return best


def main():
    races, feats = load_races()
    payouts = load_payouts()
    mkt = feats.index("f_mkt_logp")
    sets = {"sans_cote": [i for i in range(len(feats)) if i != mkt], "avec_cote": list(range(len(feats)))}
    rng = random.Random(1)
    months = sorted({r["month"] for r in races if r["month"] >= TEST_FROM})
    tested, weights = [], {}
    for month in months:
        train = [r for r in races if r["month"] < month]
        test = [r for r in races if r["month"] == month]
        stats = fit_stats(train, len(feats))
        for name, cols in sets.items():
            design(train + test, cols, stats)
            w = fit(train, len(cols))
            weights[name] = (cols, w, month)
            for r in test:
                r["s_" + name] = scores(r["X"], w)
        print(f"{month}: trained on {len(train)} races, testing {len(test)}")
        tested += test

    strategies = defaultdict(list)
    ll = defaultdict(float)
    for r in tested:
        n = len(r["nums"])
        p_mkt = [-math.log(o) for o in r["odds"]]  # strengths = log implied win probability
        probs = {"marche": p_top5(p_mkt, rng)}
        for name in sets:
            probs[name] = p_top5(r["s_" + name], rng)
            ll[name] += pl_loglik(r["s_" + name], r["order"])
        ll["marche"] += pl_loglik(p_mkt, r["order"])
        ll["hasard"] += sum(-math.log(n - k) for k in range(len(r["order"])))
        top = lambda p, k=5: sorted(range(n), key=lambda i: -p[i])[:k]
        shortlist = top(probs["sans_cote"], 8)
        value = sorted(shortlist, key=lambda i: -probs["sans_cote"][i] / max(probs["marche"][i], 1e-6))[:5]
        value.sort(key=lambda i: -probs["sans_cote"][i])
        tickets = {
            "Favoris (cotes)": top([-o for o in r["odds"]]),
            "IA sans cotes": top(probs["sans_cote"]),
            "IA avec cotes": top(probs["avec_cote"]),
            "Bon rapport": value,
            "Hasard": rng.sample(range(n), 5),
        }
        top5 = set(r["order"])
        for name, t in tickets.items():
            nums = [r["nums"][i] for i in t]
            rank, eur = ticket_return(nums, payouts[r["id"]])
            strategies[name].append({"found": len(top5 & set(t)), "top3": set(r["order"][:3]) <= set(t),
                                     "rank": rank, "gain": STAKE * eur, "race": r["id"], "ticket": nums,
                                     "type": r["type"]})

    k = len(tested)
    summary = {"courses_testees": k, "du": tested[0]["id"][:10], "au": tested[-1]["id"][:10],
               "mise_par_course_euros": STAKE, "strategies": {}}
    for name, res in strategies.items():
        summary["strategies"][name] = {
            "chevaux_du_top5_trouves_par_course": round(sum(x["found"] for x in res) / k, 2),
            **{rk.lower(): sum(x["rank"] == rk for x in res) for rk in ("BONUS_3", "BONUS_4SUR5", "DESORDRE", "ORDRE")},
            "mise_euros": STAKE * k, "rendu_euros": round(sum(x["gain"] for x in res), 2)}
    (DATA / "backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nTest: {k} Quinté races ({tested[0]['id'][:10]} -> {tested[-1]['id'][:10]}), one {STAKE:.0f} € ticket per race\n")
    print(f"{'strategy':<17}{'top5 found':>11}{'3 first in':>11}{'Bonus 3':>8}{'Bonus 4/5':>10}{'Désordre':>9}{'Ordre':>6}"
          f"{'staked':>8}{'returned':>10}{'net':>9}")
    for name, res in strategies.items():
        cnt = lambda rk: sum(x["rank"] == rk for x in res)
        back = sum(x["gain"] for x in res)
        print(f"{name:<17}{sum(x['found'] for x in res) / k:>11.2f}{sum(x['top3'] for x in res):>11}"
              f"{cnt('BONUS_3'):>8}{cnt('BONUS_4SUR5'):>10}{cnt('DESORDRE'):>9}{cnt('ORDRE'):>6}"
              f"{STAKE * k:>7.0f}€{back:>9.0f}€{back - STAKE * k:>+8.0f}€")
    print("\nBy race type: top-5 horses found per race | euros returned for 2 € staked per race")
    types = sorted({x["type"] for x in strategies["Hasard"]})
    print(f"{'strategy':<17}" + "".join(f"{t + ' (' + str(sum(x['type'] == t for x in strategies['Hasard'])) + ')':>22}" for t in types))
    for name, res in strategies.items():
        cells = []
        for t in types:
            sub = [x for x in res if x["type"] == t]
            cells.append(f"{sum(x['found'] for x in sub) / len(sub):.2f} | {sum(x['gain'] for x in sub):.0f}/{STAKE * len(sub):.0f} €")
        print(f"{name:<17}" + "".join(f"{c:>22}" for c in cells))
    print("\nAverage log-likelihood of the real top-5 order per race (higher = better forecast):")
    for name in ("hasard", "marche", "sans_cote", "avec_cote"):
        print(f"  {name:<10}{ll[name] / k:8.3f}")
    for name in sets:
        cols, w, month = weights[name]
        print(f"\nWeights of '{name}' (fit on races before {month}, standardised features, largest first):")
        for c, wi in sorted(zip(cols, w), key=lambda x: -abs(x[1]))[:12]:
            print(f"  {feats[c]:<22}{wi:+.3f}")
    wins = [(name, x) for name, res in strategies.items() for x in res if x["rank"] in ("DESORDRE", "ORDRE")]
    for name, x in wins:
        print(f"\n{name}: {x['rank']} on {x['race']} with {x['ticket']} -> {x['gain']:.0f} €")
    if not wins:
        print("\nNo Désordre or Ordre hit by any strategy.")


if __name__ == "__main__":
    main()
