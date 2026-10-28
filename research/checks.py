"""
Automated checks: leakage, probability consistency, ticket settlement.

    python research/checks.py          # leakage + settlement (plain Python)
    <venv>/python research/checks.py   # also the probability checks (numpy)

Exits with an error if a check fails. No downloads.
"""
import copy
import csv
import importlib.util
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import backtest as bt  # noqa: E402
import features as ft  # noqa: E402
from paths import DATA  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def load(name):
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def feats_of(rows, race_id):
    return {r["numPmu"]: {k: v for k, v in r.items() if k.startswith("f_")} for r in rows if r["race_id"] == race_id}


# ---------------------------------------------------------------- leakage

def leakage(compute, label, strict=True):
    """Features of a race must not change when everything after it is removed, when its own
    result is blanked, or when the post-start odds change."""
    runners, past = load("runners.csv"), load("past_runs.csv")
    days = sorted({r["date"] for r in runners})
    probe = [days[len(days) // 5], days[len(days) // 2], days[4 * len(days) // 5]]
    full = compute(runners, past)
    bad = 0
    for d in probe:
        rid = next(r["race_id"] for r in runners if r["date"] == d)
        cut = compute([r for r in runners if r["date"] <= d], [p for p in past if p["race_id"][:10] <= d])
        a, b = feats_of(full, rid), feats_of(cut, rid)
        diff = [(n, k) for n in a for k in a[n] if a[n][k] != b[n][k]]
        bad += len(diff)
        if not strict:
            print(f"      {label} {rid}: {len(diff)} feature values change when later races are removed"
                  + (f", e.g. {sorted({k for _, k in diff})[:4]}" if diff else ""))
    if strict:
        check(f"{label}: features unchanged when all later races are removed ({len(probe)} dates)", bad == 0, f"{bad} differences")
        rid = next(r["race_id"] for r in runners if r["date"] == probe[1])
        blank = [dict(r, res_place="", res_top5="", res_incident="", cote_finale=str(random.random() * 50))
                 if r["race_id"] == rid else r for r in runners]
        noisy = [dict(r, cote_finale=str(random.random() * 50)) for r in runners]
        check(f"{label}: features unchanged when the race's own result is blanked", feats_of(compute(blank, past), rid) == feats_of(full, rid))
        check(f"{label}: features unchanged when post-start odds (cote_finale) change everywhere",
              feats_of(compute(noisy, past), rid) == feats_of(full, rid))
    return bad


def musique_snapshot():
    """The runner snapshot must be pre-race: the first musique token should match the horse's
    previous race, not today's result."""
    runners, past = load("runners.csv"), load("past_runs.csv")
    last = {}
    for p in past:
        key = (p["race_id"], p["numPmu"])
        if key not in last or p["past_date"] > last[key]["past_date"]:
            last[key] = p
    same_prev = same_today = n = 0
    for r in runners:
        m = re.match(r"(\d+)", r["musique"] or "")
        key = (r["race_id"], r["numPmu"])
        if not m or key not in last or not r["res_place"] or last[key]["statusArrivee"] not in ("PLACE", "NON_PLACE"):
            continue
        n += 1
        tok = int(m.group(1))
        same_prev += tok == int(float(last[key]["place"] or 0)) or (tok == 0 and float(last[key]["place"] or 0) >= 10)
        same_today += tok == int(r["res_place"])
    check("runner snapshot is pre-race (musique starts with the previous race, not today's result)",
          same_prev / n > 0.8 and same_today / n < 0.3,
          f"first token = previous race {same_prev / n:.0%}, = today's place {same_today / n:.0%} ({n} runners)")


# ---------------------------------------------------------------- settlement

def settlement():
    rows = [("QUINTE_PLUS:ORDRE", [1, 2, 3, 4, 5], 1000.0), ("QUINTE_PLUS:DESORDRE", [1, 2, 3, 4, 5], 50.0)]
    rows += [("QUINTE_PLUS:BONUS_4SUR5", c, 5.0) for c in ([1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5])]
    rows += [("QUINTE_PLUS:BONUS_3", [1, 2, 3], 2.0)]
    cases = {
        "exact order -> Ordre": ([1, 2, 3, 4, 5], ("ORDRE", 1000.0)),
        "same five, other order -> Désordre": ([5, 4, 3, 2, 1], ("DESORDRE", 50.0)),
        "4 of the first 5 -> Bonus 4sur5": ([1, 2, 3, 4, 9], ("BONUS_4SUR5", 5.0)),
        "4 of the first 5 without the winner -> Bonus 4sur5": ([2, 3, 4, 5, 9], ("BONUS_4SUR5", 5.0)),
        "first three only -> Bonus 3": ([3, 1, 2, 8, 9], ("BONUS_3", 2.0)),
        "3 of the first 5 but not the first three -> nothing": ([1, 2, 4, 8, 9], (None, 0.0)),
    }
    for name, (ticket, want) in cases.items():
        check(f"settlement: {name}", bt.ticket_return(ticket, rows) == want, str(bt.ticket_return(ticket, rows)))
    dead = [("QUINTE_PLUS:DESORDRE", [1, 2, 3, 4, 5], 40.0), ("QUINTE_PLUS:DESORDRE", [1, 2, 3, 4, 6], 45.0)]
    check("settlement: dead heat pays whichever Désordre row matches",
          bt.ticket_return([1, 2, 3, 4, 6], dead) == ("DESORDRE", 45.0) and bt.ticket_return([1, 2, 3, 4, 5], dead) == ("DESORDRE", 40.0))
    nowin = [("QUINTE_PLUS:ORDRE", [1, 2, 3, 4, 5], 0.0), ("QUINTE_PLUS:DESORDRE", [1, 2, 3, 4, 5], 50.0)]
    check("settlement: Ordre with no winner (dividend 0) falls back to the Désordre dividend",
          bt.ticket_return([1, 2, 3, 4, 5], nowin) == ("DESORDRE", 50.0))
    pay = bt.load_payouts()
    check("settlement: Tirelire jackpot rows are not paid (eligibility not verifiable)",
          all(p != "QUINTE_PLUS:ORDRE_TIRELIRE" for v in pay.values() for p, _, _ in v))
    real = pay["2026-10-27_R1C1"]  # arrival 10-6-9-15-3
    check("settlement replay 2026-10-27: 10-5-6-15-12 wins nothing", bt.ticket_return([10, 5, 6, 15, 12], real) == (None, 0.0))
    check("settlement replay 2026-10-27: 10-9-6-15-12 -> Bonus 4sur5 33 € per 1 €",
          bt.ticket_return([10, 9, 6, 15, 12], real) == ("BONUS_4SUR5", 33.0))
    # every race: the official arrival as a ticket gets the official Désordre (or Ordre) dividend
    runners = load("runners.csv")
    arrivals = defaultdict(list)
    for r in runners:
        if r["res_place"] and int(r["res_place"]) <= 5:
            arrivals[r["race_id"]].append((int(r["res_place"]), int(r["numPmu"])))
    bad = 0
    for rid, a in arrivals.items():
        if len(a) != 5 or rid not in pay:
            continue
        ticket = [n for _, n in sorted(a)]
        rank, eur = bt.ticket_return(ticket, pay[rid])
        bad += rank not in ("ORDRE", "DESORDRE") or eur <= 0
    check("settlement replay: each official arrival is paid Ordre or Désordre", bad == 0, f"{bad} races not paid")


# ---------------------------------------------------------------- probabilities (numpy)

def probabilities():
    try:
        import itertools

        import numpy as np
        import plmodel as plm
    except ImportError:
        print("[skip] probability checks need numpy/scipy (run with the research venv)")
        return
    rng = np.random.default_rng(3)
    for lam in (np.ones(5), np.array([1, .9, .8, .75, .7])):
        s = rng.normal(size=15)
        pw = plm.win_probs(s, lam[0])
        check("probabilities: win probabilities sum to 1", abs(pw.sum() - 1) < 1e-12)
        two = plm.perm_probs(s, lam, np.array(list(itertools.permutations(range(15), 2))))
        check("probabilities: all ordered pairs for 1st-2nd sum to 1", abs(two.sum() - 1) < 1e-10)
        top = tuple(np.argsort(-s)[:5])
        exact = plm.perm_probs(s, lam, np.array([top[:2]]))[0]
        orders = plm.sample_orders(s, lam, 400000, rng)
        mc = np.mean((orders[:, 0] == top[0]) & (orders[:, 1] == top[1]))
        se = np.sqrt(exact * (1 - exact) / len(orders))
        check("probabilities: simulated first-two order matches the exact value (within 4 SE)", abs(mc - exact) < 4 * se,
              f"exact {exact:.5f}, simulated {mc:.5f}")
        p5 = plm.top5_probs(orders, 15)
        check("probabilities: top-5 probabilities sum to 5 and lie in [0, 1]", abs(p5.sum() - 5) < 1e-9 and p5.min() >= 0 and p5.max() <= 1)
        pset = plm.set_prob(s, lam, top)
        mc_set = np.mean((np.sort(orders, axis=1) == np.sort(np.array(top))).all(axis=1))
        se = np.sqrt(pset * (1 - pset) / len(orders))
        check("probabilities: simulated 5-horse set matches the exact 120-order sum (within 4 SE)",
              abs(mc_set - pset) < 4 * se, f"exact {pset:.5f}, simulated {mc_set:.5f}")
        prod = np.prod(np.sort(p5)[-5:])
        print(f"      (product of the five top-5 probabilities = {prod:.4f} vs real set probability {pset:.5f}: "
              f"multiplying marginals is wrong)")


def main():
    random.seed(0)
    leakage(ft.compute, "features v2")
    spec = importlib.util.spec_from_file_location("features_v1", DATA / "research" / "features_v1.py")
    if spec and (DATA / "research" / "features_v1.py").exists():
        v1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v1)
        print("[info] the previous feature code (v1), for comparison:")
        leakage(v1.compute, "features v1", strict=False)
    musique_snapshot()
    settlement()
    probabilities()
    print(f"\n{len(FAILED)} check(s) failed" if FAILED else "\nall checks passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
