# Pre-registered hold-out test (written 2026-10-28, before the data was downloaded)

The 2023-10-27 → 2025-10-26 Quinté data is being downloaded now. Nobody has seen it: not the
model, not me. Everything below is fixed **before** it arrives, so the hold-out result can't be
tuned. The git commit of this file is the timestamp.

## Hold-out period
- **Test: 2024-11-01 → 2025-10-26.** Walk-forward: each month is predicted by a model refitted on
  every earlier race since 2023-10-27, never on the month itself or later.
- 2023-10-27 → 2024-10-31 is training only (the first test month trains on 12 months, as production does).
- Cutoffs: morning (no odds) and T-30 (reference odds, 30 min before the start). No post-start odds.

## Frozen models (no tuning on this period)
| Name | Features | Plackett-Luce settings |
|---|---|---|
| before, no odds | v1 feature code (`data/research/features_v1.py`), market column excluded | L2 5, no stage discounts |
| before, with odds | v1 feature code, all columns | L2 5, no stage discounts |
| after, no odds (production) | current `features.py`, all groups, market excluded | L2 20, stage discounts |
| after, with odds (production) | current `features.py`, all groups | L2 10, stage discounts |
| market fitted | market log-probability only | L2 0.01, stage discounts |
| market raw, uniform | - | - |

## Frozen strategies (one €2 Quinté+ ticket per race, official dividends)
- favourites (5 shortest reference odds); random (seeded by race id);
- top 5 of each model above (by P(top 5));
- bon rapport: the 5 of the "after, no odds" model's top 8 with the highest model/market top-5
  ratio (production rule);
- expected value: best 5-of-top-8 ticket of "after, with odds", bet only if expected net > €2
  (threshold chosen on the 2026 validation months), payout model fitted on earlier races only,
  using the bonus rule in force for that race (Bonus 4 or Bonus 4sur5).

## Pre-declared reading of the result
- Prediction: "after" improves on "before" if the mean log-likelihood gain of the top-5 order is
  positive **and** its 95% week-block interval excludes 0.
- Market: the model adds information beyond the odds only if "after, with odds" beats "market
  fitted" with an interval excluding 0.
- Betting: a strategy shows an edge only if its ROI interval lies entirely above 0 **and** it stays
  positive without its largest payout. Anything else is reported as no demonstrated edge.
- Rule eras (Bonus 4 vs Bonus 4sur5) and disciplines are reported separately.

---

# Morning model: four feature ideas (written 2026-11-16, before any run on the test periods)

Context: the user plays in the morning, and the bon rapport list is dropped. The morning prono (no
odds) finds 2.41 of the first five against 2.59 with the odds 30 minutes before the start, and its
log-likelihood is 0.34 per race behind the fitted market (`results/baseline_2026-11-15.json`). The
with-odds model has almost no room left. So the ideas target what the bettors know and the morning
model does not.

## What was looked at before writing this
Only the design year, 2023-10-27 → 2024-10-31, which is never a test period: coverage of the new
columns, and the log-likelihood gain on 2024-05..10 (184 races, model fitted on earlier races).
No-odds model: memory +0.015, field +0.028, musique +0.009, speed +0.012, the four together +0.047
[+0.006, +0.085]. With odds: between −0.010 and +0.006. Nothing was tuned on it: the definitions
below are the first ones written. The code is `research/ideas.py` at this commit.

## The ideas (all known in the morning, none uses the day's odds)
| Idea | Columns | Source |
|---|---|---|
| memory | `f_prev_mkt`: mean over the horse's last 3 Quinté+ starts in the past 365 days of log(market win chance × runners), from the reference odds of those races. `f_prev_runs`: log(1 + such starts). `f_prev_top5`: smoothed share of them finished in the first five | `runners.csv`, earlier race days only |
| field | `f_form_pct`: mean finishing position as a share of the field over the last 5 runs (0 = won, 1 = last or unplaced; runs not finished left out). `f_form_pct_best`: the best of them. `f_class_perf`: log(prize of the past race / today's prize), averaged with weight (1 − share) | `past_runs.csv` |
| musique | Over the last 10 runs of the form line: mean place (unplaced or not finished = 10), smoothed top-3, win and not-finished rates | `runners.csv` (`musique`, a pre-race snapshot) |
| speed | `f_trot_gap`, `f_trot_best_rel`, exactly as in `speed_eval.py` | `past_runs.csv` |

The speed idea is a re-test: REPORT.md §13 saw +0.012 / +0.016 / +0.013 for the no-odds model on
these same periods, and rejected it because the rule also required a gain with odds. Its result here
is therefore not new evidence; it is in the pack because the morning model is now the target.

## Models and settings (frozen)
Production settings at this commit, read by `benchmark.py` from `daily.py`: no odds L2 20, with odds
L2 10, stage discounts, every current feature. A candidate is the production model plus the columns
of one idea, or of the four together ("morning pack"). No other change, no tuning.
Walk-forward by month, 2024-11 → 2026-11-15, 743 races.

## Rule
For a model (no odds, or with odds), a candidate **passes** if its mean log-likelihood gain over
production is positive in each of the three periods (hold-out 2024-11..2025-10, validation
2026-02..05, test 2026-06..10) **and** the 95% week-block interval of the gain over all 743 races
lies above 0.
- Primary test: the morning pack on the no-odds model. If it passes, the pack is adopted for the
  morning model.
- If it fails, a single idea that passes on its own may be adopted for the morning model; that is
  four more tests, and the report must say so.
- With odds: same rule, the pack first. Not passing means the with-odds model stays as it is.
- The 18 races from 2026-10-29 are reported, not used: too few.

Also reported, not part of the rule: horses of the first five found by the top-5 ticket, days the
ticket pays, euros returned. A better log-likelihood is not a betting edge; the report must not
present it as one.

## Forward test
Whatever is adopted is frozen at the adoption commit. Its clean test is the races from 2026-11-16
on, scored by `benchmark.py` (period "fresh"), read once at least 150 such races exist.

