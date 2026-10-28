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
