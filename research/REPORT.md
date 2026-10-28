# Quinté+ model: leakage audit, improvements and honest evaluation (2026-10-28)

**Bottom line.** The model's predictions improved a little, and the leaks are gone. **No strategy shows a
betting edge.** Every ticket rule loses money on the races it never saw, roughly in line with what the PMU keeps.
The new model predicts the finishing order slightly better than the old one, but no better than the odds alone.
Better prediction accuracy is not the same as profitable betting. This report shows both and keeps them apart.

## 1. Data and cutoffs

- 367 Quinté+ races, 2025-10-27 → 2026-10-28, PMU only, product `QUINTE_PLUS`. All in France. Base stake €2 on
  every race, and the same payout categories all year: Ordre, Désordre, Bonus 4sur5, Bonus 3, and Ordre +
  Tirelire on 29 days. Disciplines: trot attelé 182, flat 143, obstacle 40, trot monté 2.
- One row per horse per race: no duplicate (race, number) or (race, horse id), and horse ids map 1:1 to names.
  2,709 distinct horses.
- **Odds timestamps** (from the raw feed): the "reference" odds are taken **30 minutes before the start**
  (median −29.9 min, never after). The "direct" odds are taken **after the start** (median +1.5 min).
- **Two prediction cutoffs are used:**
  - **Morning (06:00):** no odds. The history has no morning odds, so a model with odds can't be tested at 06:00.
  - **T-30:** reference odds allowed.
  - Post-start odds are never used.
- Scratches: 138 non-runners, 57 of them with reference odds (late scratches). Non-finishers among starters:
  11.6% (trot 17.5%, mostly disqualified for breaking stride; obstacle 25.6%; flat 0.3%). Dead heats in 16 races
  (4 with six horses in the top 5). 1 cancelled race, which is excluded and refunded.
- Payout units: `dividendePourUnEuro` is in cents for €1. It matches the base-stake dividend on every row. The
  Désordre ≥ Bonus 4/5 ≥ Bonus 3 ordering holds on every race. On 17 days nobody won the Ordre (dividend 0).
- Full audit: `research/results/audit.json` (`python research/audit.py`).

## 2. Leakage found and fixed (`features.py`)

| Issue | Effect | Fix |
|---|---|---|
| Jockey statistics built from past-performance files of **later** races (a ride was only known because the horse later ran a Quinté) | v1 features of a past race changed when later races were removed: 27–31 values per probe date | A ride counts only once a file listing it existed (file date ≤ race date). The shrinkage prior is computed as of that date |
| Market feature fell back to **post-start odds** when reference odds were missing | Never triggered here (reference odds exist for every starter), but a latent leak | Reference odds only |
| Global jockey prior computed over the whole year | Small look-ahead | Prior as of the race date |

`research/checks.py` tests these automatically:
- removing every later race leaves the features unchanged;
- blanking the race's own result changes nothing;
- changing the post-start odds changes nothing;
- the runner snapshot is pre-race: the first `musique` token matches the previous race 99% of the time and
  today's place only 8% of the time.

## 3. Protocol

- Walk-forward: refit every month on all earlier races, and every runner of a race stays in the same split.
- **Validation** 2026-02 → 05 (119 races): every choice is made there, including the penalty, stage discounts,
  discipline interactions, feature groups, LightGBM settings and the betting threshold.
- **Test** 2026-06 → 10 (150 races): frozen choices, reported once.
- Feature scaling is fitted on training races only. Tickets are chosen before the race and settled with the
  official dividends.
- Uncertainty: 95% intervals by resampling whole weeks, because consecutive races share conditions.
- Metrics:
  - log-likelihood of the observed top-5 order (main metric);
  - win log loss and Brier score;
  - per-horse top-5 log loss and Brier score;
  - calibration (ECE);
  - top-5 horses found.

## 4. Models compared

- **Market:** raw (log 1/odds), and fitted (PL on the market's log-probability with stage discounts).
- **Simple form:** 4 recent-form features.
- **Before:** the existing algorithm. Leaky features, Plackett-Luce (PL), L2 = 5.
- **After:** leak-free features plus trot disqualification rate, last run failed, and long break. The
  validation selected stage discounts and L2 = 20 (no odds) or 10 (with odds).
- **LightGBM** lambdarank grouped by race. Its scores become probabilities through a temperature fitted on
  held-out months; raw scores are never used as probabilities.
- Not selected on validation: discipline interactions (separate trot weights). Starting method (autostart or
  volte) is handled implicitly: draw only exists in autostart and distance handicaps only in volte.
- Not available: shoeing *change* (the feed only gives today's shoeing) and odds movement (no timestamped odds
  history; see §8).

## 5. Prediction results

**Test, 150 races** (higher log-likelihood is better; for the others, lower is better except "found"):

| Model | Cutoff | log-lik top-5 order | win log loss | top-5 Brier | found /5 | ECE top-5 |
|---|---|---|---|---|---|---|
| Uniform | - | −12.896 | 2.726 | 0.2200 | 1.79 | 0.004 |
| Simple form | morning | −12.627 | 2.614 | 0.2125 | 2.12 | 0.015 |
| LightGBM | morning | −12.397 | 2.546 | 0.2069 | 2.31 | 0.014 |
| **Before**, no odds | morning | −12.209 | 2.475 | 0.2008 | 2.44 | 0.019 |
| After (frozen protocol), no odds | morning | −12.248 | 2.480 | 0.2035 | 2.35 | 0.027 |
| **After (production)**, no odds ¹ | morning | **−12.168** | 2.464 | 0.2013 | 2.39 | 0.021 |
| Market raw | T-30 | −11.971 | 2.313 | 0.1963 | 2.55 | 0.031 |
| Market fitted | T-30 | −11.854 | 2.293 | 0.1958 | 2.54 | 0.020 |
| **Before**, with odds | T-30 | −11.914 | 2.336 | 0.1949 | 2.58 | 0.025 |
| After (frozen protocol), with odds | T-30 | −11.876 | 2.315 | 0.1961 | 2.53 | 0.028 |
| **After (production)**, with odds ¹ | T-30 | **−11.839** | 2.305 | 0.1951 | 2.61 | 0.021 |

¹ The production settings drop the "feature groups dropped on validation" step, after seeing that it failed on
test (see below). These two rows are therefore **not** clean test results.

**Which change does what** (mean log-likelihood change per race; `research/diagnostics.py`):

| Change | Validation, no odds / with odds | Test, no odds / with odds |
|---|---|---|
| Leak fix + new features | −0.04 / −0.01 | −0.01 / 0.00 |
| Stage discounts + tuned L2 | **+0.05 / +0.09** | **+0.05 / +0.08** |
| Feature groups dropped on validation | +0.01 / +0.03 | −0.08 / −0.04 |

- The leak fix costs a little measured accuracy, as expected, because the old score was partly borrowed from
  the future.
- The stage discounts and tuned L2 are the only robust gain.
- Dropping groups on 119 races was overfitting, so production keeps every group. Groups whose removal hurt
  most on validation: jockey/trainer, class/earnings, form, weight/handicap/draw. Freshness and horse-profile
  groups were neutral.

**Beyond the market** (with odds minus fitted market, test): −0.022 per race, 95% interval [−0.071, +0.026].
By discipline the result is the same: the fitted market is as good as or better than the model in flat, trot
and obstacle. **The racing features add no detectable information beyond the odds 30 minutes before the start.**

**After vs before, with odds:**
- +0.04 on test, interval [−0.03, +0.11];
- +0.07 on validation + test, interval [+0.01, +0.14].

That's a small but consistent gain.

## 6. Joint probabilities (Plackett-Luce)

- Win probabilities are an exact softmax. Ordered tickets and 5-horse sets (sum over 120 orders) are exact.
  Top-5 per horse and the Bonus categories use 20,000 Gumbel simulations; `checks.py` compares them with the
  exact values within 4 standard errors.
- **Multiplying the five individual top-5 chances is wrong.** In the check it overstates the set's real
  probability by about 2.5 times, so the code never does it.
- **PL assumption check:** plain PL treats every place like the win. The fitted stage factors (0.80, 0.70,
  0.56, 0.47 for places 2–5 on the latest fit) show the later places are much less predictable. Modelling this
  is the gain in §5.
- Calibration of top-5 chances is good from 10% to 80%. Above 80% the model is overconfident: 0.86 predicted vs
  0.65 observed, on 37 horses. Part of that is non-finish risk (trot disqualifications); the "trot disqualified
  rate" feature only partly captures it.
- Rare ordered outcomes (Ordre ≈ 0.01–0.1% per ticket) can't be checked for calibration on 269 races.

## 7. Payouts, expected value and paper betting

**Payout model** (separate from prediction, fitted on earlier races only):

- Form: log dividend ~ market popularity of the winning combination (sum of log implied win probabilities
  from the T-30 odds) + field size + discipline.
- Fit before the test (R² / residual SD in log): Ordre 0.79 / 0.77, Désordre 0.86 / 0.67, Bonus 4/5 0.84 / 0.36,
  Bonus 3 0.83 / 0.34.
- On the real arrivals, actual/predicted dividend averages about 0.99 (median 0.86–0.94).
- The PMU feed exposes no combination-level pool data in the endpoints used here. Combination popularity is
  therefore estimated from past payouts, not from multiplying odds.

**Expected value of a ticket** = Σ over categories P(category) × E[dividend | category] − €2, best category only.

- Tirelire is excluded: eligibility can't be verified.
- An Ordre with no winner falls back to the Désordre dividend.
- Candidates: every 5-horse set from the model's top 8. Bet only if the expected value is above a threshold
  chosen on validation (€2); otherwise no bet.

**Where the expected value comes from** (model's top-5 ticket, 269 races, €538 staked):

| Category | Predicted | Realised |
|---|---|---|
| Bonus 4/5 + Bonus 3 | €300 | €283 |
| Désordre + Ordre | €269 | €16 |
| Total | €569 (+6%) | €299 (−44%) |

- Hit counts match: set hits expected 3.0, observed 3; Bonus 4/5 expected 43, observed 36; Bonus 3 expected 22,
  observed 30.
- The frequent categories are well estimated, and on their own they return about 53% of the stake.
- A positive expected value depends entirely on the rare Ordre and Désordre tail. Only 0.15 Ordre hits were
  expected over these 269 races, so this tail can't be validated here. It's probably optimistic: a ticket in
  the favourites' order is the most heavily backed order, so it pays less than an average Ordre on that set.

**Paper betting, test, 150 races, €2 per ticket, official dividends:**

| Strategy | Bets | Returned / staked | ROI | 95% CI (weeks) | ROI without the best payout | Max drawdown |
|---|---|---|---|---|---|---|
| Random | 150 | €15 / €300 | −95% | [−100%, −85%] | −100% | −€285 |
| Favourites (T-30) | 150 | €171 / €300 | −43% | [−66%, −11%] | −55% | −€160 |
| Before: top 5 with odds | 150 | €213 / €300 | −29% | [−53%, +4%] | −41% | −€119 |
| After (production): top 5 with odds ¹ | 150 | €186 / €300 | −38% | [−55%, −20%] | −44% | −€114 |
| Before: top 5 no odds | 150 | €147 / €300 | −51% | [−66%, −34%] | −55% | −€153 |
| After (production): top 5 no odds ¹ | 150 | €150 / €300 | −50% | [−67%, −31%] | −56% | −€150 |
| Before: bon rapport | 150 | €169 / €300 | −44% | [−81%, +4%] | −63% | −€133 |
| After (production): bon rapport ¹ | 150 | €113 / €300 | −62% | [−88%, −29%] | −73% | −€189 |
| Expected value > €2, no-bet otherwise | 9 | €14 / €18 | −22% | too few bets | −64% | −€12 |
| Expected value > 0 (diagnostic) | 122 | €194 / €244 | −20% | [−82%, +101%] | −74% | −€155 |

- Validation (119 races) tells the same story: everything loses, from −35% to −81%.
- By discipline and by month (`evaluation.json`), no strategy is positive in a way that survives removing its
  largest payout. The "expected value > 0" row is −20% only because of one €130 Désordre; without it, −74%.
- The differences between the model-based strategies and the favourites are well inside the confidence
  intervals. **No strategy demonstrates profitability.**

## 8. What changed in production (same app, same screens)

- `features.py`: leaks fixed, and 3 trot/form features added (`f_dq_rate`, `f_last_failed`, `f_long_break`).
- `backtest.py`: `fit_pl` adds stage discounts and a tunable L2, and `p_top5` simulates with those discounts.
  It's plain Python and matches the numpy fit (weight correlation 0.9999998, same test log-likelihood).
- `daily.py`: both models are fitted with stage discounts (L2 20 without odds, 10 with odds) and all features
  are kept. The prono, bon rapport and refresh work as before. Each refresh now also saves a timestamped odds
  snapshot (`data/daily/raw/YYYY-MM-DD/participants_HHMM.json`) to build the odds-movement history that is
  missing today.

## 9. Limitations and what would be needed for a credible profitability test

1. **Sample size.** Over 150 races, ROI intervals are about ±20–40 percentage points. Detecting an edge of a
   few points would need thousands of races, years of daily Quintés (estimate). One big payout can flip any
   result.
2. **Morning odds history.** The app bets in the morning, but historical odds exist only 30 minutes before the
   start. The odds snapshots now being saved will allow a morning-odds test after a few months.
3. **Combination-level pool information** (Quinté+ combination popularity or projected payouts) is not in the
   endpoints used. Whether the PMU publishes it for the Quinté+ is unverified; testing an endpoint needs one
   request and your go-ahead.
4. **Rules not verified from official documents:**
   - Tirelire eligibility: excluded.
   - Non-runner handling on a Quinté+ ticket: our tickets only use declared starters at the cutoff; 57 late
     scratches show the risk exists.
   - Flexi stakes: not modelled; every ticket is a €2 unit.
   - Paying the best category only: assumed, consistent with the data.
5. **Jockey/trainer form from all French races** was tested (section 11) and not adopted. Only one year of
   all-race data exists, so the test has 269 races with the features.
6. **Shoeing change** isn't derivable (no previous shoeing in the feed).
7. **Selection after test:** the production settings differ from the frozen protocol by one step, the
   feature-group drop. Their test rows are labelled.

## 10. Pre-registered hold-out year (2024-11 → 2025-10)

The data now covers 3 years (2023-10-27 → 2026-10-27, 1,097 Quintés). The 2024-11 → 2025-10 year was
downloaded only after `PREREGISTRATION.md` was committed (558b1f2). Models, settings and strategies are
frozen, with no tuning. There are 364 races, all under the Bonus 4sur5 rule (141 flat, 181 trot,
42 jumps), and none has incomplete reference odds. The programme of 2024-11-07 is missing (two
attempts), so that race is absent.

| Model | Log-lik. top-5 order | Win log-loss | Horses found in top 5 |
|---|---|---|---|
| uniform | −12.863 | 2.720 | 1.64 |
| market raw (T-30) | −11.837 | 2.327 | 2.54 |
| market fitted (T-30) | −11.783 | 2.310 | 2.55 |
| before, no odds | −12.141 | 2.463 | 2.38 |
| before, with odds | −11.830 | 2.349 | 2.55 |
| after, no odds | −12.134 | 2.450 | 2.40 |
| after, with odds | −11.790 | 2.324 | 2.57 |

Pre-declared verdicts:
- **After vs before, with odds:** +0.041 per race, CI [+0.009, +0.073]. This is an **improvement**, confirmed
  out of sample.
- **After vs before, no odds:** +0.007, CI [−0.019, +0.034]. Not demonstrated.
- **Model vs market:** −0.007, CI [−0.042, +0.027]. **No information beyond the odds.**
- **Betting:** no strategy has a demonstrated edge. Favourites −57%, random −84%, top 5 after with odds −57%,
  bon rapport −78%, EV > €2 −52% (14 bets). "Top 5 before, no odds" shows +21%, CI [−60%, +140%], but it
  becomes −25% without its largest payout. That comes from a few large Désordre payouts, not an edge.

Details are in `results/holdout_2024-11_2025-10.json`.

## 11. Jockey/trainer form from all French races (not adopted)

All French races from 2025-10-27 to 2026-10-27 were downloaded (12,451 races, 162,640 runners;
`data/all_runs.csv`). From them we build 90-day jockey and trainer form, run counts, and 30-day trainer form,
dated strictly before the race. The leakage checks pass.

Pre-declared rule (`form_eval.py`): adopt only if the mean log-likelihood gain is positive on both the
validation and the test months, for both models.

| Model | Validation (119 races) | Test (150 races) |
|---|---|---|
| no odds | **+0.074** [+0.036, +0.109] | +0.029 [−0.013, +0.070] |
| with odds | +0.017 [−0.007, +0.040] | −0.002 [−0.019, +0.015] |

**Not adopted.** With odds, the form adds nothing: the market already prices it. Without odds, the gain is
clear on validation and positive but uncertain on test. That's a plausible effect worth re-testing once
more all-race months exist. Adopting it would also need a daily download of all French races (about 35
requests a day). Production is unchanged.

## Reproduce

```
python build_dataset.py && python features.py              # tables (plain Python)
python research/audit.py                                   # data audit
python research/checks.py                                  # leakage + settlement checks
python -m venv .venv && .venv/Scripts/pip install -r research/requirements.txt
.venv/Scripts/python research/checks.py                    # + probability checks
.venv/Scripts/python research/evaluate.py                  # frozen protocol (~15 min), results/evaluation.json
.venv/Scripts/python research/diagnostics.py               # decomposition + EV diagnostics (post-hoc)
.venv/Scripts/python research/diagnostics.py --production  # production settings, betting (post-hoc)
```
`evaluate.py` also needs `data/research/features_v1.csv` (the pre-fix feature table, kept locally) for the
"before" rows.
