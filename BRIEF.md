# Quinté IA — data and model brief

Private hobby app for one person (Michael's father-in-law, in France). He plays the
PMU Quinté+ daily, about €10–20 a week. The phone web app already exists as a
prototype (tips, per-horse explanations, "bon rapport" mode, scoreboard). This
project builds the data and the model behind it.

## Goal
1. Collect ~6 months (ideally 12) of French PMU races: runners, results, odds, payouts,
   ground, weather.
2. Learn which factors predict the top 5 and each horse's own preferences.
3. Backtest two strategies on races the model never saw, after the PMU's cut:
   - **Prono**: the AI's 5 most likely finishers.
   - **Bon rapport**: horses whose AI chance is higher than the chance implied by their odds.
4. Produce, each morning, a JSON file of the day's Quinté that the app can display.

## Ground rules
- Personal use only. One request per second, no parallel calls, cache everything.
- The PMU feed (turfinfo) is undocumented. Never assume a field name: inspect real
  files first (`inspect_json.py`), then write the parser.
- Any number shown to the user that is estimated must be labelled as such.

## Phases
| Phase | What | Status |
|---|---|---|
| 1 | Download daily programmes → `data/raw/programme/YYYY-MM-DD.json` | script ready |
| 2 | Inspect structure; list which fields exist (meetings, races, discipline, distance, ground/going, weather, Quinté flag, arrival) | to do |
| 3 | Download per-race runners (`.../programme/DDMMYYYY/R{n}/C{n}/participants`): musique, jockey, trainer, weight, draw, odds | to do |
| 4 | Find the payouts endpoint (not verified yet) and download rapports for Quinté races at least | to do |
| 5 | Flatten into `data/runners.csv` (one row per runner per race) | to do |
| 6 | Features + model + backtest | to do |
| 7 | Daily job: today's Quinté → `today.json` for the app | to do |

## Features to build (once we know the data has them)
- Form: places in last 3/5 runs, win and top-3 rates, trend.
- Freshness: days since last run, runs in last 30/90 days.
- Course/distance/surface record: runs and places at this track, at ±200 m, on PSF vs turf.
- Ground: the horse's record on the day's going (bon, souple, lourd…).
- Weight vs field average, age, sex, draw (corde), field size.
- Jockey and trainer win rate over the last 60 days.
- Market: final odds and implied probability (the baseline to beat).

## Model and test
- Start simple: logistic regression or gradient boosting for "finishes top 5", per discipline (plat / trot attelé / trot monté / obstacle).
- Train on all French races in the window, test only on later races (time split, no shuffling).
- Report: hit rates for Bonus 3, Tiercé désordre, Quinté désordre; money bet vs money returned
  per strategy; whether the model adds anything beyond the odds alone.
