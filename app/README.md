# Le Quinté du Jour — phone web app

Private web app for one person: the day's Quinté+ with the AI's picks, a "bon rapport"
(value) mode, a one-tap explanation per horse, input for Le Veinard's picks, and a
scoreboard. French UI, phone-sized, no login, no build step.

## Run it locally
The app loads its data with `fetch()`, which browsers block for files opened directly
(`file://`). Serve the folder instead:

```bash
cd app
python -m http.server 8000
```

Open http://localhost:8000. To try it on a phone on the same Wi-Fi, open
`http://<your-computer-ip>:8000`.

## Files
| File | Role |
|---|---|
| `index.html` | Page shell: three tabs (Le Quinté, Mes numéros, Le match) and the horse sheet. |
| `styles.css` | All styling. Colors are tokens on `:root`, with a dark-mode set. |
| `app.js` | Reads `data/today.json` and renders everything. Holds no race data. |
| `data/today.json` | The day's race, runners, picks, tickets, explanations, yesterday's result, scoreboard. **Written by the data pipeline (phase 7 of `BRIEF.md`).** |

`data/today.json` is not in the repo: run `python daily.py` from the repo root to generate it
(it writes `app/data/today.json`). `app_export.py` converts the pipeline output to the format below.

## `today.json` contract (schema_version 1)
The pipeline must produce exactly this shape. The app computes nothing about horses:
ranks, chances, verdicts and factor explanations all come from the pipeline.

```jsonc
{
  "schema_version": 1,
  "generated_at": "ISO-8601 UTC",
  "generated_by": "free text: model name/version",
  "odds_updated": "11h42",           // Paris time of the odds behind the prono; null if no odds yet
  "race": {
    "date": "YYYY-MM-DD",            // also used as the key for saved picks on the phone
    "date_label": "Mardi 27 octobre",
    "reunion": "R1", "course": "C1",
    "hippodrome": "Deauville", "prix": "Prix d'Utah Beach",
    "start_time": "13h55", "distance_m": 1300,
    "surface": "sable", "race_type": "Handicap", "runners_count": 15
  },
  "runners": [{
    "num": 2, "name": "Michael The Great", "jockey": "S. Pasquier",
    "weight_kg": 59.5, "sex_age": "Mâle 3 ans", "draw": 1,
    "musique": "3p 1p 3p 2p 9p (25) 1p ...",   // "(25)" marks the start of 2025
    "non_runner": false,
    "ai_rank": 1,                 // null for non-runners
    "win_chance_pct": 21,         // model estimate, rounded; null for non-runners
    "verdict": "One plain French sentence.",
    "factors": [                  // shown in the horse sheet, in this order
      { "sign": "plus|minus|eq|q", "title": "Forme", "text": "Plain French explanation." }
    ]
  }],
  "modes": {
    "prono": { /* mode object */ },
    "value": { /* mode object */ }
  },
  "other_tips": { "sources": [{ "source": "Zone-Turf", "picks": [4, 9, 2, 7, 13] }], "note": "..." },
  "yesterday": {
    "label": "Lundi 26 octobre · Chantilly",
    "arrival": [12, 3, 10, 13, 9],
    "sources": [{ "source": "Zone-Turf", "picks": [13, 12, 3, 16, 1] }]  // empty picks = "Pas de prono"
  },
  "season": {
    "players": [{ "name": "L'IA qui parle aux chevaux", "bet_eur": 0, "won_eur": 0 }],
    "note": "..."
  }
}
```

Mode object:
```jsonc
{
  "title": "Mes 5 chevaux pour l'arrivée",
  "intro": "optional sentence",
  "picks": [2, 12, 4, 14, 10],
  "ticket": {
    "title": "Mon ticket conseillé",
    "lines": [{ "bet": "Quinté+ unitaire", "nums": [2, 12, 4, 14, 10], "price_eur": 2 }],
    "total_eur": 3,
    "note": "optional"
  },
  "payouts": {                     // optional; label every figure as an estimate
    "title": "Si ça passe (estimation)",
    "rows": [{ "label": "Tiercé dans l'ordre", "range": "80 à 430 €" }],
    "note": "optional"
  },
  "notices": ["optional warning boxes shown above the ticket"]
}
```

`factors[].sign`: `plus` = helps, `minus` = hurts, `eq` = neutral or uncertain, `q` = data not available yet.

## Known limits of the prototype
- Picks for the phone ("Le Veinard", "Mon jeu") are saved in that phone's browser only.
  The scoreboard can't count them until they're sent somewhere durable (phase 7).
- The fatigue and ground/weather factors are placeholders until the history data exists.
- Odds were not published when this file was made, so the "bon rapport" payouts are estimates
  from assumed odds.

## Deploying on Railway
`server.py` (repo root, standard library only) serves this folder and runs the daily job at
06:00 Paris time. `/data/today.json` comes from the Railway volume, `/status` shows the job log.
The "Actualiser les cotes" button posts to `/refresh`: the server re-ranks the prono with the latest
odds (one feed request, no refit, at most once a minute) until the start; "Le bon rapport" keeps its list.

1. New Railway project → Deploy from GitHub repo `quinte-plus`. Railway builds the `Dockerfile`.
2. Add a **volume** to the service (mount path e.g. `/data`, 1 GB is plenty). Railway sets
   `RAILWAY_VOLUME_MOUNT_PATH`, which the scripts use for all data.
3. Optional variable: `APP_PASSWORD` = a password (the phone asks for it once; any user name).
   Without it the app is open to anyone who has the URL.
4. Settings → Networking → Generate domain. Keep the URL private.

First start on the empty volume: the server downloads 12 months of Quinté data (~1,460 requests,
one per second, ~25 min) and runs the backtest before the first tips appear. Watch `/status`.
