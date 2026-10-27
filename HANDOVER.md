# Handover — status at 2026-10-27

Read BRIEF.md first, then this file. This summarises the first working session so a
new session can continue without the original conversation.

## Where we are
| Phase | Status |
|---|---|
| 1 Programmes | **done**: 183/183 days, 2026-04-26 → 2026-10-25, all from the `online` host, 0 missing. `data/raw/programme/`, log `data/raw/fetch_log.txt` |
| 2 Structure | **done**: see "Programme fields" below; `structure_recent.txt`, `structure_old.txt` |
| 3 Runners | **Quinté only done**: 183/183 races, 0 missing. `data/raw/participants/YYYY-MM-DD/R{n}C{n}.json`, log `data/raw/participants_log_quinte.txt`. Full French run (6,581 races, ~2.5–3 h) **not started** |
| 4 Payouts | **script written, not run, endpoint not verified**. `fetch_rapports.py` uses `.../R{n}/C{n}/rapports-definitifs`. The one test from the work laptop was blocked by the corporate filter, so we know nothing about this endpoint yet |
| 5–7 | to do |

## Scripts
- `fetch_programmes.py`: phase 1. Stops after 10 MISSING days in a row.
- `fetch_participants.py --quinte | --france | --all`: phase 3. `download(scope, endpoint, out)` is shared with phase 4.
- `fetch_rapports.py --quinte | --france`: phase 4.
- `inspect_json.py FILE`: prints the key structure (first item of each list only, so also scan all items for keys that only some entries have).
- All scripts skip files already saved. Note: re-running rewrites that scope's log file with only the new lines.

## Next steps (personal laptop)
1. Test the payouts endpoint on one race by hand:
   `curl https://online.turfinfo.api.pmu.fr/rest/client/7/programme/25102026/R1/C5/rapports-definitifs`
   If it's not JSON, look for the right endpoint before running anything in bulk.
2. If it works: `python fetch_rapports.py --quinte` (183 requests, ~4 min), then inspect one file and list
   the Quinté+ / Tiercé / Bonus payout fields with exact names, as was done for the other phases.
3. Then decide on `python fetch_participants.py --france` (~2.5–3 h).

## Programme fields (`data/raw/programme/*.json`), verified on all races of 2 files
- Meetings: `programme.reunions[]`: `numOfficiel` (R), `hippodrome.code/libelleCourt`, `pays.code` (foreign meetings included: filter `FRA`), `dateReunion`.
- Races: `reunions[].courses[]`: `numOrdre` (C), `libelle`, `heureDepart` (ms epoch), `nombreDeclaresPartants`, `categorieParticularite`, `conditionAge`, `conditionSexe`, `montantPrix`.
- Discipline: `discipline` (PLAT, ATTELE, MONTE, HAIE, STEEPLECHASE, CROSS); `specialite` (PLAT, TROT_ATTELE, TROT_MONTE, OBSTACLE).
- Distance: `distance` (int), `distanceUnit`; `parcours` text.
- Surface: `typePiste` (HERBE, POUZZOLANE) often missing (almost always missing in trot). All-weather (PSF) appears only as `penetrometre.intitule` = "PSF RAPIDE". `corde` = CORDE_GAUCHE/DROITE (most races).
- Going: `penetrometre.intitule` (Bon, Bon souple, Souple, Très souple, Collant, Bon léger…) and `valeurMesure` ("4,4"). Only about 33–64% of races, rarely trot.
- Weather: per meeting, `reunions[].meteo`: `nebulositeCode`, `nebulositeLibelleCourt`, `temperature`, `forceVent`, `directionVent`.
- Quinté+: no flag. It's the race with a `paris[]` entry where `codePari` == "QUINTE_PLUS" (exactly 1 per day).
- Arrival: `ordreArrivee` = list of lists of runner numbers; `statut`; `incidents[]` (`type`, `numeroParticipants`).
- Not in this file: odds, payouts, runners (`participants` is always empty).
- Files are valid UTF-8 (a "�" in the Windows console is a display issue only).

## Runner fields (`data/raw/participants/...`), verified on all 2,873 Quinté runners
- `numPmu`, `nom`, `idCheval`, `age`, `sexe`, `statut` (PARTANT / NON_PARTANT), `musique`, `driver` (jockey too), `driverChange`, `entraineur`, `proprietaire`.
- Weight: `handicapPoids` (e.g. 615, likely kg × 10, to confirm), `poidsConditionMonte`. Flat and obstacle only, not in trot.
- Draw: `placeCorde`: flat all; trot only autostart (266/1,239); obstacle never.
- Odds: `dernierRapportDirect.rapport` (SIMPLE_GAGNANT; timestamp a median of 2 min after the start, so treat as final odds). `dernierRapportReference.rapport` is taken 30 min before the start. No place odds.
- Result: `ordreArrivee` (int, finishers only), `incident`, `distanceChevalPrecedent`, trot `reductionKilometrique`, `tempsObtenu`.
- Career: `nombreCourses`, `nombreVictoires`, `nombrePlaces`, `gainsParticipant.*`, `handicapValeur` (flat/obstacle).
- Trot only: `deferre`, `handicapDistance`, `avisEntraineur`. Also `oeilleres`, `nomPere`, `nomMere`.
- Quinté races include hurdles/steeplechase, not just flat and trot.

## Working rules agreed with the user
- One request per second, no parallelism; stop and report if more than ~10 MISSING in a row.
- Test any new endpoint with one request before a bulk run; inspect real files before writing a parser.
- Propose each phase with request count and run time, and wait for the go-ahead.
- Do not download from the Siemens work laptop/network (PMU is in the blocked "Gambling" category).
