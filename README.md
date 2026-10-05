# NBA Prediction System

10-factor team model + player props engine for the **2026-27 NBA season**.

---

## Scripts

| Script | Purpose |
|---|---|
| `nba_combined.py` | **Main entry point.** Runs team model and player props together for today's games. |
| `newnbapredictor.py` | 10-factor team model (win probability, spread, over/under). |
| `playerlinepredictor.py` | Player props engine (projections vs market lines, DvP, calibration). |
| `oddstracker.py` | Fetches and logs book odds; computes lean/devig for calibration. |
| `calibrate.py` | Reads `bet_log.csv` → produces `calibration.json` for confidence adjustments. |
| `train_model.py` | Walk-forward backtest that learns regression weights from historical data. |
| `diagnose.py` | Diagnostic script for manual inspection of model internals. |

---

## How to run

### 1. Install dependencies

Use one Python version for local runs and CI (CI uses 3.11):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you see TLS errors on macOS (LibreSSL), run `pip install --upgrade certifi`.

### 2. Configure API keys

Copy the example env file and fill in your keys (do not commit `.env`):

```bash
cp .env.example .env
```

Edit `.env`:

```
ODDS_API_KEY=your_key_here          # https://the-odds-api.com/
API_FOOTBALL_KEY=your_key_here      # https://www.api-football.com/
```

Scripts load `.env` automatically via `python-dotenv`. You can still `export ODDS_API_KEY=...` in the shell if you prefer; existing environment variables are not overwritten.

### 3. Run one slate

Every run belongs to one slate date on the US Eastern (NBA) calendar:

```bash
python3 nba_combined.py                     # today's slate
python3 nba_combined.py --date 2026-10-20   # a specific slate
python3 nba_combined.py --next              # the next date with games (scans 44 days)
python3 nba_combined.py --date 2026-10-20 --verbose
```

Only odds events that tip off on that date feed props, spreads and book lean, and every team is stored with its NBA Stats code (`team_codes.py`), whichever source it came from.

### 4. Daily cycle

```bash
python3 oddstracker.py 2026-10-20           # odds snapshot for that slate only
python3 nba_combined.py --date 2026-10-20   # predictions and picks → predictions.db
python3 checkresults.py 2026-10-20          # after the games: grade props and games
python3 calibrate.py --summary              # refresh calibration.json
```

Each run appends one line to `run_log.jsonl`, with a status for every stage:

| Status | Meaning |
|---|---|
| `ok` | The source answered and returned data |
| `no_games` | The source answered: nothing scheduled or played |
| `market_unavailable` | The source answered: no lines posted for that event |
| `fetch_failed` | A request failed, or games weren't final: unknown, not empty |
| `skipped` | Nothing to do |

Exit codes: `0` ok or no games, `2` failed (nothing saved, rerun later), `3` partial (some games' odds failed). Grading writes nothing when stats.nba.com can't be reached or a game isn't final, and regrading a date never duplicates rows in `bet_log.csv` or `game_log.csv`.

### Auditable prop cycle (paper stakes)

Run the entry snapshot and prediction on the same Eastern slate date, then
collect closing quotes and grade after the games are final:

```bash
python3 oddstracker.py YYYY-MM-DD
python3 nba_combined.py --date YYYY-MM-DD
python3 oddstracker.py --closing YYYY-MM-DD          # next 30 minutes only
python3 oddstracker.py --closing YYYY-MM-DD --minutes 45
python3 checkresults.py YYYY-MM-DD
python3 calibrate.py
```

`prop_predictions` keeps the entire first priced entry for each
`(date, player, market)`, including its line, direction, price, timestamp,
model version and prediction run ID. Until priced, a proposal can update.
Every prediction run also appends its proposal to `pick_runs`, keyed by
`(run_id, date, player, market)`; retries with the same run ID cannot rewrite it.
`paper_stake` records the proposed amount when a paper bankroll is configured,
and `stake_pct` retains the proposal's fraction. No wagers are placed.
Historical rows without run provenance are left unlabelled.

Closing collection reads event IDs and start times from saved picks, so it
does not fetch the full event feed. It quotes each eligible event once per
invocation, excludes started games and other slate dates, and appends only to
`prop_odds_snapshots.csv`. Entry and closing observations are tagged separately;
an entry observation alone leaves `clv_unmatched_reason="no_closing_snapshot"`.
Old untagged snapshots retain their existing CLV behavior. Grading carries the
frozen entry's `run_id` and `model_version` into `bet_log.csv`.

Example cron entry (replace both absolute paths; create the log directory first):

```cron
*/15 * * * * cd /absolute/path/to/project && /absolute/path/to/python3 oddstracker.py --closing >> /absolute/path/to/closing.log 2>&1
```

This checks every 15 minutes through every game window. Outside the 30-minute
window, or without saved picks, it makes no Odds API requests. Omitting DATE
uses the current Eastern slate date, including when local time is past midnight.
The scheduler host must be awake and its environment must load the repo's `.env`.

Credit estimate: the current request includes eight markets. A populated US
response costs up to 8 credits per event per check; two checks in the 30-minute
window cost up to 16 per picked event (160 for ten games). If US books are empty,
the existing five-region fallback can cost up to 40 credits per populated
response, or 80 per event over two checks. Empty responses do not consume quota;
these are upper estimates because billing depends on returned markets. See the
[Odds API usage rules](https://the-odds-api.com/liveapi/guides/v4/).
Check the job's `run_log.jsonl` stages for `ok`, `market_unavailable`, `skipped`
or `fetch_failed`. Scheduling this job is a manual deployment step.

### 5. (Optional) Train learned model weights

```bash
python3 train_model.py --no-fetch           # writes learned_params.json
```

A learned game model goes live only if it beat the hand-tuned model on validation **and** on the held-out test season, and that test season is the latest completed one. Otherwise the hand-tuned model stays in charge and the run prints why.

---

## Model overview

### Team model (`newnbapredictor.py`)

Combines 10 factors into an expected margin, then converts to win probability via a logistic sigmoid:

1. Net rating differential
2. Home court advantage (venue-adjusted)
3. Injury impact (PPG-weighted)
4. Four Factors edge (eFG%, TOV%, ORB%, FTR)
5. Matchup-specific adjustments
6. Rest & schedule (B2B penalty, days-rest diff, road trip fatigue)
7. Recent form (L10 rolling net rating, EWMA)
8. Pace-talent interaction
9. Head-to-head history (shrinkage-weighted)
10. Clutch performance (close-game adjustment)

The saved training recommendation controls selection automatically. A `swap` recommendation means the learned model beat the baseline on both validation log loss and Brier score; `use_learned` is also accepted for compatibility. `monitor`, `keep_hand_tuned`, and missing or unknown recommendations keep the hand-tuned sigmoid. Learned predictions require a matching input version, all saved features, and valid scaler/coefficient values; otherwise the predictor logs the reason and falls back to the hand-tuned sigmoid. An approved Kalman model takes priority over the static learned model.

`game_features.py` calculates the same inputs for training and live predictions from regular-season game logs. It uses only earlier dates, resets history each season, and supplies the altitude, venue, and star-form inputs. Live prediction downloads each required season's logs once per run. After changing input definitions, rerun training; older weights and Kalman states remain inactive until rebuilt.

`game_odds_tracker.py` fetches the `h2h` market for those NBA events and appends one row per game per observation to `game_odds_tracker.csv`. It keeps event IDs, UTC observation/start times, both teams, paired-book counts, consensus moneyline prices, fair probabilities, vig and power-method k. Prices average decimal payouts; fair probabilities, vig and k average the paired books after de-vigging. Game dates use Eastern time to match the NBA/ESPN calendar.

The combined pipeline automatically loads or fetches quotes for its game date. Quotes older than 30 minutes, started games, incomplete books and invalid probability pairs cannot join. Each team's moneyline diagnostic uses the final team probability: `edge = p_model - p_market`, `EV = p_model * decimal_odds - 1`. Both sides appear in the output; the winner pick's comparison and quote provenance are stored in SQLite. These comparisons do not select bets or change stakes, spread/total picks or rankings. A positive model edge alone does not imply positive EV at the quoted price.

For a manual snapshot, run `python3 game_odds_tracker.py YYYY-MM-DD`. Check the displayed home/away prices against the books and confirm the favorite's fair probability, then inspect the combined pipeline's moneyline diagnostics. `odds_api.py` supplies the shared verified fetcher: if requests fails its TLS handshake, it retries with a standard Python HTTPS client that still authenticates the certificate and hostname. API keys are excluded from error messages.

A live production snapshot on September 18, 2026 recorded the October 20 slate: BOS at DET, PHI at NYK and OKC at SAS, with five paired books each. NYK's consensus price was about -194 against PHI +159, producing a 64.0% fair home probability; the other games were approximately 53% home / 47% away. Reading that real CSV through the combined lookup and joining a controlled 68% home probability produced +4.01 percentage points of edge and +3.04% EV per unit. A -400/+300 test fixture produced 78.24% home probability; a controlled 83% model probability gave +4.76 percentage points of edge and +3.75% EV per unit. Controlled probabilities verify arithmetic and are not game forecasts.

### Props model (`playerlinepredictor.py`)

For each player prop line:
- Projects the stat using playoff + regular-season logs (EWMA, per-minute rates)
- Applies DvP (defense vs position), usage rate, and H2H adjustments
- Blends projections if `learned_params.json` is present (15% learned / 85% existing)
- Anchors to market line (12%) to dampen outlier projections
- Evaluates Over/Under probability via Normal CDF (Poisson regime) or Negative Binomial (overdispersed)
- Applies confidence penalties: B2B, bench minutes, minute volatility, injury status, thin-market book count

SD falls back from the player's own logs to a supported position estimate, then to the existing playoff/regular-season league tables. `project_stat()` threads its resolved position into this SD calculation. Combo props retain empirical or league combo SD because marginal position variances do not include covariance.

The trainer obtains positions with explicit NBA G/F/C filters; unknown or ambiguous players are never automatically labeled guards. It records position support and pools variances toward the league. A split activates only with at least 100 training observations across 10 players, 50 validation observations across 5 players, a 5% SD difference and improved validation Gaussian residual log score. Old unverified artifacts remain inactive. `--no-fetch` does not fetch missing position labels.

The current saved artifact failed this check: Booker/Joe (G), Johnson/Kuminga (F), and Hartenstein/Bitadze (C) all have the same saved PTS SD, 6.086, matching the league value. Position uncertainty stays inactive until training obtains verified position labels and saves supported estimates. After obtaining those labels, rerun `python3 train_model.py --no-bootstrap` to rebuild the artifact without replacing the live calibration.

### Calibration (`calibrate.py`)

Reads historical `bet_log.csv` and computes:
- Per-(market, direction) smoothed hit rates with Laplace smoothing + shrinkage toward prior
- Per-(player, market) projection bias (signed residuals, shrinkage toward 0)

Outputs `calibration.json`, loaded at startup to adjust confidence and projections.

---

## Files produced at runtime

| File | Contents |
|---|---|
| `predictions.db` | Prop and game picks per slate, with odds provenance (gitignored) |
| `bet_log.csv`, `game_log.csv` | Graded props and games, upserted so reruns never duplicate (gitignored) |
| `odds_tracker.csv`, `prop_odds_snapshots.csv` | Consensus lines and per-book quotes for each slate (gitignored) |
| `run_log.jsonl` | One status line per pipeline run (gitignored) |
| `calibration.json` | Market-direction hit rates and diagnostics (tracked) |
| `learned_params.json` | Learned model weights from `train_model.py` (gitignored) |
| `data_cache/` | Cached NBA API responses from `train_model.py` (gitignored) |

---

## Tests

```bash
NBA_SEASON=2025-26 python3 -m pytest -q   # last completed season
NBA_SEASON=2026-27 python3 -m pytest -q   # live season
```

The season decides which fitted artifacts are active: `probability_calibration.json` and `residual_corrections.json` switch on only for seasons that start after their training cutoff. The suite must pass under both settings, and CI runs both. Tests stub every network call and never write to your data files.

---

## Known limitations

- **Hand-tuned weights are the default.** Training saves `swap` only when the learned game model beats the baseline on both validation metrics and on the held-out test season, and the predictor also requires that test season to be the latest completed one.
- **Season rollover.** `config.current_season()` switches seasons every Oct 1; set `NBA_SEASON=YYYY-YY` to backtest another season. Early in a season players have few current-season games, and that regime is not validated yet.
- **Odds API key required.** Copy `.env.example` to `.env` and set `ODDS_API_KEY`. Without it, scripts exit immediately instead of calling the Odds API.
- **LibreSSL on macOS Python 3.9** may produce TLS warnings. `pip install --upgrade certifi` resolves this.
- **No live score integration.** The model is pre-game only; it does not update based on in-game events.
