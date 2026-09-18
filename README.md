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

```bash
pip install -r requirements.txt
```

If you see TLS errors on macOS (LibreSSL), run:

```bash
pip install --upgrade certifi
```

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

### 3. Run the full pipeline

```bash
python3 nba_combined.py           # today's games
python3 nba_combined.py --verbose # with per-factor breakdown
```

### 4. Grade results and recalibrate (after games)

```bash
# Append graded bets to bet_log.csv with: checkresults.py (manual grading)
python3 calibrate.py --summary    # recompute calibration.json
```

### 5. (Optional) Train learned model weights

```bash
python3 train_model.py            # writes learned_params.json
```

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
| `calibration.json` | Market-direction hit rates + player-market bias (gitignored) |
| `learned_params.json` | Logistic/ridge regression weights from `train_model.py` (gitignored) |
| `predictions_YYYY-MM-DD.json` | Top prop picks exported after each run (gitignored) |
| `game_predictions_YYYY-MM-DD.json` | Team model picks exported after each run (gitignored) |
| `bet_log.csv` | Graded bet history for calibration (gitignored) |
| `data_cache/` | Cached NBA API responses from `train_model.py` (gitignored) |

---

## Tests

```bash
python3 -m pytest tests/ -v
```

72 tests covering pure functions in `calibrate.py`, `oddstracker.py`, `playerlinepredictor.py`, and `newnbapredictor.py`. Tests stub all NBA API network calls so they run offline.

CI runs automatically on every push via `.github/workflows/ci.yml`.

---

## Known limitations

- **Hand-tuned weights are the default.** Training saves a `swap` recommendation only when the learned game model beats the baseline on both validation metrics. The predictor applies that recommendation automatically when the required inputs are available; no manual source-code toggle is needed.
- **2026-27 season.** Season constants are hardcoded; update `SEASON` in `nba_combined.py` and `playerlinepredictor.py` when the season rolls over.
- **Odds API key required.** Copy `.env.example` to `.env` and set `ODDS_API_KEY`. Without it, scripts exit immediately instead of calling the Odds API.
- **LibreSSL on macOS Python 3.9** may produce TLS warnings. `pip install --upgrade certifi` resolves this.
- **No live score integration.** The model is pre-game only; it does not update based on in-game events.
