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

If `learned_params.json` exists and `USE_LEARNED_GAME_MODEL = True`, a logistic regression model trained by `train_model.py` replaces the hand-tuned sigmoid.

### Props model (`playerlinepredictor.py`)

For each player prop line:
- Projects the stat using playoff + regular-season logs (EWMA, per-minute rates)
- Applies DvP (defense vs position), usage rate, and H2H adjustments
- Blends projections if `learned_params.json` is present (15% learned / 85% existing)
- Anchors to market line (12%) to dampen outlier projections
- Evaluates Over/Under probability via Normal CDF (Poisson regime) or Negative Binomial (overdispersed)
- Applies confidence penalties: B2B, bench minutes, minute volatility, injury status, thin-market book count

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

- **Hand-tuned weights are the default.** `learned_params.json` is only active when `USE_LEARNED_GAME_MODEL = True` in `newnbapredictor.py`. The learned model requires a full backtest run via `train_model.py` first.
- **2026-27 season.** Season constants are hardcoded; update `SEASON` in `nba_combined.py` and `playerlinepredictor.py` when the season rolls over.
- **Odds API key required.** Copy `.env.example` to `.env` and set `ODDS_API_KEY`. Without it, scripts exit immediately instead of calling the Odds API.
- **LibreSSL on macOS Python 3.9** may produce TLS warnings. `pip install --upgrade certifi` resolves this.
- **No live score integration.** The model is pre-game only; it does not update based on in-game events.
