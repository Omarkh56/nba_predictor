# Phase 3 minutes forecast gate

**Decision: leave `MINUTES_FORECAST_ENABLED = False`.** The gate fails in both playoff projection-MAE checks and both held-out bias checks.

| Evaluation | Regime | Player-games paired | Minutes MAE control → candidate | Projection MAE control → candidate | Macro bias control → candidate |
|---|---|---:|---:|---:|---:|
| validation 2024-25 | Playoffs | 1774 | 5.3723 → 5.2641 | 3.8386 → 3.8577 | +20.51% → +21.97% |
| validation 2024-25 | Regular Season | 24131 | 4.9873 → 4.9788 | 3.8560 → 3.7591 | +11.73% → +8.41% |
| heldout 2025-26 | Playoffs | 1887 | 5.3134 → 5.1401 | 3.5836 → 3.5903 | +18.84% → +19.91% |
| heldout 2025-26 | Regular Season | 24729 | 4.9456 → 4.9316 | 3.8082 → 3.7072 | +12.49% → +9.19% |

## First three weeks of 2025–26

2025-10-21 through 2025-11-10, fewer than ten prior games, 2474 paired player-games: minutes MAE **4.5800 → 4.5955**, projection MAE **3.9794 → 3.7456**, macro bias **+7.97% → +7.80%**.

The early-season projection improvement does not meet the full gate; early-season minutes MAE slightly worsens.

## Projection bias by signed minutes-error bucket

Bucket error is forecast minutes minus actual minutes. Bias here is pooled (projection − actual) / actual, with regular season and playoffs kept separate.

| Evaluation | Regime | Minutes error | Candidate rows | Control bias | Candidate bias |
|---|---|---|---:|---:|---:|
| 2024-25 | Playoffs | <-10 | 424 | -34.80% | -37.52% |
| 2024-25 | Playoffs | -10:-5 | 1656 | -12.69% | -12.17% |
| 2024-25 | Playoffs | -5:-2 | 2216 | +0.78% | -1.86% |
| 2024-25 | Playoffs | -2:2 | 3472 | +8.89% | +12.28% |
| 2024-25 | Playoffs | 2:5 | 2264 | +43.44% | +35.79% |
| 2024-25 | Playoffs | 5:10 | 2848 | +77.04% | +76.89% |
| 2024-25 | Playoffs | >10 | 1312 | +230.50% | +168.11% |
| 2024-25 | Regular Season | <-10 | 13240 | -38.87% | -41.63% |
| 2024-25 | Regular Season | -10:-5 | 28848 | -17.98% | -18.59% |
| 2024-25 | Regular Season | -5:-2 | 34072 | -4.57% | -4.52% |
| 2024-25 | Regular Season | -2:2 | 52264 | +9.74% | +10.40% |
| 2024-25 | Regular Season | 2:5 | 30104 | +32.20% | +33.91% |
| 2024-25 | Regular Season | 5:10 | 26024 | +68.90% | +68.23% |
| 2024-25 | Regular Season | >10 | 8496 | +195.28% | +173.41% |
| 2025-26 | Playoffs | <-10 | 616 | -30.40% | -32.28% |
| 2025-26 | Playoffs | -10:-5 | 1824 | -15.18% | -15.35% |
| 2025-26 | Playoffs | -5:-2 | 2264 | +1.37% | +3.16% |
| 2025-26 | Playoffs | -2:2 | 3792 | +15.63% | +14.88% |
| 2025-26 | Playoffs | 2:5 | 2640 | +35.55% | +35.51% |
| 2025-26 | Playoffs | 5:10 | 2728 | +80.13% | +75.72% |
| 2025-26 | Playoffs | >10 | 1232 | +233.45% | +207.92% |
| 2025-26 | Regular Season | <-10 | 12608 | -39.88% | -42.37% |
| 2025-26 | Regular Season | -10:-5 | 28792 | -17.79% | -18.30% |
| 2025-26 | Regular Season | -5:-2 | 35608 | -3.94% | -3.86% |
| 2025-26 | Regular Season | -2:2 | 53448 | +9.85% | +10.48% |
| 2025-26 | Regular Season | 2:5 | 31120 | +31.19% | +31.72% |
| 2025-26 | Regular Season | 5:10 | 26872 | +66.41% | +68.03% |
| 2025-26 | Regular Season | >10 | 9384 | +194.59% | +176.04% |

## Coverage and interpretation

- 2024-25: 207240 shared projection rows; 5320 candidate-only rows; 0 control-only rows. Gate comparisons use shared rows only.
- 2025-26: 212928 shared projection rows; 4752 candidate-only rows; 0 control-only rows. Gate comparisons use shared rows only.

Historical lineups are unavailable: the replay uses a recorded minutes-based starter proxy. Historical spreads are zero, so blowout effects have unit-test evidence only. Scheduled-game gaps are not confirmed injuries or roster membership. The paired population is conditional on playing and at least one current-season appearance; this is not prospective price or betting evidence.

## Reproduction and provenance

```sh
python3 scripts/replay/replay_props.py run --label phase3-control-202425 --season 2024-25 --min-prior-games 1 --workers 4 --cache-only --skip-rosters --cache-dir /tmp/nba-phase3-20261005/cache --out-dir /tmp/nba-phase3-20261005/output --reports-dir /tmp/nba-phase3-20261005/reports --set MINUTES_FORECAST_ENABLED=False
python3 scripts/replay/replay_props.py run --label phase3-minutes-202425 --compare phase3-control-202425 --season 2024-25 --min-prior-games 1 --workers 4 --cache-only --skip-rosters --cache-dir /tmp/nba-phase3-20261005/cache --out-dir /tmp/nba-phase3-20261005/output --reports-dir /tmp/nba-phase3-20261005/reports --set MINUTES_FORECAST_ENABLED=True
```

```sh
python3 scripts/replay/replay_props.py run --label phase3-control-202526 --season 2025-26 --min-prior-games 1 --workers 4 --cache-only --cache-dir /tmp/nba-phase3-20261005/cache --out-dir /tmp/nba-phase3-20261005/output --reports-dir /tmp/nba-phase3-20261005/reports --set MINUTES_FORECAST_ENABLED=False
python3 scripts/replay/replay_props.py run --label phase3-minutes-202526 --compare phase3-control-202526 --season 2025-26 --min-prior-games 1 --workers 4 --cache-only --cache-dir /tmp/nba-phase3-20261005/cache --out-dir /tmp/nba-phase3-20261005/output --reports-dir /tmp/nba-phase3-20261005/reports --set MINUTES_FORECAST_ENABLED=True
```

Forecast coefficients fixed before either evaluation; held-out results were not used to tune.
2024-25 and 2025-26 are separate full chronological evaluation seasons, not train/val/test rows pooled together.
Opening games without a current-season appearance are excluded from the paired replay population.
The first preliminary validation baseline lacked learned_params.json and is discarded; official controls use the deployed artifact.
Both candidate model runs finished and saved all rows; initial report step failed on game_id object/int type mismatch.
Candidate reports rebuilt offline from saved rows after fixing ID parsing; original runtime counters unavailable for these report-only rebuilds.
Later CLI/report fixes do not change any evaluated projection formula.
Validation skipped roster lookups (manual positions only); heldout used cached rosters, identical within each season pair.
The current deployed correction is disabled in both raw replay variants; its reported -2.3%/+4.2% biases are fixed benchmarks.
The learned artifact was trained on 2020-21 through 2022-23, validated on 2023-24, originally tested on 2024-25; 2024-25 is now reused as validation, 2025-26 is held out.

The adjacent JSON records input/output hashes, all gate checks, exact-market biases, feature coverage, and first-three-week buckets. Per-row gzip outputs are retained in `/tmp/nba-phase3-20261005/output`; they are not committed.

No production defaults, stakes, edge thresholds, or Kelly settings changed.
