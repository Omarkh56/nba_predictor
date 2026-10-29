# Deduplicated, graded-only `calibration.json`

Compared with the tracked pre-cleanup artifact at commit `46d11d8`:

| field | before | after |
|---|---:|---:|
| graded bets | 1,163 | 956 |
| overall hit rate | 50.6% | 51.7% |
| market/direction groups | 16 | 16 |
| player/market residual entries | 56 | 474 |
| fitted flag penalties | 24 | 24 |
| PTS OVER sample / hit rate | 113 / 38.9% | 93 / 43.0% |

First, the idempotent migration removed 194 duplicated rows (1,163 → 969).
Then 13 entries labeled `MISS` despite a missing `actual` were moved to
`backups/bet_log.void-or-ungraded.csv` (969 → 956). Local backups are ignored
by Git; no simulated proxy bets went into `bet_log.csv`.

The larger player/market diagnostic count follows the recovered projection
column. v9 deliberately does **not** use selected-bet projection residuals
or hit-rate multipliers to alter live projections; the separate unconditional
replay correction handles that. `calibration.json` retains historic
market-direction confidence and flag diagnostics.
