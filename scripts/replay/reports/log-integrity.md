# Bet-log integrity migration

- Source: `bet_log_backfilled.csv` → local canonical `bet_log.csv`
- Rows: **1163 → 969**
- Duplicate groups/rows removed: **194 / 194**
- Conflicting populated fields: **0**
- Projection coverage: **969/969 (100%)**
- p_model coverage: **969/969 (100%)**
- Picks: {'OVER': 777, 'UNDER': 192}; results: {'HIT': 494, 'MISS': 475}
- Calibration diagnostic only: 1163 → 969 bets; hit rate 0.506 → 0.51
- The tracked `calibration.json` was restored; final post-fix probability calibration belongs to Commit 7.

## Market-direction diagnostic

| key | old n | new n | old raw | new raw | old conf | new conf |
|---|---:|---:|---:|---:|---:|---:|
| 3PM_OVER | 95 | 81 | 0.537 | 0.543 | 2.7 | 2.9 |
| 3PM_UNDER | 12 | 12 | 0.25 | 0.25 | -11.4 | -11.5 |
| AST_OVER | 204 | 169 | 0.534 | 0.533 | 2.7 | 2.1 |
| AST_UNDER | 77 | 66 | 0.662 | 0.667 | 13.3 | 13.0 |
| PRA_OVER | 71 | 64 | 0.423 | 0.422 | -7.0 | -7.2 |
| PRA_UNDER | 5 | 5 | 0.4 | 0.4 | -2.4 | -2.5 |
| PTS+AST_OVER | 106 | 85 | 0.5 | 0.494 | -0.5 | -1.4 |
| PTS+AST_UNDER | 9 | 7 | 0.778 | 0.714 | 10.0 | 6.2 |
| PTS+REB_OVER | 20 | 16 | 0.3 | 0.375 | -12.0 | -7.0 |
| PTS+REB_UNDER | 7 | 7 | 0.286 | 0.286 | -6.7 | -6.8 |
| PTS_OVER | 113 | 93 | 0.389 | 0.43 | -10.4 | -6.9 |
| PTS_UNDER | 36 | 31 | 0.472 | 0.484 | -2.4 | -1.8 |
| REB+AST_OVER | 63 | 55 | 0.444 | 0.418 | -5.0 | -7.3 |
| REB+AST_UNDER | 22 | 19 | 0.682 | 0.684 | 10.7 | 9.9 |
| REB_OVER | 271 | 214 | 0.517 | 0.523 | 1.0 | 1.3 |
| REB_UNDER | 52 | 45 | 0.577 | 0.578 | 5.6 | 5.2 |
