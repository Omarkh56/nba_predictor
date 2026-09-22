# Replay report: `market-edge-postcommit`

- season: **2025-26**, season types: Regular Season, Playoffs
- code commit: `29670c82ef` on `codex/phase1-bias-fixes` (working tree dirty at run time: False), source fingerprint `3c1e3add41b9b668`
- replay config: `replay-cfg-1` (eligibility `elig-2`), cache schema `nba-cache-1`
- replay date range: regular 2025-11-09..2026-04-12; playoffs 2026-04-18..2026-06-13; combined 2025-11-09..2026-06-13
- training cutoff: no fitted residual correction is applied in this run; the model's learned-parameter artifact was trained on ['2020-21', '2021-22', '2022-23']
- sample: **182128 rows** over **22766 player-games** (regular 167072 rows, playoffs 15056 rows, combined 182128 rows)
- market naming: internal stat column `FG3M` is displayed/priced as market `3PM` (same quantity).
- residual = projection - actual; % bias = mean(projection - actual) / mean(actual); P(actual>proj) = share of observations where the actual exceeded the projection.

## Runtime and cache

- cache state: **warm** (entries before run: 34), dir `/Users/omarkhatib/Downloads/advanced databases/advanced databases/project/replay-cache`
- API: 0 requests, 0 retries, 0 failures, 34 cache hits, 0 misses, 0 writes, rate-limit delay 0.0s, backoff delay 0.0s
- runtime: data load 0.2s + replay 243.1s (9 worker(s), 2004.0s cumulative worker time) = **244.7s total**

## Population and exclusions

```
{
 "played_rows": 28465,
 "ineligible_insufficient_prior_games": 5447,
 "ineligible_low_recent_minutes": 0,
 "eligible_player_games": 23018,
 "projection_rows": 182128,
 "dnp_void_player_games": 0
}
reconstruction failures: {"project_prop_returned_none": 2016}
bundle: {"fetch_status": {"P/Regular Season": "ok", "T/Regular Season": "ok", "P/Playoffs": "ok", "T/Playoffs": "ok"}, "roster_fetch_status_counts": {"ok": 30}, "raw_player_rows": 28572, "played_rows": 28465, "dnp_rows_dropped": 107, "date_range": {"Playoffs": ["2026-04-18", "2026-06-13"], "Regular Season": ["2025-10-21", "2026-04-12"]}, "rows_by_season_type": {"Regular Season": 26568, "Playoffs": 1897}}
```

## REGULAR: all replayed observations

| market | n | mean actual | mean proj | mean resid | % bias | median resid | MAE | RMSE | P(actual>proj) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PTS | 20884 | 11.223 | 12.541 | +1.318 | +11.7% | +1.70 | 4.972 | 6.383 | 0.376 |
| REB | 20884 | 4.234 | 4.765 | +0.531 | +12.5% | +0.80 | 2.045 | 2.624 | 0.368 |
| AST | 20884 | 2.596 | 2.940 | +0.344 | +13.3% | +0.50 | 1.457 | 1.921 | 0.361 |
| 3PM | 20884 | 1.288 | 1.543 | +0.255 | +19.8% | +0.30 | 0.999 | 1.316 | 0.308 |
| PTS+REB | 20884 | 15.456 | 17.263 | +1.807 | +11.7% | +2.30 | 6.012 | 7.626 | 0.374 |
| PTS+AST | 20884 | 13.818 | 15.476 | +1.658 | +12.0% | +2.00 | 5.510 | 7.049 | 0.369 |
| REB+AST | 20884 | 6.829 | 7.704 | +0.875 | +12.8% | +1.20 | 2.840 | 3.602 | 0.357 |
| PRA | 20884 | 18.052 | 20.198 | +2.147 | +11.9% | +2.60 | 6.577 | 8.337 | 0.367 |
| ALL | 167072 | 9.187 | 10.304 | +1.117 | +12.2% | +0.90 | 3.801 | 5.516 | 0.360 |

Chronological split (pooled over markets):

| split | n | % bias | MAE | RMSE |
|---|---:|---:|---:|---:|
| train | 84464 | +13.3% | 3.787 | 5.491 |
| val | 42096 | +12.1% | 3.760 | 5.444 |
| test | 40512 | +9.8% | 3.874 | 5.640 |

Macro-average of per-market % bias: **+13.2%**

## PLAYOFFS: all replayed observations

| market | n | mean actual | mean proj | mean resid | % bias | median resid | MAE | RMSE | P(actual>proj) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PTS | 1882 | 9.745 | 11.419 | +1.674 | +17.2% | +2.10 | 4.636 | 6.029 | 0.327 |
| REB | 1882 | 3.839 | 4.347 | +0.508 | +13.2% | +0.70 | 1.937 | 2.544 | 0.359 |
| AST | 1882 | 2.108 | 2.642 | +0.534 | +25.3% | +0.60 | 1.335 | 1.775 | 0.301 |
| 3PM | 1882 | 1.079 | 1.347 | +0.268 | +24.9% | +0.30 | 0.911 | 1.205 | 0.275 |
| PTS+REB | 1882 | 13.584 | 15.744 | +2.160 | +15.9% | +2.70 | 5.733 | 7.310 | 0.334 |
| PTS+AST | 1882 | 11.853 | 14.070 | +2.217 | +18.7% | +2.50 | 5.190 | 6.706 | 0.309 |
| REB+AST | 1882 | 5.947 | 6.988 | +1.041 | +17.5% | +1.20 | 2.666 | 3.414 | 0.332 |
| PRA | 1882 | 15.692 | 18.396 | +2.704 | +17.2% | +3.10 | 6.285 | 8.009 | 0.318 |
| ALL | 15056 | 7.981 | 9.369 | +1.388 | +17.4% | +1.00 | 3.586 | 5.263 | 0.319 |

Chronological split (pooled over markets):

| split | n | % bias | MAE | RMSE |
|---|---:|---:|---:|---:|
| train | 7896 | +18.6% | 3.681 | 5.332 |
| val | 3552 | +14.4% | 3.432 | 5.119 |
| test | 3608 | +17.8% | 3.531 | 5.254 |

Macro-average of per-market % bias: **+18.7%**

## COMBINED: all replayed observations

| market | n | mean actual | mean proj | mean resid | % bias | median resid | MAE | RMSE | P(actual>proj) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PTS | 22766 | 11.101 | 12.448 | +1.348 | +12.1% | +1.80 | 4.944 | 6.355 | 0.372 |
| REB | 22766 | 4.201 | 4.730 | +0.529 | +12.6% | +0.80 | 2.036 | 2.618 | 0.367 |
| AST | 22766 | 2.555 | 2.915 | +0.360 | +14.1% | +0.50 | 1.447 | 1.910 | 0.356 |
| 3PM | 22766 | 1.271 | 1.527 | +0.256 | +20.1% | +0.30 | 0.992 | 1.307 | 0.305 |
| PTS+REB | 22766 | 15.302 | 17.138 | +1.836 | +12.0% | +2.30 | 5.989 | 7.600 | 0.370 |
| PTS+AST | 22766 | 13.656 | 15.360 | +1.704 | +12.5% | +2.10 | 5.484 | 7.021 | 0.364 |
| REB+AST | 22766 | 6.756 | 7.645 | +0.889 | +13.2% | +1.20 | 2.825 | 3.586 | 0.355 |
| PRA | 22766 | 17.857 | 20.049 | +2.193 | +12.3% | +2.70 | 6.553 | 8.311 | 0.363 |
| ALL | 182128 | 9.087 | 10.227 | +1.139 | +12.5% | +0.90 | 3.784 | 5.495 | 0.357 |

Chronological split (pooled over markets):

| split | n | % bias | MAE | RMSE |
|---|---:|---:|---:|---:|
| train | 92360 | +13.7% | 3.778 | 5.478 |
| val | 45648 | +12.3% | 3.734 | 5.419 |
| test | 44120 | +10.4% | 3.846 | 5.609 |

Macro-average of per-market % bias: **+13.6%**

## By pre-game role stratum (rolling minutes of the last 10 played games; reporting only, never an exclusion)

low = RS < 15 / PO < 10; core >= 20; mid = in between

### REGULAR

| stratum | player-games | rows | % bias | MAE | RMSE | PTS | REB | AST | 3PM | PTS+REB | PTS+AST | REB+AST | PRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 3784 | 30272 | +19.5% | 3.045 | 4.367 | +18% | +21% | +25% | +46% | +18% | +19% | +22% | +18% |
| mid | 3337 | 26696 | +13.2% | 3.409 | 4.841 | +14% | +12% | +16% | +30% | +12% | +13% | +13% | +12% |
| core | 13763 | 110104 | +11.3% | 4.104 | 5.936 | +11% | +12% | +12% | +16% | +11% | +11% | +12% | +11% |

### PLAYOFFS

| stratum | player-games | rows | % bias | MAE | RMSE | PTS | REB | AST | 3PM | PTS+REB | PTS+AST | REB+AST | PRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 202 | 1616 | +47.5% | 2.151 | 2.989 | +36% | +47% | +116% | +66% | +38% | +47% | +69% | +46% |
| mid | 597 | 4776 | +39.2% | 2.993 | 4.243 | +39% | +35% | +57% | +54% | +36% | +41% | +41% | +38% |
| core | 1083 | 8664 | +12.7% | 4.181 | 6.046 | +13% | +7% | +18% | +17% | +12% | +14% | +11% | +13% |

### COMBINED

| stratum | player-games | rows | % bias | MAE | RMSE | PTS | REB | AST | 3PM | PTS+REB | PTS+AST | REB+AST | PRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 3986 | 31888 | +20.1% | 3.000 | 4.308 | +18% | +21% | +27% | +47% | +18% | +19% | +23% | +19% |
| mid | 3934 | 31472 | +15.6% | 3.346 | 4.755 | +16% | +14% | +19% | +33% | +14% | +16% | +16% | +15% |
| core | 14846 | 118768 | +11.4% | 4.110 | 5.944 | +11% | +11% | +12% | +16% | +11% | +11% | +12% | +11% |

## Change from previous run `probability-postcommit`

### regular (paired rows 167072 of 167072 previous / 167072 current)

| market | Δ % bias (pts) | Δ MAE | Δ RMSE |
|---|---:|---:|---:|
| PTS | +0.00 | +0.0000 | +0.0000 |
| REB | +0.00 | +0.0000 | +0.0000 |
| AST | +0.00 | +0.0000 | +0.0000 |
| 3PM | +0.00 | +0.0000 | +0.0000 |
| PTS+REB | +0.00 | +0.0000 | +0.0000 |
| PTS+AST | +0.00 | +0.0000 | +0.0000 |
| REB+AST | +0.00 | +0.0000 | +0.0000 |
| PRA | +0.00 | +0.0000 | +0.0000 |
| ALL | +0.00 | +0.0000 | +0.0000 |

Paired bootstrap over game dates (all markets, negative = current is better): abs_bias +0.0000 [+0.0000, +0.0000] (n.s.); mae +0.0000 [+0.0000, +0.0000] (n.s.); rmse +0.0000 [+0.0000, +0.0000] (n.s.)

### playoffs (paired rows 15056 of 15056 previous / 15056 current)

| market | Δ % bias (pts) | Δ MAE | Δ RMSE |
|---|---:|---:|---:|
| PTS | +0.00 | +0.0000 | +0.0000 |
| REB | +0.00 | +0.0000 | +0.0000 |
| AST | +0.00 | +0.0000 | +0.0000 |
| 3PM | +0.00 | +0.0000 | +0.0000 |
| PTS+REB | +0.00 | +0.0000 | +0.0000 |
| PTS+AST | +0.00 | +0.0000 | +0.0000 |
| REB+AST | +0.00 | +0.0000 | +0.0000 |
| PRA | +0.00 | +0.0000 | +0.0000 |
| ALL | +0.00 | +0.0000 | +0.0000 |

Paired bootstrap over game dates (all markets, negative = current is better): abs_bias +0.0000 [+0.0000, +0.0000] (n.s.); mae +0.0000 [+0.0000, +0.0000] (n.s.); rmse +0.0000 [+0.0000, +0.0000] (n.s.)

### combined (paired rows 182128 of 182128 previous / 182128 current)

| market | Δ % bias (pts) | Δ MAE | Δ RMSE |
|---|---:|---:|---:|
| PTS | +0.00 | +0.0000 | +0.0000 |
| REB | +0.00 | +0.0000 | +0.0000 |
| AST | +0.00 | +0.0000 | +0.0000 |
| 3PM | +0.00 | +0.0000 | +0.0000 |
| PTS+REB | +0.00 | +0.0000 | +0.0000 |
| PTS+AST | +0.00 | +0.0000 | +0.0000 |
| REB+AST | +0.00 | +0.0000 | +0.0000 |
| PRA | +0.00 | +0.0000 | +0.0000 |
| ALL | +0.00 | +0.0000 | +0.0000 |

Paired bootstrap over game dates (all markets, negative = current is better): abs_bias +0.0000 [+0.0000, +0.0000] (n.s.); mae +0.0000 [+0.0000, +0.0000] (n.s.); rmse +0.0000 [+0.0000, +0.0000] (n.s.)

