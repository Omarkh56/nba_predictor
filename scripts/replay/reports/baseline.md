# Replay report: `baseline`

- season: **2025-26**, season types: Regular Season, Playoffs
- code commit: `6cf8be3f2a` on `codex/phase1-bias-fixes` (working tree dirty at run time: False), source fingerprint `292c9135f95e5df8`
- replay config: `replay-cfg-1` (eligibility `elig-2`), cache schema `nba-cache-1`
- replay date range: regular 2025-11-09..2026-04-12; playoffs 2026-04-18..2026-06-13; combined 2025-11-09..2026-06-13
- training cutoff: no fitted residual correction is applied in this run; the model's learned-parameter artifact was trained on ['2020-21', '2021-22', '2022-23']
- sample: **182128 rows** over **22766 player-games** (regular 167072 rows, playoffs 15056 rows, combined 182128 rows)
- market naming: internal stat column `FG3M` is displayed/priced as market `3PM` (same quantity).
- residual = projection - actual; % bias = mean(projection - actual) / mean(actual); P(actual>proj) = share of observations where the actual exceeded the projection.

## Runtime and cache

- cache state: **warm** (entries before run: 34), dir `/Users/omarkhatib/Downloads/advanced databases/advanced databases/project/replay-cache`
- API: 0 requests, 0 retries, 0 failures, 34 cache hits, 0 misses, 0 writes, rate-limit delay 0.0s, backoff delay 0.0s
- runtime: data load 0.2s + replay 282.4s (9 worker(s), 2403.0s cumulative worker time) = **283.8s total**

## Population and exclusions

```
{
 "played_rows": 28465,
 "ineligible_insufficient_prior_games": 5447,
 "ineligible_low_recent_minutes": 0,
 "eligible_player_games": 23018,
 "projection_rows": 182128
}
reconstruction failures: {"project_prop_returned_none": 2016}
bundle: {"fetch_status": {"P/Regular Season": "ok", "T/Regular Season": "ok", "P/Playoffs": "ok", "T/Playoffs": "ok"}, "roster_fetch_status_counts": {"ok": 30}, "raw_player_rows": 28572, "played_rows": 28465, "dnp_rows_dropped": 107, "date_range": {"Playoffs": ["2026-04-18", "2026-06-13"], "Regular Season": ["2025-10-21", "2026-04-12"]}, "rows_by_season_type": {"Regular Season": 26568, "Playoffs": 1897}}
```

## REGULAR: all replayed observations

| market | n | mean actual | mean proj | mean resid | % bias | median resid | MAE | RMSE | P(actual>proj) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PTS | 20884 | 11.223 | 11.511 | +0.288 | +2.6% | +0.90 | 4.975 | 6.414 | 0.438 |
| REB | 20884 | 4.234 | 4.396 | +0.162 | +3.8% | +0.40 | 2.053 | 2.664 | 0.423 |
| AST | 20884 | 2.596 | 2.708 | +0.112 | +4.3% | +0.30 | 1.444 | 1.928 | 0.404 |
| 3PM | 20884 | 1.288 | 1.434 | +0.146 | +11.3% | +0.30 | 0.988 | 1.315 | 0.336 |
| PTS+REB | 20884 | 15.456 | 15.866 | +0.410 | +2.7% | +1.00 | 6.064 | 7.707 | 0.446 |
| PTS+AST | 20884 | 13.818 | 14.214 | +0.395 | +2.9% | +1.00 | 5.533 | 7.086 | 0.441 |
| REB+AST | 20884 | 6.829 | 7.104 | +0.275 | +4.0% | +0.60 | 2.850 | 3.651 | 0.427 |
| PRA | 20884 | 18.052 | 18.569 | +0.517 | +2.9% | +1.10 | 6.644 | 8.425 | 0.446 |
| ALL | 167072 | 9.187 | 9.475 | +0.288 | +3.1% | +0.50 | 3.819 | 5.563 | 0.420 |

Chronological split (pooled over markets):

| split | n | % bias | MAE | RMSE |
|---|---:|---:|---:|---:|
| train | 84464 | +3.8% | 3.795 | 5.529 |
| val | 42096 | +1.3% | 3.774 | 5.490 |
| test | 40512 | +3.6% | 3.915 | 5.708 |

Macro-average of per-market % bias: **+4.3%**

## PLAYOFFS: all replayed observations

| market | n | mean actual | mean proj | mean resid | % bias | median resid | MAE | RMSE | P(actual>proj) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PTS | 1882 | 9.745 | 10.858 | +1.113 | +11.4% | +1.90 | 4.686 | 6.047 | 0.351 |
| REB | 1882 | 3.839 | 4.142 | +0.303 | +7.9% | +0.70 | 1.972 | 2.567 | 0.376 |
| AST | 1882 | 2.108 | 2.550 | +0.442 | +21.0% | +0.60 | 1.355 | 1.857 | 0.327 |
| 3PM | 1882 | 1.079 | 1.303 | +0.224 | +20.8% | +0.30 | 0.907 | 1.203 | 0.283 |
| PTS+REB | 1882 | 13.584 | 14.972 | +1.388 | +10.2% | +2.30 | 5.812 | 7.354 | 0.365 |
| PTS+AST | 1882 | 11.853 | 13.417 | +1.564 | +13.2% | +2.20 | 5.271 | 6.785 | 0.343 |
| REB+AST | 1882 | 5.947 | 6.693 | +0.746 | +12.6% | +1.10 | 2.738 | 3.514 | 0.353 |
| PRA | 1882 | 15.692 | 17.530 | +1.838 | +11.7% | +2.60 | 6.413 | 8.114 | 0.355 |
| ALL | 15056 | 7.981 | 8.933 | +0.952 | +11.9% | +0.90 | 3.644 | 5.319 | 0.344 |

Chronological split (pooled over markets):

| split | n | % bias | MAE | RMSE |
|---|---:|---:|---:|---:|
| train | 7896 | +13.3% | 3.725 | 5.366 |
| val | 3552 | +9.9% | 3.505 | 5.169 |
| test | 3608 | +10.9% | 3.603 | 5.362 |

Macro-average of per-market % bias: **+13.6%**

## COMBINED: all replayed observations

| market | n | mean actual | mean proj | mean resid | % bias | median resid | MAE | RMSE | P(actual>proj) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PTS | 22766 | 11.101 | 11.457 | +0.356 | +3.2% | +1.00 | 4.951 | 6.385 | 0.431 |
| REB | 22766 | 4.201 | 4.375 | +0.174 | +4.1% | +0.40 | 2.046 | 2.656 | 0.419 |
| AST | 22766 | 2.555 | 2.695 | +0.139 | +5.5% | +0.40 | 1.437 | 1.922 | 0.398 |
| 3PM | 22766 | 1.271 | 1.423 | +0.152 | +12.0% | +0.30 | 0.981 | 1.306 | 0.332 |
| PTS+REB | 22766 | 15.302 | 15.792 | +0.491 | +3.2% | +1.20 | 6.043 | 7.678 | 0.439 |
| PTS+AST | 22766 | 13.656 | 14.148 | +0.492 | +3.6% | +1.10 | 5.511 | 7.062 | 0.433 |
| REB+AST | 22766 | 6.756 | 7.070 | +0.314 | +4.6% | +0.70 | 2.841 | 3.640 | 0.421 |
| PRA | 22766 | 17.857 | 18.483 | +0.626 | +3.5% | +1.20 | 6.625 | 8.400 | 0.438 |
| ALL | 182128 | 9.087 | 9.430 | +0.343 | +3.8% | +0.50 | 3.804 | 5.544 | 0.414 |

Chronological split (pooled over markets):

| split | n | % bias | MAE | RMSE |
|---|---:|---:|---:|---:|
| train | 92360 | +4.5% | 3.789 | 5.516 |
| val | 45648 | +1.9% | 3.753 | 5.466 |
| test | 44120 | +4.2% | 3.890 | 5.681 |

Macro-average of per-market % bias: **+5.0%**

## By pre-game role stratum (rolling minutes of the last 10 played games; reporting only, never an exclusion)

low = RS < 15 / PO < 10; core >= 20; mid = in between

### REGULAR

| stratum | player-games | rows | % bias | MAE | RMSE | PTS | REB | AST | 3PM | PTS+REB | PTS+AST | REB+AST | PRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 3784 | 30272 | +33.4% | 3.385 | 4.739 | +31% | +35% | +42% | +60% | +31% | +32% | +37% | +32% |
| mid | 3337 | 26696 | +0.2% | 3.326 | 4.818 | +1% | -1% | +3% | +18% | -1% | +1% | +0% | -1% |
| core | 13763 | 110104 | +0.6% | 4.058 | 5.930 | +0% | +1% | +1% | +6% | +0% | +1% | +1% | +1% |

### PLAYOFFS

| stratum | player-games | rows | % bias | MAE | RMSE | PTS | REB | AST | 3PM | PTS+REB | PTS+AST | REB+AST | PRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 202 | 1616 | +116.1% | 2.880 | 3.936 | +99% | +119% | +208% | +120% | +104% | +115% | +147% | +115% |
| mid | 597 | 4776 | +41.0% | 3.141 | 4.527 | +39% | +35% | +69% | +53% | +37% | +44% | +44% | +40% |
| core | 1083 | 8664 | +3.7% | 4.064 | 5.915 | +4% | -2% | +9% | +10% | +3% | +5% | +2% | +4% |

### COMBINED

| stratum | player-games | rows | % bias | MAE | RMSE | PTS | REB | AST | 3PM | PTS+REB | PTS+AST | REB+AST | PRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 3986 | 31888 | +35.5% | 3.359 | 4.702 | +33% | +37% | +45% | +62% | +33% | +35% | +40% | +34% |
| mid | 3934 | 31472 | +4.1% | 3.298 | 4.775 | +4% | +2% | +8% | +21% | +3% | +5% | +4% | +3% |
| core | 14846 | 118768 | +0.8% | 4.058 | 5.929 | +1% | +1% | +2% | +6% | +0% | +1% | +1% | +1% |

