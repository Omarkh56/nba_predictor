# Minute-history candidate comparison

Selected: **robust** — all played appearances influence expected minutes; extreme short appearances have reduced leverage. Per-minute production rates retain the 15-minute RS / 10-minute PO reliability filters.

| candidate | RS bias | RS MAE | RS test bias | RS test MAE | PO bias | PO MAE | PO test bias | PO test MAE | combined bias | combined RMSE | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minute-filtered-control | +3.16% | 3.819 | +3.66% | 3.915 | +12.12% | 3.654 | +11.13% | 3.602 | +3.81% | 5.545 | 288.1s |
| minute-all-played | -3.78% | 3.694 | -4.15% | 3.798 | +3.24% | 3.339 | +2.57% | 3.312 | -3.27% | 5.482 | 295.7s |
| minute-robust | -1.88% | 3.701 | -2.08% | 3.796 | +5.43% | 3.379 | +4.48% | 3.331 | -1.35% | 5.461 | 356.2s |

## Regular bias by market

| market | filtered control | all played | robust |
|---|---:|---:|---:|
| PTS | +2.6% | -3.9% | -2.1% |
| REB | +3.9% | -4.7% | -2.4% |
| AST | +4.3% | -2.2% | -0.4% |
| 3PM | +11.3% | +4.2% | +6.2% |
| PTS+REB | +2.7% | -4.3% | -2.4% |
| PTS+AST | +2.9% | -3.6% | -1.8% |
| REB+AST | +4.0% | -3.8% | -1.7% |
| PRA | +2.9% | -4.0% | -2.1% |

## Playoffs bias by market

| market | filtered control | all played | robust |
|---|---:|---:|---:|
| PTS | +11.6% | +3.4% | +5.5% |
| REB | +8.1% | -1.7% | +0.7% |
| AST | +21.2% | +10.7% | +13.1% |
| 3PM | +21.1% | +11.6% | +14.0% |
| PTS+REB | +10.4% | +1.8% | +4.0% |
| PTS+AST | +13.4% | +4.8% | +6.9% |
| REB+AST | +12.8% | +2.7% | +5.1% |
| PRA | +11.9% | +3.1% | +5.3% |

All comparisons use the same 182,128 replay rows and frozen chronological splits. The robust candidate improves combined absolute bias from 3.81% to 1.35% and RMSE from 5.545 to 5.461. The all-played candidate has the lowest playoff MAE, but creates a larger regular-season under-projection; robust is the more stable cross-regime choice.
