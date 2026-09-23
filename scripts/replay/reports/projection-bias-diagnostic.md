# Projection bias diagnostic

Final chronological holdout from the 2025-26 replay. Confidence intervals
resample whole game dates. Lower MAE is better; positive bias means the model
projects too high.

## Answer

- Raw projections are high: regular season +9.82% and playoffs +17.77%.
- The v9 chronological correction reduces this to regular season -2.27% and playoffs +4.22%.
- The regular-season corrected 95% interval is -5.03% to +0.17%; zero is plausible.
- The playoff corrected interval is +1.22% to +7.12%; the remaining upward bias is unlikely to be random in this holdout.

| regime | rows | raw bias | corrected bias | corrected MAE |
|---|---:|---:|---:|---:|
| Regular Season | 40,512 | +9.82% | -2.27% | 3.718 |
| Playoffs | 3,608 | +17.77% | +4.22% | 3.286 |

## Corrected bias by market

| regime | market | bias | MAE |
|---|---|---:|---:|
| Regular Season | 3PM | -1.51% | 0.920 |
| Regular Season | AST | -2.64% | 1.433 |
| Regular Season | PRA | -2.31% | 6.457 |
| Regular Season | PTS | -3.01% | 4.923 |
| Regular Season | PTS+AST | -2.97% | 5.475 |
| Regular Season | PTS+REB | -2.22% | 5.890 |
| Regular Season | REB | +0.09% | 1.940 |
| Regular Season | REB+AST | -0.98% | 2.707 |
| Playoffs | 3PM | -1.12% | 0.867 |
| Playoffs | AST | +6.69% | 1.180 |
| Playoffs | PRA | +4.43% | 5.689 |
| Playoffs | PTS | +5.36% | 4.354 |
| Playoffs | PTS+AST | +5.70% | 4.730 |
| Playoffs | PTS+REB | +4.00% | 5.335 |
| Playoffs | REB | +0.05% | 1.766 |
| Playoffs | REB+AST | +2.30% | 2.366 |

## Where the remaining error sits

The corrected model is too low in the lowest projection quartile and too high
in the highest quartile. That pattern explains why multiplicative correction
continues to deserve testing. Minutes misses are even more important: when
projected minutes land within two minutes of actual minutes, final-test bias is
close to zero. Actual minutes are used only to diagnose the error after the game.

| regime | projection level | bias | MAE |
|---|---|---:|---:|
| Regular Season | low | -23.34% | 2.810 |
| Regular Season | mid-low | -10.19% | 3.561 |
| Regular Season | mid-high | -2.30% | 3.875 |
| Regular Season | high | +7.22% | 4.627 |
| Playoffs | low | -25.81% | 1.780 |
| Playoffs | mid-low | -3.63% | 2.775 |
| Playoffs | mid-high | -2.30% | 3.738 |
| Playoffs | high | +15.08% | 4.856 |

| regime | minutes error | rows | bias | MAE |
|---|---|---:|---:|---:|
| Regular Season | over by >5 | 9,344 | +64.07% | 3.773 |
| Regular Season | within 2 | 10,160 | -1.54% | 3.107 |
| Regular Season | over by 2-5 | 6,856 | +15.21% | 2.981 |
| Regular Season | under by 2-5 | 6,216 | -14.07% | 3.620 |
| Regular Season | under by >5 | 7,936 | -34.66% | 5.150 |
| Playoffs | under by >5 | 632 | -21.90% | 4.650 |
| Playoffs | under by 2-5 | 592 | -5.03% | 3.006 |
| Playoffs | within 2 | 904 | +0.49% | 2.727 |
| Playoffs | over by >5 | 880 | +58.23% | 3.716 |
| Playoffs | over by 2-5 | 600 | +18.66% | 2.337 |

## Improvement experiment

Validation selected **multiplicative**, k=250, 1 projection bucket(s), fitted separately by regime and market.
Its validation objective was 6.810, versus 6.875 for the current correction.

| regime | current test bias | candidate test bias | current MAE | candidate MAE |
|---|---:|---:|---:|---:|
| Regular Season | -2.27% | -2.54% | 3.718 | 3.688 |
| Playoffs | +4.22% | +3.00% | 3.286 | 3.233 |

The candidate is not deployed. Playoff fit uses the current postseason's early outcomes; only one unseen postseason is available, so stability and playoff cold-start behavior are unproven.
The next model change should improve role/minutes forecasting and be validated
on another unseen season or postseason before replacing the current correction.
