# Confirmed-injury ramp replay

The old calendar-gap heuristic discounted 13,924/20,884 regular-season player-games and 1,028/1,882 playoff player-games (a player-game counted once via the PTS row). Its mean active discount was ~16%. It is now disabled in the absence of both missed scheduled team games and historical injury-status evidence. The counting function includes played games across RS + PO if evidence is supplied.

| regime | prior bias | after bias | prior MAE | after MAE | prior RMSE | after RMSE |
|---|---:|---:|---:|---:|---:|---:|
| regular | -1.88% | +12.16% | 3.701 | 3.801 | 5.495 | 5.516 |
| playoffs | +5.43% | +17.40% | 3.379 | 3.586 | 5.068 | 5.263 |
| combined | -1.35% | +12.54% | 3.674 | 3.784 | 5.461 | 5.495 |

## Regular by market

| market | prior bias | after bias |
|---|---:|---:|
| PTS | -2.1% | +11.7% |
| REB | -2.4% | +12.5% |
| AST | -0.4% | +13.3% |
| 3PM | +6.2% | +19.8% |
| PTS+REB | -2.4% | +11.7% |
| PTS+AST | -1.8% | +12.0% |
| REB+AST | -1.7% | +12.8% |
| PRA | -2.1% | +11.9% |

## Playoffs by market

| market | prior bias | after bias |
|---|---:|---:|
| PTS | +5.5% | +17.2% |
| REB | +0.7% | +13.2% |
| AST | +13.1% | +25.3% |
| 3PM | +14.0% | +24.9% |
| PTS+REB | +4.0% | +15.9% |
| PTS+AST | +6.9% | +18.7% |
| REB+AST | +5.1% | +17.5% |
| PRA | +5.3% | +17.2% |

The rise is expected: a false injury discount had masked an upward projection error. Future evidence-backed activation requires a player-specific as-of team schedule and historical injury dates; current ESPN status is not proof of a prior absence.
