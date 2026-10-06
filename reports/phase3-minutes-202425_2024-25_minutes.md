# Minutes replay: phase3-minutes-202425 / 2024-25

Minutes MAE is counted once per player-game; bias is macro averaged over markets.

| Regime | Model | Player-games | Minutes MAE | Bias | Projection MAE |
|---|---|---:|---:|---:|---:|
| Playoffs | candidate | 1774 | 5.2641 | +21.9657% | 3.8577 |
| Playoffs | baseline | 1774 | 5.3723 | +20.5088% | 3.8386 |
| Regular Season | candidate | 24131 | 4.9788 | +8.4056% | 3.7591 |
| Regular Season | baseline | 24131 | 4.9873 | +11.7301% | 3.8560 |

Full minutes-error buckets and early-season results are in the adjacent JSON.

- Historical starter flag uses a minutes proxy.
- Historical book spreads are held at zero; blowout feature is unit-tested only.
- Returns are scheduled-game absences, not inferred injury diagnoses.
- Replay population is conditional on playing; unavailable projections are reported separately.
