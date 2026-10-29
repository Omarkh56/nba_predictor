# Minutes replay: phase3-minutes-202526 / 2025-26

Minutes MAE is counted once per player-game; bias is macro averaged over markets.

| Regime | Model | Player-games | Minutes MAE | Bias | Projection MAE |
|---|---|---:|---:|---:|---:|
| Playoffs | candidate | 1887 | 5.1401 | +19.9128% | 3.5903 |
| Playoffs | baseline | 1887 | 5.3134 | +18.8434% | 3.5836 |
| Regular Season | candidate | 24729 | 4.9316 | +9.1861% | 3.7072 |
| Regular Season | baseline | 24729 | 4.9456 | +12.4862% | 3.8082 |

Full minutes-error buckets and early-season results are in the adjacent JSON.

- Historical starter flag uses a minutes proxy.
- Historical book spreads are held at zero; blowout feature is unit-tested only.
- Returns are scheduled-game absences, not inferred injury diagnoses.
- Replay population is conditional on playing; unavailable projections are reported separately.
