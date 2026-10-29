# Early-season fallback gate

Decision: **on** (generated 2026-10-16T07:22:20Z, commit `a6df849b43`, working tree dirty: False)

Rule: regular season and fewer than 10 current-season games -> project from last season (minutes_forecast + prior-season rate shrinkage). Every criterion was fixed before running.

| season | opening night off -> on | week 1 off -> on | paired corrected MAE off -> on (change, 95% CI) | rule-only rows: n, bias, MAE | rule-active: actual > proj | checks |
|---|---|---|---|---|---|---|
| 2024-25 | 0/42 -> 37/42 | 54% -> 90% | 3.790 -> 3.655 (-0.135, [-0.227, -0.059]) | 6120, +0.4%, 3.250 | 0.493 | 1_coverage=pass, 2_paired_mae_not_worse=pass, 3_rule_only_quality=pass, 4_no_directional_lean=pass |
| 2025-26 | 0/39 -> 37/39 | 52% -> 91% | 3.804 -> 3.645 (-0.160, [-0.234, -0.096]) | 6312, +0.1%, 3.394 | 0.481 | 1_coverage=pass, 2_paired_mae_not_worse=pass, 3_rule_only_quality=pass, 4_no_directional_lean=pass |

Full 2025-26 population (>= 10 prior games, RS+PO): 182128 paired rows, 0 unpaired, max |delta projection| = 0.0 -> pass.

Mean bias uses the full-season residual correction, which over-corrects the early-season mean; the median residual and the actual > projection share are reported so that skew stays visible.

Limits: 2025-26 is partly in-sample (the minutes_forecast constants were hand-set while studying it); 2024-25 is the cleaner out-of-sample season. 2024-25 replays use manual positions only (its rosters are not cached). Live team assignment for 0-game players (current rosters) is not replayable historically and is covered by unit tests and a live smoke test instead.
