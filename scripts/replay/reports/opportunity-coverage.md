# 2025-26 RS and 2026 PO pre-game proxy coverage

Synthetic L10/L5 lines are calibration diagnostics only, never ROI/edge/profit evidence.
A prior played team plus that day's schedule identifies the pre-game candidate.
A player not observed for that team that day is a void. Trades/long absences
and missing injury records limit historical roster reconstruction.

- Replay dates: 208; eligible candidate player-games (L5): 35029; projected market rows: 267824.
- DNP/ambiguous candidate player-games: 9676; historical same-day changed-team ambiguity: 36; stale prior-team candidates excluded: 5372.
- Reconstruction failures: 12408 market projections unavailable; these are never fabricated as bets.
- Warm local API cache: 0 live requests; wall runtime 340.6s.

| proxy prerequisite | excluded for too few played games | of these, Oct–Nov | of these, low recent minutes | eligible low-minute roles |
|---|---:|---:|---:|---:|
| L5 | 4075 | 2823 | 2438 | 9804 |
| L10 | 8324 | 5439 | 4676 | 7566 |

Early season and low-minute categories overlap. Low-minute status uses
the prior 10 played games and is **reported**, not used to reject a candidate.
L10 is primary; L5 is sensitivity. Only graded rows enter calibration.
