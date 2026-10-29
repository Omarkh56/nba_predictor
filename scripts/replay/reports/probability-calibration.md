# v9 chronological probability calibration

Synthetic rolling-average lines are **calibration diagnostics only**. They are not
bookmaker prices; no ROI, edge or profit is inferred from them.

- Real logged-line coverage: 956/956; void/ungraded excluded before fitting: 0.
- Held-out real dates after 2026-05-03: 226 rows on 8 dates.
- Paired date-bootstrap, Platt − identity: Brier -0.0192 (95% upper -0.0033, margin +0.005); log loss -0.0413 (upper -0.0067, margin +0.010).
- Gate passed; 2026-27 artifact: **platt**. Runtime: 19.0s.

| regime / line | sample | identity Brier | Platt Brier | identity log loss | Platt log loss |
|---|---:|---:|---:|---:|---:|
| proxy_l10 playoffs_test | 3568 | 0.2477 | 0.2390 | 0.6992 | 0.6722 |
| proxy_l10 playoffs_val | 3544 | 0.2328 | 0.2279 | 0.6615 | 0.6477 |
| proxy_l10 regular_test | 40168 | 0.2479 | 0.2447 | 0.6905 | 0.6825 |
| proxy_l10 regular_val | 41328 | 0.2456 | 0.2433 | 0.6859 | 0.6796 |
| proxy_l5 playoffs_test | 3568 | 0.2439 | 0.2374 | 0.6939 | 0.6733 |
| proxy_l5 playoffs_val | 3544 | 0.2339 | 0.2295 | 0.6649 | 0.6520 |
| proxy_l5 regular_test | 41248 | 0.2420 | 0.2402 | 0.6793 | 0.6739 |
| proxy_l5 regular_val | 42168 | 0.2392 | 0.2380 | 0.6736 | 0.6694 |
| real logged all | 956 | 0.2743 | 0.2547 | 0.7512 | 0.7032 |
| real logged heldout | 226 | 0.2647 | 0.2455 | 0.7247 | 0.6835 |

L5 and L10 cover different eligible players and use different lines; their
absolute scores are not a paired comparison. Reliability bins by exact market
and both directions are in `probability-calibration.json`.

## Logged bets: model probability versus observed hit rate

| logged pick | n | old logged p_model | fixed replay raw p_model | fixed replay calibrated p_model | realized |
|---|---:|---:|---:|---:|---:|
| OVER | 765 | 0.635 | 0.589 | 0.502 | 0.502 |
| UNDER | 191 | 0.713 | 0.749 | 0.661 | 0.576 |

## Pick replay at the same historical real lines

This reselects a side at each of the *logged* opportunities, not all sportsbook
opportunities. Confidence/bookmaker/market-cap filters cannot be reconstructed
from this log; the displayed filter applies only the direction-specific edge floor.

| population | n | OVER | UNDER | remaining hit rate |
|---|---:|---:|---:|---:|
| logged | 956 | 765 | 191 | 0.517 |
| replayed_all | 956 | 403 | 553 | 0.490 |
| replayed_over_edge_floor | 666 | 366 | 300 | 0.480 |

Among 93 logged PTS OVER bets, 39 switch to UNDER; 0 remain OVER but fall below its edge floor.
Overall, 290 of 956 logged-line opportunities fail the edge floor.
This replay does not show an improved selected-pick hit rate; probability score
improvement does not establish bet profitability or justify a staking claim.

## Candidate/void coverage

A pre-game team-inferred roster is evaluated even if the player does not play;
such rows are VOID and are excluded from fit and score. Current rosters are not
a complete historical injury or transaction archive. Proxy opportunities and
void counts, early-season/low-minute exclusions and runtime are in
`opportunity-coverage.md`. Large row files are gitignored.
