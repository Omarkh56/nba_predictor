# Chronological staking-policy backtest

- Verified real-odds rows: 92
- Split: whole dates in chronological 60/20/20 order
- Evidence: inconclusive: 92 verified real-odds rows is below the 200-row minimum and chronological holdouts are thin
- Production Kelly: 0.25 with a 5% cap (unchanged)
- No missing prices were filled and no assumed -110 odds were used.

## Train (2026-04-30 to 2026-05-06)

| Policy | Real-odds rows | Bets | Profit | ROI | Max DD | Volatility | Median stake | Largest stake | Losing streak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_1pct | 59 | 35 | +3.4609 | 9.58% | 3.3091 | 0.0091 | 1.0354 | 1.0677 | 2 |
| quarter_kelly_cap_1pct | 59 | 35 | +3.4609 | 9.58% | 3.3091 | 0.0091 | 1.0354 | 1.0677 | 2 |
| quarter_kelly_cap_2pct | 59 | 35 | +6.7862 | 9.10% | 7.0566 | 0.0182 | 2.1432 | 2.2769 | 2 |
| quarter_kelly_cap_3pct | 59 | 35 | +10.2246 | 8.89% | 10.9754 | 0.0273 | 3.3252 | 3.6360 | 2 |
| quarter_kelly_cap_5pct | 59 | 35 | +18.2399 | 9.49% | 18.0551 | 0.0439 | 5.7858 | 6.8148 | 2 |
| fractional_035_cap_5pct_comparison | 59 | 35 | +17.2451 | 8.60% | 19.6550 | 0.0449 | 5.8824 | 6.8450 | 2 |

## Validation (2026-05-08 to 2026-05-09)

| Policy | Real-odds rows | Bets | Profit | ROI | Max DD | Volatility | Median stake | Largest stake | Losing streak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_1pct | 14 | 8 | +0.1092 | 1.37% | 0.1224 | 0.0104 | 0.9994 | 1.0000 | 2 |
| quarter_kelly_cap_1pct | 14 | 8 | +0.1092 | 1.37% | 0.1224 | 0.0104 | 0.9994 | 1.0000 | 2 |
| quarter_kelly_cap_2pct | 14 | 8 | +0.2178 | 1.36% | 0.2448 | 0.0207 | 1.9976 | 2.0000 | 2 |
| quarter_kelly_cap_3pct | 14 | 8 | +0.3258 | 1.36% | 0.3673 | 0.0311 | 2.9945 | 3.0000 | 2 |
| quarter_kelly_cap_5pct | 14 | 8 | -1.3731 | -3.69% | 1.3731 | 0.0502 | 4.9361 | 5.0000 | 2 |
| fractional_035_cap_5pct_comparison | 14 | 8 | +0.2886 | 0.73% | 0.6121 | 0.0516 | 4.9847 | 5.0000 | 2 |

## Test (2026-05-10 to 2026-05-12)

| Policy | Real-odds rows | Bets | Profit | ROI | Max DD | Volatility | Median stake | Largest stake | Losing streak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_1pct | 19 | 11 | -0.5342 | -4.80% | 2.8048 | 0.0088 | 1.0150 | 1.0227 | 4 |
| quarter_kelly_cap_1pct | 19 | 11 | -0.5342 | -4.80% | 2.8048 | 0.0088 | 1.0150 | 1.0227 | 4 |
| quarter_kelly_cap_2pct | 19 | 11 | -1.1613 | -5.17% | 5.7025 | 0.0176 | 2.0591 | 2.0908 | 4 |
| quarter_kelly_cap_3pct | 19 | 11 | -1.8791 | -5.51% | 8.6910 | 0.0264 | 3.1315 | 3.2044 | 4 |
| quarter_kelly_cap_5pct | 19 | 11 | +0.0466 | 0.09% | 13.0946 | 0.0409 | 5.0000 | 5.6571 | 4 |
| fractional_035_cap_5pct_comparison | 19 | 11 | -2.4204 | -4.25% | 14.2768 | 0.0430 | 5.0000 | 5.5928 | 4 |

## Production-policy result by market

| Group | Bets | Profit | ROI | Max DD |
|---|---:|---:|---:|---:|
| 3PM | 5 | -3.5937 | -12.88% | 12.2316 |
| AST | 18 | +27.4765 | 27.02% | 13.6295 |
| PRA | 1 | -6.5971 | -100.00% | 6.5971 |
| PTS | 2 | -0.6986 | -6.82% | 5.0000 |
| PTS+AST | 2 | -0.8005 | -7.63% | 5.2478 |
| REB | 17 | +32.4569 | 37.05% | 5.8581 |
| REB+AST | 9 | -31.5728 | -57.06% | 31.5728 |

## Production-policy result by direction

| Group | Bets | Profit | ROI | Max DD |
|---|---:|---:|---:|---:|
| OVER | 39 | +8.8394 | 4.08% | 28.1157 |
| UNDER | 15 | +7.8313 | 9.42% | 11.5715 |

## Limits

The verified prices cover a small, playoff-heavy period. The validation and test samples are too small to support a profitability or cap-selection conclusion. Results are implementation diagnostics, not evidence of a profitable strategy.

