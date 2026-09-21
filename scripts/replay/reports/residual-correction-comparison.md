# Chronological unconditional residual correction

Residual is actual − raw projection. Each game's adjustment is fitted using earlier game dates only.
Regular-season and playoff validation select the form and k; the final test is not used for selection.
Chosen: **additive**, k=1000; validation criterion: sum over RS+PO validation of MAE + absolute mean residual.

| candidate | validation score | RS val bias | RS val MAE | PO val bias | PO val MAE | RS test bias | RS test MAE | PO test bias | PO test MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw k=0 | — | +12.09% | 3.760 | +14.37% | 3.432 | +9.82% | 3.874 | +17.77% | 3.531 |
| additive k=1000 | 6.875 | -0.24% | 3.584 | +0.83% | 3.203 | -2.27% | 3.718 | +4.22% | 3.286 |
| additive k=250 | 6.895 | -0.94% | 3.582 | +0.38% | 3.199 | -2.76% | 3.716 | +3.79% | 3.282 |
| additive k=50 | 6.901 | -1.14% | 3.581 | +0.25% | 3.198 | -2.89% | 3.716 | +3.67% | 3.280 |
| multiplicative k=250 | 6.944 | -0.48% | 3.564 | +1.89% | 3.186 | -2.43% | 3.688 | +4.85% | 3.252 |
| multiplicative k=50 | 6.950 | -0.67% | 3.563 | +1.77% | 3.185 | -2.57% | 3.687 | +4.74% | 3.251 |
| multiplicative k=1000 | 6.959 | +0.20% | 3.568 | +2.29% | 3.191 | -1.95% | 3.691 | +5.26% | 3.257 |
| additive k=5000 | 7.274 | +2.50% | 3.603 | +2.81% | 3.223 | -0.18% | 3.731 | +6.16% | 3.308 |
| multiplicative k=5000 | 7.385 | +2.84% | 3.591 | +4.08% | 3.214 | +0.08% | 3.707 | +7.08% | 3.283 |

## Regular final test, bias by exact market

| market | raw | chosen additive | multiplicative k=250 | chosen MAE |
|---|---:|---:|---:|---:|
| PTS | +8.7% | -3.0% | -3.2% | 4.923 |
| REB | +12.7% | +0.1% | +0.0% | 1.940 |
| AST | +10.3% | -2.6% | -3.0% | 1.433 |
| 3PM | +17.3% | -1.5% | -2.2% | 0.920 |
| PTS+REB | +9.4% | -2.2% | -2.3% | 5.890 |
| PTS+AST | +8.9% | -3.0% | -3.2% | 5.475 |
| REB+AST | +11.7% | -1.0% | -1.2% | 2.707 |
| PRA | +9.5% | -2.3% | -2.5% | 6.457 |

## Playoffs final test, bias by exact market

| market | raw | chosen additive | multiplicative k=250 | chosen MAE |
|---|---:|---:|---:|---:|
| PTS | +18.5% | +5.4% | +5.9% | 4.354 |
| REB | +13.0% | +0.0% | +0.5% | 1.766 |
| AST | +22.6% | +6.7% | +7.8% | 1.180 |
| 3PM | +19.6% | -1.1% | -0.2% | 0.867 |
| PTS+REB | +16.9% | +4.0% | +4.5% | 5.335 |
| PTS+AST | +19.4% | +5.7% | +6.4% | 4.730 |
| REB+AST | +16.4% | +2.3% | +3.1% | 2.366 |
| PRA | +17.7% | +4.4% | +5.1% | 5.689 |

The multiplicative candidates lower MAE more, but the additive k=1000 candidate balances absolute residual bias across both validation regimes better under the predefined criterion. The static artifact is fit on all completed 2025-26 RS and PO data for use in future seasons; it must never be applied to earlier dates in the replay.
