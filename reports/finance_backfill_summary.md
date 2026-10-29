# Historical finance backfill

Only actual stored consensus quotes matching date, normalized player, exact market, and exact line were used. Stakes, profits, closing prices, CLV, and timestamps were not reconstructed.

- Input rows: 956
- Rows with entry odds recovered: 92
- Rows with closing odds recovered: 0
- Rows with line clv: 0
- Rows with price clv: 0
- Rows remaining without recovered entry odds: 864
- Ambiguous matches: 0
- Legacy decimal odds without exact line match: 26
- Stakes backfilled: 0
- Profits backfilled: 0
- Timestamps backfilled: 0

## Unmatched reasons

- invalid_selected_odds: 4
- no_exact_line_match: 34
- no_player_market_match: 826
