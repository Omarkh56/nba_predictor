# Historical NBA player-prop lines: availability and cost

The Odds API documents historical event-level player props from 2023-05-03,
on paid usage plans only. It requires the historical per-event endpoint and
a specific pre-tip snapshot timestamp; availability of a given market/book
at any snapshot varies.

Cost: **10 × distinct markets returned × regions × events × snapshots**
in usage credits. One returned player market in one region for one event and
one snapshot costs 10 credits; eight markets cost up to 80 per event/snapshot.
Fetching historical event IDs separately is also on paid plans. We made no
historical request and purchased no credits.

The configured `ODDS_API_KEY` exists, but `odds_api.py` only wraps normal
verified HTTPS requests and cannot inspect subscription entitlements. A
documented zero-credit `/v4/sports/` check failed because of TLS/network
errors, including a verified TLS fallback. Its historical access therefore
remains **unverified**; plan status and remaining quota require an account
dashboard check or an expressly approved endpoint probe. The proxy-line
calibration does not imply the existence of those real historical prices.

Sources: [Odds API v4 guide](https://the-odds-api.com/liveapi/guides/v4/),
[historical odds overview](https://the-odds-api.com/historical-odds-data/).
