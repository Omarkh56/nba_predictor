# Final report: Phase 1 (bias fixes) + Phase 3 (finance layer)

Consolidates the per-commit reports under `scripts/replay/reports/` and `reports/` into one document. Every
number below is quoted from a committed report; none is recomputed here. File paths are relative to the repo root.

## 1. Branches and git state

| branch | tip | purpose |
|---|---|---|
| `main` / `origin/main` | `0de4084` | unchanged throughout; both branches below are clean linear descendants of it |
| `codex/phase1-bias-fixes` | `a03304f` | replay engine + bias diagnosis/fixes (15 commits) |
| `codex/phase3-finance-layer` | `f99be53` | finance/staking layer, branched from phase1's tip (6 commits) |

Working tree: **clean** on `codex/phase3-finance-layer` (current HEAD). `git merge-base main phase1-bias-fixes`
and `git merge-base main phase3-finance-layer` both equal main's tip, and `git merge-base phase1-bias-fixes
phase3-finance-layer` equals phase1's tip exactly — phase3 is phase1 plus 6 more commits with **zero divergence**.
Either branch fast-forward-merges into `main` with no conflicts. Neither branch has been pushed.

## 2. Commits

### Phase 1 — `codex/phase1-bias-fixes`

| commit | purpose |
|---|---|
| `0d3e73b` | Preserve the owner's test edit as its own baseline commit (Commit 0) |
| `6cf8be3` | Chronological replay engine (`scripts/replay/`), `.gitignore` cleanup, `_to_delete/*.lock` removal (Commit 1) |
| `9687c43` | Record the true baseline replay (before any fix) for RS and playoffs |
| `0a03f86` | Idempotent bet-log upsert/dedup (Commit 2) |
| `578790a` | Verify replay is unaffected by log dedup (data hygiene only) |
| `b3157e1` | Preserve fetch failures as failures; log pick provenance (Commit 3) |
| `9b9dc13` | Verify replay after fetch-provenance changes |
| `be586a5` | Minute-history rework: all played appearances count for expected minutes, rate reliability filters kept separate (Commit 4) |
| `64cd790` | Injury-return ramp requires real evidence, fails closed otherwise (Commit 5) |
| `46d11d8` | Chronological unconditional residual correction; retire pick-based `MARKET_PROJ_ADJ` (Commit 6) |
| `8f02282` | Chronological probability calibration with the real-line non-inferiority gate (Commit 7a) |
| `29670c8` | Wire calibrated probability into prop market edge |
| `b044ae3` | Final replay verification after probability integration |
| `f953156` | Model-priced (zero-vig) historical odds backtest |
| `a03304f` | Held-out projection-bias diagnostic (raw vs. corrected, by projection level and by minutes error) |

### Phase 3 — `codex/phase3-finance-layer`

| commit | purpose |
|---|---|
| `85f7f17` | Finance audit (`reports/phase3_finance_audit.md`) + compatible schema additions |
| `ed39d42` | Quarter-Kelly staking gates (min edge, vig check, freshness, machine-readable skip reasons) |
| `651ac73` | Idempotent profit/bankroll grading in `checkresults.py` |
| `95fccbd` | Closing-odds capture and CLV calculation (`clv.py`) |
| `f60d9c4` | Safe exact-line historical finance backfill (`backfill_finance.py`) |
| `f99be53` | Chronological staking-policy backtest (`staking_backtest.py`) |

## 3. Tests

Every commit above was followed by a full `pytest` run per its own report or the next "docs: verify" commit.
Current state (`codex/phase3-finance-layer`, HEAD `f99be53`): **582 passed, 0 failed** (`python3 -m pytest -q`,
13.6s). The pre-Phase-1 baseline (`main`) was 469 passed; the branch added 113 tests across the replay engine,
bet-log dedup, fetch reliability, minute history, injury ramp, residual corrections, probability calibration, and
finance layer.

## 4. Replay engine, commands, cache

Full method and commands: `scripts/replay/README.md`. Core commands:

```
python3 scripts/replay/replay_props.py run --label X --season 2025-26 --season-type both [--cold]
python3 scripts/replay/replay_props.py run --label X --compare <previous-label>
```

**Cold cache** (`baseline`, commit `6cf8be3`): empty cache, 34 live requests, 0 retries, 0 failures, 38.6s
rate-limit delay, **279.7s total** (49.0s data load + 227.4s replay, 9 workers).
**Warm cache** (`market-edge-postcommit`, commit `29670c8`, run after 5 more commits of model changes): 0
requests, 34/34 cache hits, **244.7s total** (0.2s data load + 243.1s replay). `postcommit-verification.md`
confirms a same-population warm rerun after the probability/market-edge commits reproduced the prior raw replay's
182,128 rows with **max absolute projection difference 0.0**.

Replay population: 182,128 market-rows over 22,766 player-games (2025-11-09 → 2026-06-13), from 23,018 eligible
player-games (≥10 prior played games this season; no minutes floor — low-minute roles are a reporting stratum,
never excluded). `elig-2` eligibility version.

## 5. Bet-log integrity (Commit 2)

- `bet_log_backfilled.csv` reverified: 1,163 rows, 1,081 of 1,163 projections were recovered from
  `predictions.db`/JSON backfill (82 originally present), 1,163/1,163 (100%) final coverage, 0 line mismatches —
  confirmed unchanged from the figures the task description quoted.
- Idempotent dedup on `date+player+market+line+pick`: **1,163 → 969 rows, 194 duplicate rows removed, 0
  conflicting populated fields**. Final projection coverage 969/969 (100%), `p_model` coverage 969/969 (100%).
- A later pass (`calibration-dedup-diff.md`) moved 13 rows labeled `MISS` with no recorded `actual` to
  `backups/bet_log.void-or-ungraded.csv` (gitignored) — **969 → 956 real graded bets**, the population used for
  calibration and finance work. Local backups: `backups/bet_log.pre_dedupe.20260921.csv`,
  `backups/bet_log.before-void-cleanup.csv`, `backups/bet_log.void-or-ungraded.csv` (all gitignored, not archived
  into git per the task's instruction that git history already preserves the original).
- `calibration.json`, diagnostic-only rerun on the deduplicated log (`calibration-dedup-diff.md`): overall hit
  rate 50.6% → 51.7%; PTS_OVER sample/hit-rate 113/38.9% → 93/43.0%; 16 market/direction groups unchanged; 24
  fitted flag penalties unchanged; player/market residual entries 56 → 474 (recovered projections). The tracked
  file was restored afterward — Commit 7's calibration below is the real deployed one.

## 6. Fetch reliability and provenance (Commit 3)

Failed fetches are no longer cached as empty history; only genuine successful-empty responses are. Provenance
(raw/usable game counts, fetch status, data-as-of, season mode, model version) is logged with every new pick.
`b3157e1e21` replay confirmed prediction results were unchanged except for the specific cases where a previously
failed fetch had incorrectly been treated as a valid empty history.

## 7. Bias by market: baseline → each fix → final correction

All bias = `mean(projection − actual) / mean(actual)`; positive = over-projection. "ALL" pools every market;
"macro" is the unweighted mean of the 8 per-market values. Regime = Regular Season / Playoffs.

### 7a. True baseline — original, unmodified model (`baseline.json`, commit `6cf8be3`, before any Phase 1 model change)

| regime | ALL bias | macro bias |
|---|---:|---:|
| Regular | +3.1% | +4.3% |
| Playoffs | +11.9% | +13.6% |
| Combined | +3.8% | +5.0% |

Per-market (regular / playoffs): PTS +2.6%/+11.4%, REB +3.8%/+7.9%, AST +4.3%/+21.0%, 3PM +11.3%/+20.8%,
PTS+REB +2.7%/+10.2%, PTS+AST +2.9%/+13.2%, REB+AST +4.0%/+12.6%, PRA +2.9%/+11.7%.

### 7b. Minute-history fix alone (Commit 4; `minute-history-comparison.md`, "robust" candidate vs. the old filter)

| candidate | regular bias | playoff bias | combined bias | combined RMSE |
|---|---:|---:|---:|---:|
| old filter (control) | +3.16% | +12.12% | +3.81% | 5.545 |
| **robust (selected)** | **−1.88%** | **+5.43%** | **−1.35%** | **5.461** |
| all-played (rejected) | −3.78% | +3.24% | −3.27% | 5.482 |

Two other candidates were evaluated and rejected: "all played appearances, no reliability protection" had the
best playoff MAE but a worse regular-season under-projection; "robust" (all played games count toward expected
minutes with reduced leverage for extreme shorts, rate-reliability filters kept separate) was the more stable
cross-regime choice and is what shipped. Combined RMSE improved 5.545 → 5.461.

### 7c. Injury-ramp fix alone, applied after 7b (`injury-ramp-comparison.md`)

| regime | prior (7b) bias | after (evidence-gated) bias |
|---|---:|---:|
| Regular | −1.88% | **+12.16%** |
| Playoffs | +5.43% | **+17.40%** |
| Combined | −1.35% | **+12.54%** |

**Bias rose, and this is reported honestly rather than compensated elsewhere in that commit** (as the task
required): the old calendar-gap heuristic had been discounting 13,924/20,884 regular-season and 1,028/1,882
playoff player-games by ~16% on average, without evidence of an actual injury — a false downward discount that
was masking the true upward bias. It is now disabled in the absence of real evidence (missed scheduled team
games / historical injury status), which the current data cannot supply — see §14.

### 7d. Both fixes together, raw/uncorrected (`market-edge-postcommit.json`, commit `29670c8`)

| regime | ALL bias | macro bias |
|---|---:|---:|
| Regular | +12.2% | +13.2% |
| Playoffs | +17.4% | +18.7% |
| Combined | +12.5% | +13.6% |

`correction-raw-replay-check.md` confirms this raw population is byte-identical (Δ=0 everywhere) after retiring
the pick-based `MARKET_PROJ_ADJ`/`MANUAL_PLAYER_PENALTIES` mechanism — those were already inactive in this replay.

### 7e. Final: chronological unconditional residual correction (Commit 6), final held-out test split only

Chosen: **additive, k=1000** (validation objective 6.875, selected over multiplicative k=1000's 6.959 and every
other k tried, 50/250/1000/5000, on the predeclared "MAE + |mean residual|" criterion summed over RS+PO
validation). Same-date outcomes never contribute to that date's own correction.

| regime | rows | raw test bias | corrected test bias | corrected test MAE |
|---|---:|---:|---:|---:|
| Regular Season | 40,512 | +9.82% | **−2.27%** (95% CI −5.03% to +0.17%) | 3.718 |
| Playoffs | 3,608 | +17.77% | **+4.22%** (95% CI +1.22% to +7.12%) | 3.286 |

Regular-season corrected bias is not distinguishable from zero at 95%; **playoff corrected bias remains positive
and the interval excludes zero** — an acknowledged, not hidden, residual limitation.

Corrected bias by market, final test (regular / playoffs): PTS −3.0%/+5.4%, REB +0.1%/+0.0%, AST −2.6%/+6.7%,
3PM −1.5%/−1.1%, PTS+REB −2.2%/+4.0%, PTS+AST −3.0%/+5.7%, REB+AST −1.0%/+2.3%, PRA −2.3%/+4.4%.

**Multiplicative candidates** (e.g. k=250: regular −2.43%/+3.79% MAE-comparable) had *slightly* lower MAE but a
worse balance across the two regimes under the predeclared criterion and were not selected.

**Bucketed correction** was evaluated separately in the later projection-bias diagnostic (`a03304f`): a
multiplicative, per-regime-and-market, 1-bucket candidate scored 6.810 vs. the deployed additive's 6.875 on the
same validation objective (regular test −2.54%/MAE 3.688, playoffs test +3.00%/MAE 3.233 — both slightly better
than deployed). **It was not promoted**: its playoff fit uses the current postseason's own early outcomes, only
one unseen postseason exists to validate against, and stability/cold-start behavior is unproven. The evidence is
preserved in `scripts/replay/reports/projection-bias-diagnostic.md`; the deployed correction is unchanged.

The static artifact `residual_corrections.json` (form=additive, k=1000, per-market correction/shrinkage/sample)
is fit on **all** completed 2025-26 RS+PO data for future-season use, and replay workers explicitly disable it
and project the raw model when replaying 2025-26 itself (applying it to its own training dates would leak).

**Where the remaining error sits** (`a03304f`, for future work — see §15): the corrected model under-projects the
lowest projection quartile (regular −23.3%, playoffs −25.8%) and over-projects the highest (regular +7.2%,
playoffs +15.1%); bias is close to zero when projected minutes land within 2 of actual, and swings to roughly
±20–65% when minutes miss by 5+ in either direction — minutes forecasting, not the rate model, is the dominant
remaining error source.

### MAE/RMSE, before → after (combined, pooled)

| | baseline (7a) | raw after fixes (7d) | corrected, final test (7e) |
|---|---:|---:|---:|
| MAE | 4.058 | 3.784 | 3.718 (regular) / 3.286 (playoffs) |
| RMSE | 5.929 | 5.495 | — (MAE reported per regime in the final-test table above; combined pre-correction RMSE 5.495) |

## 8. OVER/UNDER pick ratio and hit rate, before → after

From `probability-calibration.md`, replaying picks at the **same historical real lines** (956 logged
opportunities):

| population | n | OVER | UNDER | OVER share | hit rate |
|---|---:|---:|---:|---:|---:|
| logged (old model, as bet) | 956 | 765 | 191 | 80.0% | 0.517 |
| replayed, fixed model, all | 956 | 403 | 553 | 42.2% | 0.490 |
| replayed, fixed model, past edge floor | 666 | 366 | 300 | 55.0% | 0.480 |

The fixed model flips the pick mix from 80% OVER to a minority-OVER book (of 93 logged PTS OVER bets, 39 switch
to UNDER). **This does not show an improved selected-pick hit rate** (0.490/0.480 vs. 0.517 old) — stated exactly
this way in the source report. Probability-score improvement (§9) is not evidence of profitability, and no ROI
claim is made from it.

## 9. Probability calibration (Commit 7)

Regenerated by replaying the fixed, corrected model at each logged bet's original line, using only pre-game
information; 0 logged bets excluded for unreconstructable inputs (956/956 coverage). Chronological
train/validation/final-test; the final 226 rows on 8 dates after 2026-05-03 were never used to choose the method.

| view | n | Brier | log loss |
|---|---:|---:|---:|
| stored historical `p_model` (before-only baseline) | 956 | — (not re-scored; see OVER 0.635/UNDER 0.713 mean predicted vs. 0.517 realized below) | — |
| replayed raw (uncalibrated) | 956 all / 226 held-out | 0.2743 / 0.2647 | 0.7512 / 0.7247 |
| replayed, Platt-calibrated | 956 all / 226 held-out | 0.2547 / **0.2455** | 0.7032 / **0.6835** |

Mean model probability vs. realized hit rate, by logged pick:

| pick | n | old logged p_model | fixed raw p_model | fixed calibrated p_model | realized |
|---|---:|---:|---:|---:|---:|
| OVER | 765 | 0.635 | 0.589 | 0.502 | 0.502 |
| UNDER | 191 | 0.713 | 0.749 | 0.661 | 0.576 |

**Real-line non-inferiority gate** (predeclared margins: Brier 0.005, log loss 0.01), Platt vs. identity, paired
by date (226 rows / 8 dates): Δbrier −0.0192 (one-sided 95% CI upper −0.0033), Δlog loss −0.0413 (upper −0.0067).
**Both upper bounds beat their margins → gate passed. Deployed for 2026-27: `platt`.**

Calibration was fit and gated as a **single global model** pooled across markets (226 held-out rows split 8
markets leaves too few per market to gate independently); it was not selected per-market, so there is no
per-market identity-calibration fallback to report separately this cycle.

### Proxy-line diagnostics (L10 primary, L5 sensitivity) — synthetic, never ROI/edge/profit evidence

| population | n (RS/PO) | mean hit rate |
|---|---:|---:|
| proxy_l10 | 165,536 / 14,984 | 0.560 / 0.647 |
| proxy_l5 | 183,592 / 15,024 | 0.584 / 0.641 |

Brier improved under Platt in every proxy stratum reported (e.g. proxy_l10 regular_test 0.2479→0.2447, playoffs_test
0.2477→0.2390; proxy_l5 regular_test 0.2420→0.2402, playoffs_test 0.2439→0.2374). L10 and L5 cover different
eligible players and different lines, so their absolute scores are **not a paired comparison**; no matched-sample
L10-vs-L5 intersection table was produced this cycle (open item, §16).

Opportunity/coverage accounting (`opportunity-coverage.md`): 208 replay dates, L5 eligible candidate player-games
35,029 / L10 eligible 30,780 (computed as 35,029 minus the extra 4,249 L10 requires), 9,676 DNP/ambiguous
candidates, 12,408 reconstruction failures never fabricated as bets, 5,372 stale prior-team candidates excluded,
36 same-day team-change ambiguities. Low-minute status is reported, never used to reject a candidate.

## 10. Model-priced (zero-vig) historical backtest

`model-book-2025-26.md`: games 1,262 (home Brier 0.2093, log loss 0.6069, 67.6% most-likely-side accuracy; the
game model's training seasons 2020-23 make 2025-26 an unseen holdout). Props: 180,520 graded quotes + 60,880
void quotes; raw/calibrated OVER Brier 0.2480/0.2451, log loss 0.6901/0.6832. Max fair-odds overround error
2.2e-16 (a genuine zero-vig book). This is a calibration/prediction test, **not** a betting-ROI test — no
historical sportsbook prices or customer bets were used.

## 11. `calibration.json`, old vs. new

`calibration-dedup-diff.md` (compared with the tracked pre-cleanup artifact at `46d11d8`):

| field | before | after |
|---|---:|---:|
| graded bets | 1,163 | 956 |
| overall hit rate | 50.6% | 51.7% |
| player/market residual entries | 56 | 474 |
| PTS OVER sample / hit rate | 113 / 38.9% | 93 / 43.0% |

Market-direction and flag-penalty group counts (16 and 24 respectively) are unchanged. `calibration.json`
retains historic market-direction/flag diagnostics for reporting; it is **not** used to adjust live projections
— that mechanism was retired in Commit 6 (§7).

## 12. Historical Odds API availability (informational only — no purchase, no request made)

`historical-odds-availability.md`: the Odds API documents historical event-level player props from
**2023-05-03**, paid plans only, via the historical per-event endpoint with a specific pre-tip snapshot
timestamp. Cost formula: **10 × distinct markets returned × regions × events × snapshots** credits (e.g. 8
markets × 1 region × 1 event × 1 snapshot = 80 credits). A documented zero-credit `/v4/sports/` entitlement check
failed on TLS/network errors including a verified fallback, so **current plan/quota remain unverified** without
an account-dashboard check. No historical request was made and no credits were spent.

## 13. Finance layer (Phase 3)

Audit (`reports/phase3_finance_audit.md`, 2026-09-23) found: 956 real graded picks with projection/actual/result/
p_model, only 130 with a de-vig market probability and 118 with a decimal price (spanning Apr 30–Jun 8 only); 0
stored stakes or realized profit anywhere; the odds tracker had no timestamped repeated snapshots so no closing
price / historical CLV could exist; Kelly sizing had no minimum-edge, vig-comparison, freshness or skip-reason
logic; grading was not idempotent for bankroll math. Six commits closed every one of those gaps:

- **Staking gates** (`ed39d42`): `compute_stake_pct()` now enforces a configured minimum edge, compares edge
  against the paired-market vig, checks data freshness, and returns a machine-readable skip reason instead of a
  silent zero.
- **Idempotent profit/bankroll grading** (`651ac73`): `checkresults.py` settles profit only with valid entry odds
  and a positive stake; `NBA_STARTING_BANKROLL` enables per-date bankroll tracking sized from that date's opening
  balance (CSV row order can't move stakes); writes `reports/finance_summary.{json,md}`,
  `reports/bankroll_curve.csv`.
- **CLV** (`95fccbd`): `clv.py` computes directional line CLV and same-line de-vig/raw price CLV from the closing
  snapshot strictly before tip-off; failed joins keep CLV blank with a machine-readable `clv_unmatched_reason`
  rather than guessing.
- **Safe backfill** (`f60d9c4`): `backfill_finance.py` recovered entry odds for exactly the rows matching
  date/player/exact-market/exact-line — **92 of 956** rows. It did **not** reconstruct stakes, closing prices,
  CLV or timestamps for historical rows (`reports/finance_backfill_summary.md`); 26 legacy decimal prices with no
  exact-line match were correctly excluded rather than assumed.
- **Staking-policy backtest** (`f99be53`): `staking_backtest.py`, chronological 60/20/20 split over the 92
  verified real-odds rows.

**Current real-money state**: `reports/finance_summary.md` shows 0 finance-eligible rows (956 settled, 0 have a
recorded positive stake — no historical stakes were ever fabricated); `reports/clv_summary.md` shows 0 CLV rows
(no historical closing snapshots exist to join against — both are correct, not bugs, given the recorded gaps).
Going forward, new bets graded with a `NBA_STARTING_BANKROLL` set and CLV capture enabled will populate both.

**Staking-policy backtest conclusion** (`reports/staking_policy_backtest.md`): declared a 200-row evidence
minimum in advance; only 92 rows were available, so **the comparison is explicitly reported as inconclusive**.
Production policy (quarter-Kelly, 5% cap) is left unchanged. By market, REB (+37.05% ROI, 17 bets) and AST
(+27.02%, 18 bets) were positive and REB+AST (−57.06%, 9 bets) and PRA (−100%, 1 bet) were sharply negative on
this thin sample — presented only as diagnostics, explicitly not as evidence of a profitable strategy.

## 14. Known limitations (not hidden)

- Playoff corrected bias remains positive and distinguishable from zero (§7e); minutes-forecast error is the
  dominant remaining driver (§7e).
- The injury-ramp fix increased raw bias by unmasking a prior false discount (§7c); no compensating change was
  made in that commit, by design.
- The fixed model's selected-pick hit rate on real historical lines (0.480–0.490) is **not** better than the old
  model's (0.517) — better probability calibration is not the same as a better betting strategy (§8, §9).
- Probability calibration was gated as one global model, not per market, due to thin per-market holdout samples
  (§9).
- L10 and L5 proxy results cover different populations and were never matched into one intersection comparison
  (§9, §16).
- The bucketed/multiplicative residual-correction candidate scored slightly better on validation but was not
  promoted for lack of a second unseen postseason (§7e).
- The finance layer's staking-policy backtest is built on 92 rows, below its own declared 200-row minimum, and
  is explicitly inconclusive (§13).
- Historical Odds API entitlement/quota is unverified (no billable call was made) (§12).
- Real logged-bet data is itself selected by the old model (fewer UNDERs, systematically different players/lines
  than an unconditional population) — calibration and hit-rate figures over it inherit that selection.

## 15. Remaining `v9 TODO` items

- `playerlinepredictor.py:522` — **`TODO(v9)`**: re-evaluate thin-sample playoff-rate weighting
  (`_adaptive_po_weight`) and the analogous early-regular-season low-games regime under chronological as-of
  validation. Deliberately left unchanged this phase, per instruction; do not treat the TODO as license to change
  it without that validation.
- Multiplicative/bucketed residual correction (§7e): re-evaluate once a second unseen postseason of chronological
  data exists.
- Injury-ramp: re-enable only once a real evidence source (missed scheduled team games and/or historical
  injury-status records) is available; current data cannot support it (§7c).
- Finance layer: the staking-policy comparison needs the real-odds sample to grow past 200 rows before any
  policy decision is revisited (§13); CLV/bankroll reporting will populate naturally as new bets are graded with
  `NBA_STARTING_BANKROLL` set.
- Per-market probability-calibration gating was not attempted this cycle for lack of samples; revisit once more
  real-line holdout data accumulates (§9).
- No L10-vs-L5 matched-sample intersection report exists yet (§9); build one if a direct L5-vs-L10 comparison is
  needed rather than the two independent populations reported here.
