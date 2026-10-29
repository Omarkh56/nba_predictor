# Final v9 replay verification

After the probability-calibration commit (`8f02282`) and the market-edge
handoff commit (`29670c8`), the warm-cache replay of the same 2025-26 RS and
PO population produced **182,128/182,128 matching market rows**, the exact
same keys and the exact same projection values as the raw replay following
the injury/minutes fixes. Maximum absolute projection difference: **0.0**.
The two post-commit replay runtimes were 238.3s and 244.7s; neither made an
NBA API request. The second run used 34 cached responses.

The generated [market-edge post-commit report](market-edge-postcommit.md)
compares the full population against the prior commit's replay: every exact
market in both the regular season and playoffs has Δ bias **0.00 percentage
points**, Δ MAE **0.0000** and Δ RMSE **0.0000**. It reports the **raw** model
before the final correction artifact because applying a full-season artifact
to its training dates would leak future results.

The [chronological correction report](residual-correction-comparison.md)
contains the actual out-of-sample bias before and after the additive
correction, by market and regime. Its final-test aggregate was regular
+9.82% → −2.27% and playoff +17.77% → +4.22%. This remaining playoff bias
and the worsened selected-pick hit rate in the [probability report](probability-calibration.md)
remain visible limitations; improved probability scoring is not evidence
of profitable betting.
