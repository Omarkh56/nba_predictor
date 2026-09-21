"""v9 projection correction fitted on unconditional, pre-game replay residuals.

The artifact is trained from every eligible player-game outcome, not selected
bets. `raw_projection` must be the model output before any residual correction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional


ARTIFACT_PATH = Path(__file__).with_name("residual_corrections.json")


def fit_market(rows, form: str, k: float) -> Dict[str, float]:
    """Fit a market from `(raw_projection, actual)` pairs with n/(n+k) shrinkage."""
    if form not in ("additive", "multiplicative") or k <= 0:
        raise ValueError("form must be additive or multiplicative, and k > 0")
    clean = [(float(p), float(a)) for p, a in rows
             if math.isfinite(float(p)) and math.isfinite(float(a)) and float(p) >= 0]
    n = len(clean)
    if not n:
        return {"n": 0, "sum_projection": 0.0, "sum_actual_minus_projection": 0.0,
                "correction": 0.0}
    summed_projection = sum(p for p, _ in clean)
    residual_sum = sum(a - p for p, a in clean)
    mean_residual = residual_sum / n
    fraction = residual_sum / summed_projection if summed_projection > 0 else 0.0
    shrink = n / (n + k)
    return {"n": n, "sum_projection": summed_projection,
            "sum_actual_minus_projection": residual_sum,
            "mean_residual": mean_residual, "residual_fraction": fraction,
            "shrinkage": shrink,
            "correction": shrink * (mean_residual if form == "additive" else fraction)}


def apply_correction(raw_projection: float, market: str, artifact: Optional[dict]) -> float:
    """Apply exactly one final-market correction; unknown markets are unchanged."""
    if not artifact or artifact.get("version") != "v9-unconditional-1":
        return raw_projection
    info = artifact.get("markets", {}).get(market)
    if not info:
        return raw_projection
    c = float(info["correction"])
    if not math.isfinite(c):
        raise ValueError("non-finite residual correction for %s" % market)
    form = artifact["form"]
    if form == "additive":
        return max(0.0, raw_projection + c)
    if form == "multiplicative":
        return max(0.0, raw_projection * (1.0 + c))
    raise ValueError("invalid residual-correction form: %s" % form)


def load_artifact(path=ARTIFACT_PATH) -> dict:
    """Missing artifact leaves live projections untouched; invalid artifacts fail closed."""
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open() as fh:
        artifact = json.load(fh)
    if artifact.get("version") != "v9-unconditional-1" or artifact.get("form") not in (
        "additive", "multiplicative"
    ):
        raise ValueError("unsupported residual-correction artifact")
    return artifact
