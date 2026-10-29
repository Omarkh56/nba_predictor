"""v9 coherent player-prop probability calibration (UNDER = 1 - OVER)."""
from __future__ import annotations

import json
import math
from pathlib import Path

ARTIFACT_PATH = Path(__file__).with_name("probability_calibration.json")


def calibrate_over(p_over: float, market: str, artifact: dict) -> float:
    """Identity when no method passed the real-line gate."""
    p = min(1.0 - 1e-6, max(1e-6, float(p_over)))
    if not artifact or artifact.get("version") != "v9-probability-1" or artifact.get("method") != "platt":
        return p
    params = artifact.get("markets", {}).get(market)
    if not params:
        return p
    logit = math.log(p / (1.0 - p))
    z = float(params["intercept"]) + float(params["slope"]) * logit
    if not math.isfinite(z):
        raise ValueError("invalid probability calibration for %s" % market)
    return min(1.0 - 1e-6, max(1e-6, 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))))


def load_artifact(path=ARTIFACT_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open() as fh:
        info = json.load(fh)
    if info.get("version") != "v9-probability-1" or info.get("method") not in ("identity", "platt"):
        raise ValueError("unsupported probability-calibration artifact")
    return info
