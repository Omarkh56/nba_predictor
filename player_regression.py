"""Inference for the standardized player Ridge models saved by train_model.py."""

from typing import Optional

import numpy as np


def predict_player_stat(
    params: dict,
    stat: str,
    ewma_projection: float,
    is_home: bool,
    rest_days: Optional[float] = None,
) -> float:
    """Reconstruct a fitted Ridge prediction in the artifact's feature order.

    Coefficients in learned_params.json operate on standardized inputs. When
    pre-game rest is unavailable, use its training mean (zero after scaling)
    rather than inventing a rest period. Invalid artifacts raise ValueError or
    KeyError so the caller can retain its existing non-regression projection.
    """
    features = params["feature_names"]
    coefficient_keys = {
        f"ewma_{stat}": "ewma_weight",
        "is_home": "home_boost",
        "rest_days": "rest_coef",
    }
    if (
        not isinstance(features, list)
        or f"ewma_{stat}" not in features
        or len(set(features)) != len(features)
        or any(feature not in coefficient_keys for feature in features)
    ):
        raise ValueError(f"Unsupported player feature schema for {stat}: {features!r}")

    means = np.asarray(params["scaler_mean"], dtype=float)
    scales = np.asarray(params["scaler_scale"], dtype=float)
    expected_shape = (len(features),)
    if means.shape != expected_shape or scales.shape != expected_shape:
        raise ValueError("Player scaler dimensions do not match feature_names")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Player scaler must have finite means and positive finite scales")

    values = {f"ewma_{stat}": ewma_projection, "is_home": float(is_home)}
    if "rest_days" in features:
        values["rest_days"] = (
            means[features.index("rest_days")] if rest_days is None else rest_days
        )
    raw = np.asarray([values[feature] for feature in features], dtype=float)
    coefficients = np.asarray(
        [params[coefficient_keys[feature]] for feature in features], dtype=float
    )
    intercept = float(params["intercept"])
    if not np.isfinite(raw).all() or not np.isfinite(coefficients).all() or not np.isfinite(intercept):
        raise ValueError("Player regression inputs and coefficients must be finite")

    prediction = float(intercept + coefficients @ ((raw - means) / scales))
    if not np.isfinite(prediction):
        raise ValueError("Player regression prediction is not finite")
    return prediction
