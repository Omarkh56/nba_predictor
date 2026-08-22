"""
kalman_validate.py — Walk-forward validation for the adaptive Kalman filter.

Compares three game-model variants on 2024-25 (validation season):
  (A) Hand-tuned      — newnbapredictor.py constants, no fitting
  (B) Static learned  — batch LogReg from train_model.py (learned_params.json)
  (C) Adaptive Kalman — EKF initialised from (B), updated game-by-game

Data splits (strict no-lookahead):
  Train:     2020-21 .. 2023-24  →  initial β₀ via train_model.py (already done)
  Val split: first Q_TUNE_FRAC of 2024-25 games  →  Q hyperparameter tuning
             remaining games                       →  three-way comparison
  Test:      2025-26              →  held-out (empty until season begins;
                                     re-run this script when data arrives)

Usage:
    python3 kalman_validate.py               # full run, saves tvp_state.json
    python3 kalman_validate.py --no-save     # comparison only, no file writes
    python3 kalman_validate.py --q 1e-4      # skip grid search, use given Q
    python3 kalman_validate.py --no-fetch    # use cached CSVs only
"""

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

warnings.filterwarnings("ignore")

# train_model.py has a __main__ guard, so importing is safe.
# The NBA-API patch in train_model applies at import time — that is fine.
try:
    from train_model import (
        GAME_FEATURES,
        build_game_dataset,
        fetch_team_game_logs,
        hand_tuned_predict,
    )
except ModuleNotFoundError as exc:
    sys.exit(f"Cannot import train_model: {exc}")

from kalman_filter import (
    PARAMS_FILE,
    TVP_STATE_FILE,
    GameEKF,
    PlayerKF,
    Q_GAME_SCALAR,
    save_tvp_state,
)

_HERE = Path(__file__).parent

VAL_SEASON   = "2024-25"
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]

# Fraction of val-season games used for Q grid search (rest = comparison table).
Q_TUNE_FRAC  = 0.60

# Grid of Q magnitudes to try (log-spaced from 1e-5 to 1e-2).
Q_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


# ── Core helpers ─────────────────────────────────────────────────────────────────

def _static_predict(df) -> np.ndarray:
    """Batch-LogReg predictions from learned_params.json (no game-by-game update)."""
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(
            "learned_params.json not found — run train_model.py first."
        )
    with open(PARAMS_FILE) as fh:
        params = json.load(fh)
    gm    = params["game_model"]
    mean  = np.array(gm["scaler_mean"])
    scale = np.array(gm["scaler_scale"])
    coefs = np.array([gm["log_coefficients"][f] for f in GAME_FEATURES])
    icept = gm["log_intercept"]

    X      = df[GAME_FEATURES].values
    Xs     = (X - mean) / scale
    logits = icept + Xs @ coefs
    return 1.0 / (1.0 + np.exp(-logits))


def _walk_forward_ekf(df, q_scalar: float) -> tuple:
    """
    Walk the EKF game-by-game through df (must be sorted chronologically).
    Returns (probs_array, final_ekf_state).
    probs_array[i] is the pre-update probability for game i.
    """
    ekf   = GameEKF.from_learned_params(q_scalar=q_scalar)
    probs = []
    for _, row in df.iterrows():
        ekf.predict_step()
        x_raw = np.array([float(row[f]) for f in GAME_FEATURES])
        p     = ekf.update(x_raw, int(row["home_win"]))
        probs.append(p)
    return np.array(probs), ekf


def _metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return {
        "log_loss": float(log_loss(y, probs)),
        "brier":    float(brier_score_loss(y, probs)),
        "n":        int(len(y)),
    }


# ── Q grid search ────────────────────────────────────────────────────────────────

def tune_q(df_tune) -> float:
    """Return the Q scalar with lowest log loss on the tuning split."""
    y = df_tune["home_win"].values
    print(f"\n  ┌── Q GRID SEARCH  ({len(df_tune)} games) ───────────────────┐")
    print(f"  │ {'Q':>10}   {'Log Loss':>10}  {'Brier':>8}              │")
    print(f"  │ {'─'*42}              │")
    best_q  = Q_GRID[0]
    best_ll = float("inf")
    for q in Q_GRID:
        probs, _ = _walk_forward_ekf(df_tune, q)
        m        = _metrics(y, probs)
        marker   = " ← best" if m["log_loss"] < best_ll else ""
        print(f"  │ {q:>10.1e}   {m['log_loss']:>10.4f}  {m['brier']:>8.4f}{marker:<15}│")
        if m["log_loss"] < best_ll:
            best_ll = m["log_loss"]
            best_q  = q
    print(f"  └{'─'*56}┘")
    print(f"\n  → Chosen Q = {best_q:.1e}  (log loss on tune split = {best_ll:.4f})")
    print(f"    Justification: smallest Q that minimises predictive log loss")
    print(f"    on the first {int(Q_TUNE_FRAC*100)}% of 2024-25; avoids overfitting to")
    print(f"    recency bias (larger Q) without freezing weights (smaller Q).")
    return best_q


# ── Three-way comparison ─────────────────────────────────────────────────────────

def compare(df_compare, q_scalar: float) -> dict:
    """Build metric dicts for all three models on the comparison split."""
    y = df_compare["home_win"].values

    ht_probs          = np.clip(hand_tuned_predict(df_compare), 1e-7, 1 - 1e-7)
    st_probs          = np.clip(_static_predict(df_compare),    1e-7, 1 - 1e-7)
    kf_probs, _final  = _walk_forward_ekf(df_compare, q_scalar)
    kf_probs          = np.clip(kf_probs, 1e-7, 1 - 1e-7)

    return {
        "Hand-tuned":      _metrics(y, ht_probs),
        "Static learned":  _metrics(y, st_probs),
        "Adaptive Kalman": _metrics(y, kf_probs),
    }


def _print_table(results: dict, header: str) -> None:
    best_ll = min(r["log_loss"] for r in results.values())
    best_bs = min(r["brier"]    for r in results.values())
    n       = next(iter(results.values()))["n"]
    print(f"\n  ┌── {header} ({n} games) {'─'*(max(0, 36-len(header)-len(str(n))))}")
    print(f"  │ {'Model':<22} {'Log Loss':>10} {'Brier':>10} │")
    print(f"  │ {'─'*44} │")
    for model, r in results.items():
        ll_flag = "★" if abs(r["log_loss"] - best_ll) < 1e-8 else " "
        bs_flag = "★" if abs(r["brier"]    - best_bs) < 1e-8 else " "
        print(
            f"  │ {model:<22} "
            f"{r['log_loss']:>9.4f}{ll_flag} "
            f"{r['brier']:>9.4f}{bs_flag} │"
        )
    print(f"  └{'─'*46}┘")
    print("    ★ = best on that metric")


# ── Recommendation ───────────────────────────────────────────────────────────────

def _recommend(results: dict) -> str:
    kf = results["Adaptive Kalman"]
    ht = results["Hand-tuned"]
    st = results["Static learned"]

    kf_wins_ll = kf["log_loss"] < min(ht["log_loss"], st["log_loss"])
    kf_wins_bs = kf["brier"]    < min(ht["brier"],    st["brier"])

    if kf_wins_ll and kf_wins_bs:
        return "use_kalman"
    elif kf_wins_ll or kf_wins_bs:
        return "monitor_kalman"
    else:
        return "keep_static"


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Kalman filter walk-forward validation")
    ap.add_argument("--no-save",   action="store_true", help="Print only, no file writes")
    ap.add_argument("--no-fetch",  action="store_true", help="Use cached CSVs only")
    ap.add_argument(
        "--q", type=float, default=None,
        metavar="FLOAT",
        help="Skip grid search and use this Q scalar (e.g. 1e-4)",
    )
    args = ap.parse_args()

    if not PARAMS_FILE.exists():
        sys.exit(
            "learned_params.json not found — run train_model.py first to produce"
            " the batch-fit β₀ that initialises the Kalman filter."
        )

    # ── Load data ──────────────────────────────────────────────────────────────
    all_seasons = TRAIN_SEASONS + [VAL_SEASON]
    print(f"Loading team game logs for {all_seasons[0]} .. {all_seasons[-1]} ...")
    team_logs = fetch_team_game_logs(all_seasons, force=False)

    print("Building walk-forward game dataset ...")
    df_game = build_game_dataset(team_logs)
    if df_game.empty:
        sys.exit("Game dataset is empty — check data_cache/.")

    df_val = (
        df_game[df_game["season"] == VAL_SEASON]
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    n_train = len(df_game[df_game["season"].isin(TRAIN_SEASONS)])
    print(
        f"  Train seasons: {n_train} games"
        f"  |  Val season ({VAL_SEASON}): {len(df_val)} games"
    )

    if len(df_val) < 50:
        sys.exit(f"Too few val-season games ({len(df_val)}) — check CSV cache.")

    # ── Q tuning split ─────────────────────────────────────────────────────────
    split_idx   = int(len(df_val) * Q_TUNE_FRAC)
    df_tune     = df_val.iloc[:split_idx]
    df_compare  = df_val.iloc[split_idx:].reset_index(drop=True)

    print(f"\n  Val split: {len(df_tune)} tune games | {len(df_compare)} comparison games")
    print(f"  (Q tuned on first {int(Q_TUNE_FRAC*100)}%; comparison on remaining"
          f" {100-int(Q_TUNE_FRAC*100)}% to avoid look-ahead)")

    # ── Q selection ────────────────────────────────────────────────────────────
    if args.q is not None:
        chosen_q = args.q
        print(f"\n  Skipping grid search — using specified Q = {chosen_q:.1e}")
    else:
        chosen_q = tune_q(df_tune)

    # ── Three-way comparison on held-out comparison split ─────────────────────
    results = compare(df_compare, chosen_q)
    _print_table(results, f"COMPARISON — {VAL_SEASON} (last {100-int(Q_TUNE_FRAC*100)}%)")

    recommendation = _recommend(results)
    kf  = results["Adaptive Kalman"]
    ht  = results["Hand-tuned"]
    st  = results["Static learned"]
    print(f"\n  Δ Kalman vs Hand-tuned :  log_loss {kf['log_loss']-ht['log_loss']:+.4f}"
          f"  brier {kf['brier']-ht['brier']:+.4f}")
    print(f"  Δ Kalman vs Static     :  log_loss {kf['log_loss']-st['log_loss']:+.4f}"
          f"  brier {kf['brier']-st['brier']:+.4f}")

    rec_msg = {
        "use_kalman":     "✓ Adaptive Kalman beats both baselines → use_kalman",
        "monitor_kalman": "~ Kalman wins on one metric → monitor_kalman (not yet default)",
        "keep_static":    "✗ Baselines win → keep_static (Kalman not wired in)",
    }[recommendation]
    print(f"\n  {rec_msg}")

    # ── 2025-26 test season note ───────────────────────────────────────────────
    df_test = df_game[df_game["season"] == "2025-26"]
    if df_test.empty:
        print(
            "\n  2025-26 held-out test: no games yet — season hasn't started."
            "\n  Re-run kalman_validate.py once games are cached to get the"
            "\n  true three-way comparison on unseen data."
        )
    else:
        results_test = compare(df_test.sort_values("game_date").reset_index(drop=True), chosen_q)
        _print_table(results_test, "HELD-OUT TEST — 2025-26")

    # ── Save state ─────────────────────────────────────────────────────────────
    if args.no_save:
        print("\n  --no-save: tvp_state.json not written.")
        return

    # Walk the EKF through the entire val season (both splits) to produce the
    # season-end state that the daily predictor will continue from.
    print(f"\n  Walking EKF through all {len(df_val)} val-season games to finalise state ...")
    _, final_ekf = _walk_forward_ekf(df_val, chosen_q)

    # Player KF: initialise from learned_params only (no walk-forward here —
    # the player dataset is large and its feature engineering requires a full
    # player-game log run; use kalman_daily_update.py for incremental updates).
    player_kf = PlayerKF.from_learned_params(q_scalar=5e-5)

    today = date.today().isoformat()
    save_tvp_state(
        game_ekf       = final_ekf,
        player_kf      = player_kf,
        last_updated   = today,
        recommendation = recommendation,
        path           = TVP_STATE_FILE,
    )

    print(f"  ✓ Saved tvp_state.json")
    print(f"    Q = {chosen_q:.1e}  |  {final_ekf.n_updates} game updates"
          f"  |  recommendation = {recommendation}")
    print(
        "\n  Next steps:"
        "\n    1. python3 nba_combined.py   — daily predictions (reads tvp_state.json)"
        "\n    2. python3 checkresults.py   — grade results → feeds nba_combined.py"
        "\n       kalman_daily_update.py    — incremental EKF update after each game"
        "\n    3. Re-run this script at season end to refresh the comparison table."
    )


if __name__ == "__main__":
    main()
