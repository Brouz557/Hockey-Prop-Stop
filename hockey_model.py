# ---------------------------------------------------------------
# hockey_model.py — Trend-Weighted Version (with Safe Fallback)
# ---------------------------------------------------------------
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 1️⃣ Basic Form Model (unchanged for safety)
# ---------------------------------------------------------------
def build_basic_form(shots_df):
    """Existing simple rolling-5 model."""
    if shots_df.empty:
        return pd.DataFrame()

    df = shots_df.copy()
    df = df.rename(columns={"shooterName": "player", "teamCode": "team"})

    if "shotWasOnGoal" not in df.columns or "game_id" not in df.columns:
        print("⚠️ Missing shotWasOnGoal or game_id.")
        return pd.DataFrame()

    df["shotWasOnGoal"] = df["shotWasOnGoal"].astype(int)
    df = df.sort_values(["player", "game_id"])

    grouped = (
        df.groupby(["player", "team", "game_id"], as_index=False)["shotWasOnGoal"]
        .sum()
    )
    grouped["avg_5"] = grouped.groupby("player")["shotWasOnGoal"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    latest = grouped.groupby(["player", "team"]).tail(1).reset_index(drop=True)
    return latest[["player", "team", "avg_5"]]


# ---------------------------------------------------------------
# 2️⃣ Enhanced Trend Form Model
# ---------------------------------------------------------------
def build_trend_form(shots_df):
    """
    Calculates rolling 3/5/10/20-game averages and adds trend + direction.
    """
    if shots_df.empty:
        print("⚠️ build_trend_form: empty shots_df")
        return pd.DataFrame()

    df = shots_df.copy()
    df = df.rename(columns={"shooterName": "player", "teamCode": "team"})
    if "shotWasOnGoal" not in df.columns or "game_id" not in df.columns:
        print("⚠️ Missing key columns in shots data.")
        return pd.DataFrame()

    df["shotWasOnGoal"] = df["shotWasOnGoal"].astype(int)
    df = df.sort_values(["player", "game_id"])

    grouped = (
        df.groupby(["player", "team", "game_id"], as_index=False)["shotWasOnGoal"]
        .sum()
    )

    for w in [3, 5, 10, 20]:
        grouped[f"avg_{w}"] = (
            grouped.groupby("player")["shotWasOnGoal"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
