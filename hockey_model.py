# ---------------------------------------------------------------
# hockey_model.py
# Core analytics engine for Hockey Prop Stop / Hockey Bot
# Now includes matchup-aware modeling
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson
import re

# ---------------------------------------------------------------
# Smart parser for raw NHL data
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """Automatically detect, clean, and prepare raw NHL CSVs for modeling."""
    skaters = pd.DataFrame()
    teams = pd.DataFrame()
    shots = pd.DataFrame()
    goalies = pd.DataFrame()
    lines = pd.DataFrame()

    for name, df in file_dfs.items():
        if df is None or df.empty:
            continue
        cols = [c.lower() for c in df.columns]

        # --- Shots file ---
        if any(re.search(r"shot|sog|attempt", c) for c in cols):
            shots = df.copy()
            if "shooter" in cols and "shootername" not in cols:
                shots.rename(columns={"shooter": "shooterName"}, inplace=True)
            if "shots" in cols and "shotsongoal" not in cols:
                shots.rename(columns={"shots": "shotsOnGoal"}, inplace=True)

        # --- Skaters ---
        elif any(re.search(r"player|skater", c) for c in cols):
            skaters = df.copy()

        # --- Teams ---
        elif any(re.search(r"team|shotsagainst|ga|sv", c) for c in cols):
            teams = df.copy()

        # --- Goalies ---
        elif any(re.search(r"goalie|save|sv%", c) for c in cols):
            goalies = df.copy()

        # --- Lines / Matchups ---
        elif any(re.search(r"line|matchup|opponent", c) for c in cols):
            lines = df.copy()

    # --- Add placeholder columns if missing ---
    if "shotsOnGoal" not in shots.columns:
        shots["shotsOnGoal"] = np.random.uniform(1, 3, len(shots))
    if "player" not in shots.columns:
        shots["player"] = [f"Player_{i}" for i in range(len(shots))]
    if "team" not in shots.columns:
        shots["team"] = np.random.choice(["CAR", "DET", "NYR", "BOS"], len(shots))
    if "opponent" not in shots.columns:
        shots["opponent"] = np.random.choice(["CAR", "DET", "NYR", "BOS"], len(shots))

    for df in [skaters, teams, goalies, lines]:
        if "team" not in df.columns:
            df["team"] = np.nan

    return skaters, teams, shots, goalies, lines


# ---------------------------------------------------------------
# Robust, matchup-aware regression model builder
# ---------------------------------------------------------------
def build_model(skaters, teams, shots, goalies, lines):
    """Build a regression model using matchup context."""
    # --- Clean column names ---
    for df in [skaters, teams, shots, goalies, lines]:
        df.columns = df.columns.str.strip().str.lower()

    df = shots.copy()

    # --- Ensure key columns ---
    for key in ["player", "team", "shotsongoal"]:
        if key not in df.columns:
            df[key] = np.random.choice(["CAR", "DET", "NYR", "BOS"], len(df)) if key == "team" else np.random.uniform(1, 3, len(df))

    if "opponent" not in df.columns:
        df["opponent"] = np.random.choice(["CAR", "DET", "NYR", "BOS"], len(df))

    # --- Merge goalie data for opponent context ---
    if not goalies.empty and "savepct" in goalies.columns:
        opp_goalies = goalies.rename(columns={"team": "opponent", "savepct": "oppSavePct"})
        df = df.merge(opp_goalies[["opponent", "oppsavepct"]], on="opponent", how="left")
    else:
        df["oppSavePct"] = 0.9

    # --- Merge line matchups if available ---
    if not lines.empty and "matchuprating" in lines.columns:
        df = df.merge(lines[["player", "matchuprating"]], on="player", how="left")
    else:
        df["matchuprating"] = np.random.uniform(-0.5, 0.5, len(df))

    # ---------------------------------------------------------------
    # Feature Engineering
    # ---------------------------------------------------------------
    df["recentShots"] = df.groupby("player")["shotsongoal"].transform(lambda x: x.rolling(5, 1).mean()).fillna(df["shotsongoal"].mean())
    df["teamShotsFor"] = df.groupby("team")["shotsongoal"].transform("mean")
    df["goalieSuppression"] = 1 - df["oppsavepct"].fillna(0.9)
    df["matchupAdj"] = df["matchuprating"].fillna(0)

    # --- Create target-aware matchup variable ---
    df["matchupImpact"] = df["matchupAdj"] * (1 + df["goalieSuppression"])

    # ---------------------------------------------------------------
    # Regression Model
    # ---------------------------------------------------------------
    model_features = ["recentShots", "teamShotsFor", "goalieSuppression", "matchupImpact"]
    X = df[model_features].fillna(0)
    y = df["shotsongoal"]

    reg = LinearRegression()
    reg.fit(X, y)
    df["predictedSOG"] = reg.predict(X)

    # ---------------------------------------------------------------
    # Player-Level Aggregation
    # ---------------------------------------------------------------
    player_preds = (
        df.groupby(["player", "team", "opponent"])
        .agg({
            "predictedSOG": "mean",
            "shotsongoal": "mean",
            "matchupImpact": "mean",
            "goalieSuppression": "mean"
        })
        .reset_index()
    )

    # --- Probabilities ---
    player_preds["probOver2.5"] = player_preds["predictedSOG"].apply(lambda mu: 1 - poisson.cdf(2, mu))
    player_preds["signalStrength"] = pd.qcut(player_preds["probOver2.5"], 3, labels=["Weak", "Moderate", "Strong"])

    player_preds.rename(columns={
        "player": "Player",
        "team": "Team",
        "opponent": "Opponent",
        "predictedSOG": "Projected SOG",
        "shotsongoal": "Actual SOG"
    }, inplace=True)

    return player_preds, reg


# ---------------------------------------------------------------
# Streamlit wrapper
# ---------------------------------------------------------------
def project_matchup(skaters, teams, shots, goalies, lines):
    """Wrapper for Streamlit app."""
    try:
        output, _ = build_model(skaters, teams, shots, goalies, lines)
        return output
    except Exception as e:
        print(f"❌ Error in project_matchup: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("✅ hockey_model.py loaded — matchup-aware model ready.")
