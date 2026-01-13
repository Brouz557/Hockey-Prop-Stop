# ---------------------------------------------------------------
# hockey_model.py
# Core analytics engine for Hockey Prop Stop
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson

# ---------------------------------------------------------------
# Build regression model from uploaded data
# ---------------------------------------------------------------
def build_model(skaters, teams, shots, goalies, lines):
    # --- Clean column names for consistency ---
    for df in [skaters, teams, shots, goalies, lines]:
        df.columns = df.columns.str.lower().str.strip()

    # --- Merge datasets ---
    df = shots.copy()
    if "player" in skaters.columns:
        df = df.merge(
            skaters[["player", "team", "position"]],
            on=["player", "team"],
            how="left"
        )
    if "team" in teams.columns:
        df = df.merge(
            teams[["team", "shots_allowed_per_game"]],
            on="team",
            how="left"
        )
    if "goalie" in goalies.columns:
        df = df.merge(
            goalies[["goalie", "team", "sog_allowed"]],
            on=["goalie", "team"],
            how="left"
        )
    if "line_matchup_score" in lines.columns:
        df = df.merge(
            lines[["player", "line_matchup_score"]],
            on="player",
            how="left"
        )

    # --- Sort and compute rolling features ---
    if "game_id" not in df.columns:
        df["game_id"] = np.arange(len(df))

    df = df.sort_values(["player", "game_id"])
    df["rolling3"] = df.groupby("player")["shots_on_goal"].transform(lambda x: x.rolling(3, 1).mean())
    df["rolling5"] = df.groupby("player")["shots_on_goal"].transform(lambda x: x.rolling(5, 1).mean())
    df["rolling10"] = df.groupby("player")["shots_on_goal"].transform(lambda x: x.rolling(10, 1).mean())
    df["rolling20"] = df.groupby("player")["shots_on_goal"].transform(lambda x: x.rolling(20, 1).mean())

    # --- Prepare model inputs ---
    feature_cols = ["rolling5", "rolling10", "shots_allowed_per_game", "sog_allowed", "line_matchup_score"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].fillna(0)
    y = df["shots_on_goal"].fillna(0)

    # Weight recent games more heavily using normalized game_id
    df["weight"] = df["game_id"].rank(pct=True)
    model = LinearRegression()
    model.fit(X, y, sample_weight=df["weight"])

    return model, df


# ---------------------------------------------------------------
# Generate projections for a specific matchup
# ---------------------------------------------------------------
def project_matchup(model, df, teamA, teamB):
    matchup_df = df[df["team"].isin([teamA, teamB])].drop_duplicates("player")
    if matchup_df.empty:
        return pd.DataFrame(columns=[
            "Player", "Team", "Pos", "Projected SOG", "Line",
            "Prob Over", "Signal", "Matchup", "Lowest Odds"
        ])

    feature_cols = ["rolling5", "rolling10", "shots_allowed_per_game", "sog_allowed", "line_matchup_score"]
    X = matchup_df[feature_cols].fillna(0)
    mu = model.predict(X)

    # Poisson-based probability of exceeding line
    matchup_df["Projected SOG"] = mu
    matchup_df["Line"] = np.where(matchup_df["position"] == "d", 1.5, 2.5)
    matchup_df["Prob Over"] = 1 - poisson.cdf(matchup_df["Line"], mu)
    matchup_df["Signal"] = np.where(
        matchup_df["Prob Over"] >= 0.7, "Strong",
        np.where(matchup_df["Prob Over"] >= 0.55, "Moderate", "Weak")
    )
    matchup_df["Lowest Odds"] = (1 / matchup_df["Prob Over"] - 1) * 100

    # Matchup favorability
    league_avg = df["shots_on_goal"].mean()
    matchup_df["Matchup"] = np.where(
        matchup_df["shots_allowed_per_game"] > league_avg, "Favorable", "Unfavorable"
    )

    output = matchup_df[[
        "player", "team", "position",
        "Projected SOG", "Line", "Prob Over",
        "Signal", "Matchup", "Lowest Odds",
        "rolling3", "rolling5", "rolling10", "rolling20"
    ]].rename(columns={
        "player": "Player",
        "team": "Team",
        "position": "Pos"
    })

    # Rank by signal strength and probability
    signal_order = {"Strong": 0, "Moderate": 1, "Weak": 2}
    output["rank"] = output["Signal"].map(signal_order)
    output = output.sort_values(["rank", "Prob Over"], ascending=[True, False]).drop(columns="rank")
    return output
