# ---------------------------------------------------------------
# hockey_model.py
# Core analytics engine for Hockey Prop Stop / Hockey Bot
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
        df.columns = df.columns.str.strip()

    # --- Base: shots dataset ---
    df = shots.copy()

    # Rename shooter column for consistency
    if "shooterName" in df.columns:
        df = df.rename(columns={"shooterName": "player"})

    # --- Merge skater info ---
    if set(["player", "team"]).issubset(skaters.columns):
        df = df.merge(
            skaters[["player", "team", "position"]],
            on=["player", "team"],
            how="left"
        )

    # --- Merge team-level data (shots allowed, etc.) ---
    if "team" in teams.columns:
        shots_allowed_col = None
        for c in teams.columns:
            if "shot" in c.lower() and "against" in c.lower():
                shots_allowed_col = c
                break

        if shots_allowed_col:
            df = df.merge(
                teams[["team", shots_allowed_col]].rename(
                    columns={shots_allowed_col: "shotsOnGoalAgainst"}
                ),
                on="team",
                how="left"
            )
        else:
            teams["shotsOnGoalAgainst"] = np.nan
            df = df.merge(teams[["team", "shotsOnGoalAgainst"]], on="team", how="left")

    # --- Merge goalie suppression metrics ---
    goalie_key = None
    for c in goalies.columns:
        if "goalie" in c.lower() or "name" in c.lower():
            goalie_key = c
            break
    if goalie_key:
        df = df.merge(
            goalies[[goalie_key, "teamAgainst", "savePct", "shotsFaced"]].rename(
                columns={goalie_key: "goalie"}
            ),
            on="teamAgainst",
            how="left"
        )

    # --- Merge line matchup data ---
    if set(["player", "opponent", "line"]).issubset(lines.columns):
        df = df.merge(
            lines[["player", "opponent", "line", "matchupRating"]],
            on=["player", "opponent"],
            how="left"
        )

    # ---------------------------------------------------------------
    # Feature Engineering
    # ---------------------------------------------------------------
    df["recentShots"] = df.groupby("player")["shotsOnGoal"].transform(lambda x: x.rolling(5, 1).mean())
    df["teamShotsFor"] = df.groupby("team")["shotsOnGoal"].transform("mean")
    df["goalieSuppression"] = 1 - df["savePct"].fillna(0.9)
    df["matchupAdj"] = df["matchupRating"].fillna(0)
    df["xSOG"] = (
        0.5 * df["recentShots"].fillna(0) +
        0.3 * df["teamShotsFor"].fillna(0) +
        0.2 * df["matchupAdj"]
    )

    # ---------------------------------------------------------------
    # Regression Model: Predict SOG from matchup-driven features
    # ---------------------------------------------------------------
    model_features = ["recentShots", "teamShotsFor", "goalieSuppression", "matchupAdj"]
    model_df = df.dropna(subset=model_features + ["shotsOnGoal"]).copy()

    X = model_df[model_features]
    y = model_df["shotsOnGoal"]

    reg = LinearRegression()
    reg.fit(X, y)
    model_df["predictedSOG"] = reg.predict(X)

    # ---------------------------------------------------------------
    # Player-Level Aggregation
    # ---------------------------------------------------------------
    player_preds = (
        model_df.groupby("player")
        .agg({
            "predictedSOG": "mean",
            "shotsOnGoal": "mean",
            "xSOG": "mean",
            "matchupAdj": "mean",
            "goalieSuppression": "mean"
        })
        .reset_index()
    )

    # --- Calculate Poisson-based probabilities ---
    player_preds["probOver2.5"] = player_preds["predictedSOG"].apply(
        lambda mu: 1 - poisson.cdf(2, mu)
    )
    player_preds["probOver3.5"] = player_preds["predictedSOG"].apply(
        lambda mu: 1 - poisson.cdf(3, mu)
    )

    # --- Signal strength ---
    player_preds["signalStrength"] = pd.qcut(
        player_preds["probOver2.5"], 3, labels=["Weak", "Moderate", "Strong"]
    )

    # --- Matchup favorability ---
    player_preds["matchupFavorability"] = pd.cut(
        player_preds["matchupAdj"],
        bins=[-np.inf, -0.25, 0.25, np.inf],
        labels=["Unfavorable", "Neutral", "Favorable"]
    )

    # --- Lowest playable odds (fair odds approximation) ---
    player_preds["fairOddsOver2.5"] = 1 / player_preds["probOver2.5"]
    player_preds["fairOddsOver3.5"] = 1 / player_preds["probOver3.5"]

    # ---------------------------------------------------------------
    # Final Output
    # ---------------------------------------------------------------
    output = player_preds.sort_values("probOver2.5", ascending=False).reset_index(drop=True)
    return output, reg


# ---------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("hockey_model.py loaded — ready for analysis pipeline.")
