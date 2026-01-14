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
        df.columns = df.columns.str.strip()

    # --- Merge datasets ---
    df = shots.copy()

    # Rename the shooter column for consistency
    if "shooterName" in df.columns:
        df = df.rename(columns={"shooterName": "player"})

    # Merge skater info
    if "player" in skaters.columns and "team" in skaters.columns:
        df = df.merge(
            skaters[["player", "team", "position"]],
            on=["player", "team"],
            how="left"
        )

    # Merge team-level data (auto-detect shots against column)
    if "team" in teams.columns:
        shots_allowed_col = None
        for c in teams.columns:
            if "shot" in c.lower() and "against" in c.lower():
                shots_allowed_col = c
                break

        if shots_allowed_col:
            df = df.merge(
                teams[["team", shots_allowed_col]].rename(columns={shots_allowed_col: "shotsOnGoalAgainst"}),
                on="team",
                how="left"
            )
        else:
            teams["shotsOnGoalAgainst"] = np.nan
            df = df.merge(teams[["team", "shotsOnGoalAgainst"]], on="team", how="left")

    # Merge goalie data (auto-detec
