# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop — resilient production build with normalization
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import zscore


# ---------------------------------------------------------------
# 1️⃣ Parse and clean uploaded files
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """
    Accepts a dict of uploaded CSV dataframes.
    Returns (skaters, teams, shots, goalies, lines, team_list)
    """
    skaters = file_dfs.get("skaters", pd.DataFrame())
    teams = file_dfs.get("teams", pd.DataFrame())
    shots = file_dfs.get("shots", pd.DataFrame())
    goalies = file_dfs.get("goalies", pd.DataFrame())
    lines = file_dfs.get("lines", pd.DataFrame())

    # Standardize columns
    for df in [skaters, teams, shots, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip()

            # Normalize team and player name formats
            if "team" in df.columns:
                df["team"] = df["team"].astype(str).str.strip().str.upper()
            if "teamCode" in df.columns:
                df["teamCode"] = df["teamCode"].astype(str).str.strip().str.upper()
            if "name" in df.columns:
                df["name"] = df["name"].astype(str).str.strip()
            if "shooterName" in df.columns:
                df["shooterName"] = df["shooterName"].astype(str).str.strip()

    # Unique teams from skaters sheet
    team_list = sorted(skaters["team"].dropna().unique().tolist())

    return skaters, teams, shots, goalies, lines, team_list


# ---------------------------------------------------------------
# 2️⃣ Build rolling player form table
# ---------------------------------------------------------------
def build_player_form(shots_df):
    """
    Calculates player rolling 3/5/10/20 game averages,
    std, and z-score
