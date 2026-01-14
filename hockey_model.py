# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop — Team-based matchup model with full validation
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import zscore
import re

# ---------------------------------------------------------------
# 1️⃣ Helper: normalize team names
# ---------------------------------------------------------------
def normalize_team_name(name):
    if pd.isna(name):
        return np.nan
    s = str(name).strip().upper()
    mapping = {
        "TORONTO": "TOR", "TORONTO MAPLE LEAFS": "TOR", "TOR": "TOR",
        "BOSTON": "BOS", "BOS": "BOS",
        "EDMONTON": "EDM", "EDM": "EDM",
        "NEW JERSEY": "NJD", "NJ": "NJD", "N.J.": "NJD",
        "TAMPA BAY": "TBL", "TAMPA": "TBL", "TB": "TBL",
        "VEGAS": "VGK", "VGK": "VGK",
        "LOS ANGELES": "LAK", "LA": "LAK",
        "COLORADO": "COL", "COL": "COL",
        "FLORIDA": "FLA", "FLA": "FLA",
        "NY RANGERS": "NYR", "RANGERS": "NYR",
        "NY ISLANDERS": "NYI", "ISLANDERS": "NYI",
        "CHICAGO": "CHI", "CHI": "CHI",
        "VANCOUVER": "VAN", "VAN": "VAN",
        "MONTREAL": "MTL", "MTL": "MTL",
        "WINNIPEG": "WPG", "WPG": "WPG",
        "OTTAWA": "OTT", "OTT": "OTT",
        "DALLAS": "DAL", "DAL": "DAL",
        "CALGARY": "CGY", "CGY": "CGY",
        "SEATTLE": "SEA", "SEA": "SEA",
        "WASHINGTON": "WSH", "WSH": "WSH",
        "PHILADELPHIA": "PHI", "PHI": "PHI",
        "NASHVILLE": "NSH", "NSH": "NSH",
        "ARIZONA": "ARI", "ARI": "ARI",
        "COLUMBUS": "CBJ", "CBJ": "CBJ",
        "MINNESOTA": "MIN", "MIN": "MIN",
        "ANAHEIM": "ANA", "ANA": "ANA",
        "SAN JOSE": "SJS", "SJS": "SJS",
        "ST LOUIS": "STL", "ST. LOUIS": "STL", "STL": "STL",
        "PITTSBURGH": "PIT", "PIT": "PIT",
        "BUFFALO": "BUF", "BUF": "BUF",
        "DETROIT": "DET", "DET": "DET"
    }
    return mapping.get(s, s[:3])

# ---------------------------------------------------------------
# 2️⃣ Parse and clean uploaded files
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    skaters = file_dfs.get("skaters", pd.DataFrame())
    teams = file_dfs.get("teams", pd.DataFrame())
    shots = file_dfs.get("shots", pd.DataFrame())
    goalies = file_dfs.get("goalies", pd.DataFrame())
    lines = file_dfs.get("lines", pd.DataFrame())

    for df in [skaters, teams, shots, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip().str.replace(" ", "_")
            df.columns = [re.sub(r"[^A-Za-z0-9_]", "", c) for c in df.columns]
            if any(c for c in df.columns if c.lower().startswith("team")):
                team_col = [c for c in df.columns if c.lower().startswith("team")][0]
                df["team"] = df[team_col].apply(normalize_team_name)

    # Build team list for dropdowns
    team_list = []
    if not skaters.empty and "team" in skaters.columns:
        team_list = sorted(skaters["team"].dropna().unique().tolist())
    elif not teams.empty and "team" in teams.columns:
        team_list = sorted(teams["team"].dropna().unique().tolist())

    print("✅ parse_raw_files complete")
    return skaters, teams, shots, goalies, lines, team_list

# ---------------------------------------------------------------
# 3️⃣ Build rolling player form table (patched)
# ---------------------------------------------------------------
def build_player_form(shots_df):
    if shots_df.empty:
        print("⚠️ build_player_form: shots file empty.")
        return pd.DataFrame()

    df = shots_df.copy()
    df = df.rename(columns={"shooterName": "player", "teamCode": "team"})

    # --- Detect on-goal column (handles 'shotWasO')
    col_candidates = [c for c in df.columns if "shotWasOn" in c or "onGoal" in c]
    if not col_candidates:
        print("❌ No on-goal column found — aborting.")
        print("Available columns:", df.columns.tolist())
        return pd.DataFrame()

    on_goal_col = col_candidates[0]
    if on_goal_col != "shotWasOnGoal":
        print(f"⚠️ Renaming detected column '{on_goal_col}' → 'shotWasOnGoal'")
        df.rename(columns={on_goal_col: "shotWasOnGoal"}, inplace=True)

    # Normalize shotWasOnGoal
    df["shotWasOnGoal"] =
