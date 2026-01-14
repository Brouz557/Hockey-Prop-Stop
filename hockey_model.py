# ---------------------------------------------------------------
# hockey_model.py (debug edition)
# Hockey Prop Stop — Debug version with detailed prints
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import re

# ---------------------------------------------------------------
# Helper: normalize team names
# ---------------------------------------------------------------
def normalize_team_name(name):
    if pd.isna(name):
        return np.nan
    s = str(name).strip().upper()
    s = re.sub(r"[^A-Z]", "", s)
    return s[:3]

# ---------------------------------------------------------------
# Parse uploaded files
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    skaters = file_dfs.get("skaters", pd.DataFrame())
    teams = file_dfs.get("teams", pd.DataFrame())
    shots = file_dfs.get("shots", pd.DataFrame())
    goalies = file_dfs.get("goalies", pd.DataFrame())
    lines = file_dfs.get("lines", pd.DataFrame())

    for df in [skaters, teams, shots, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip()
            if any(c for c in df.columns if c.lower().startswith("team")):
                col = [c for c in df.columns if c.lower().startswith("team")][0]
                df["team"] = df[col].astype(str).str.strip().str.upper()

    team_list = sorted(
        skaters["team"].dropna().unique().tolist()
    ) if "team" in skaters.columns else []
    print("✅ parse_raw_files complete")
    print("Teams detected:", team_list)
    return skaters, teams, shots, goalies, lines, team_list

# ---------------------------------------------------------------
# Build player form (with debug prints)
# ---------------------------------------------------------------
def build_player_form(shots_df):
    if shots_df.empty:
        print("⚠️ build_player_form: shots file empty.")
        return pd.DataFrame()

    df = shots_df.copy()
    print("DEBUG: shots_df columns:", df.columns.tolist()[:15])
    df = df.rename(columns={"shooterName": "player", "teamCode": "team"})

    # detect truncated shotWasOnGoal column (e.g., 'shotWasO')
    col_candidates = [c for c in df.columns if "shotWasOn" in c or "onGoal" in c]
    if not col_candidates:
        print("❌ No 'shotWasOnGoal' type column found — aborting player form build.")
        return pd.DataFrame()

    on_goal_col = col_candidates[0]
    if on_goal_col != "shotWasOnGoal":
        print(f"⚠️ Renaming detected column '{on_goal_col}' → 'shotWasOnGoal'")
        df.rename(columns={on_goal_col: "shotWasOnGoal"}, inplace=True)

    df["shotWasOnGoal"] = (
        df["shotWasOnGoal"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
