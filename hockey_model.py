# ---------------------------------------------------------------
# hockey_model.py (debug edition, fixed syntax)
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

    team_list = []
    if not skaters.empty and "team" in skaters.columns:
        team_list = sorted(skaters["team"].dropna().unique().tolist())

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
    )

    if "game_id" not in df.columns:
        print("⚠️ No game_id found — cannot compute rolling windows.")
        return pd.DataFrame()

    df["team"] = df["team"].astype(str).str.strip().str.upper()
    print("DEBUG: unique teams in shots file:", df["team"].unique()[:10])

    df = df.sort_values(["player", "game_id"])
    grouped = (
        df.groupby(["player", "team", "game_id"])
        .agg({"shotWasOnGoal": "sum"})
        .reset_index()
    )
    print("DEBUG: grouped rows:", grouped.shape)

    for w in [3, 5, 10, 20]:
        grouped[f"avg_{w}"] = (
            grouped.groupby("player")["shotWasOnGoal"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )

    grouped["baseline_20"] = grouped.groupby("player")["avg_20"].transform("mean")
    grouped["std_20"] = grouped.groupby("player")["avg_20"].transform("std").fillna(0.01)
    grouped["z_score"] = (grouped["avg_5"] - grouped["baseline_20"]) / grouped["std_20"]

    latest = grouped.groupby(["player", "team"]).tail(1).reset_index(drop=True)
    print(f"✅ build_player_form: computed form for {len(latest)} players.")
    print("DEBUG: sample player_form head:\n", latest.head(5))
    return latest[
        ["player", "team", "avg_3", "avg_5", "avg_10", "avg_20", "z_score"]
    ]

# ---------------------------------------------------------------
# Simplified context builders
# ---------------------------------------------------------------
def build_team_goalie_context(teams_df, goalies_df):
    return pd.Da
