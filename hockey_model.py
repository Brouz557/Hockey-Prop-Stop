# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop — Matchup-based SOG model (final debug build)
# ---------------------------------------------------------------

import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 1️⃣ Parse and clean uploaded files
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """Reads uploaded dataframes, cleans, and returns standardized versions."""
    skaters = file_dfs.get("skaters", pd.DataFrame())
    teams = file_dfs.get("teams", pd.DataFrame())
    shots = file_dfs.get("shots", pd.DataFrame())
    goalies = file_dfs.get("goalies", pd.DataFrame())
    lines = file_dfs.get("lines", pd.DataFrame())

    # --- basic cleaning
    for df in [skaters, teams, shots, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip()
            for col in ["team", "teamCode"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

    # --- normalize team names (make all short codes like TOR, BOS)
    def normalize_team_col(df, team_col):
        if team_col in df.columns:
            df[team_col] = (
                df[team_col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(" ", "", regex=False)
                .str.replace("MAPLELEAFS", "TOR", regex=False)
                .str.replace("BRUINS", "BOS", regex=False)
                .str.replace("LIGHTNING", "TBL", regex=False)
                .str.replace("PANTHERS", "FLA", regex=False)
                .str.replace("RANGERS", "NYR", regex=False)
                .str.replace("ISLANDERS", "NYI", regex=False)
                .str.replace("DEVILS", "NJD", regex=False)
                .str.replace("CANADIENS", "MTL", regex=False)
                .str.replace("SENATORS", "OTT", regex=False)
                .str.replace("SABRES", "BUF", regex=False)
            )
        return df

    skaters = normalize_team_col(skaters, "team")
    teams = normalize_team_col(teams, "team")
    shots = normalize_team_col(shots, "teamCode")
    goalies = normalize_team_col(goalies, "team")
    lines = normalize_team_col(lines, "team")

    # --- team list for dropdowns
    team_list = sorted(skaters["team"].dropna().unique().tolist()) if "team" in skaters else []

    print(f"✅ Parsed data | Teams detected: {team_list}")
    return skaters, teams, shots, goalies, lines, team_list


# ---------------------------------------------------------------
# 2️⃣ Player rolling form
# ---------------------------------------------------------------
def build_player_form(shots_df):
    """Build rolling player form metrics from shots.csv."""
    if shots_df.empty:
        print("⚠️ build_player_form: shots file empty.")
        return pd.DataFrame()

    df = shots_df.copy()

    # rename player/team safely
    if "shooterName" in df.columns:
        df.rename(columns={"shooterName": "player"}, inplace=True)
    if "teamCode" in df.columns and "team" not in df.columns:
        df.rename(columns={"teamCode": "team"}, inplace=True)

    # drop duplicate 'team' columns if they exist
    while list(df.columns).count("team") > 1:
        dup_index = [i for i, c in enumerate(df.columns) if c == "team"][1]
        df.drop(df.columns[dup_index], axis=1, inplace=True)

    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip().str.upper()

    # find the on-goal column
    og_candidates = [c for c in df.columns if "shotwason" in c.lower() or "ongoal" in c.lower()]
    if not og_candidates:
        print("❌ build_player_form: no on-goal column found.")
        print("Available columns:", df.columns.tolist()[:40])
        return pd.DataFrame()

    on_goal_col = og_candidates[0]
    if on_goal_col != "shotWasOnGoal":
        print(f"ℹ️ Using '{on_goal_col}' as shotWasOnGoal")
        df.rename(columns={on_goal_col: "shotWasOnGoal"}, inplace=True)

    df["shotWasOnGoal"] = df["shotWasOnGoal"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)

    if "game_id" not in df.columns:
        print("❌ build_player_form: missing game_id.")
        return pd.DataFrame()

    # ensure game_id numeric
    if not pd.api.types.is_numeric_dtype(df["game_id"]):
        df["game_id"] = pd.factorize(df["game_id"])[0] + 1

    df = df.sort_values(["player", "game_id"])
    grouped = df.groupby(["player", "team", "game_id"], as_index=False)["shotWasOnGoal"].]()
