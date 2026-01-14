# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop — Learning Regression Version (Final)
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

    # Basic cleanup
    for df in [skaters, teams, shots, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip()
            for col in ["team", "teamCode"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

    # Normalize team names
    def normalize_team(df, col):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(" ", "", regex=False)
                .replace({
                    "MAPLELEAFS": "TOR",
                    "BRUINS": "BOS",
                    "LIGHTNING": "TBL",
                    "PANTHERS": "FLA",
                    "RANGERS": "NYR",
                    "ISLANDERS": "NYI",
                    "DEVILS": "NJD",
                    "CANADIENS": "MTL",
                    "SENATORS": "OTT",
                    "SABRES": "BUF",
                })
            )
        return df

    skaters = normalize_team(skaters, "team")
    teams = normalize_team(teams, "team")
    shots = normalize_team(shots, "teamCode")
    goalies = normalize_team(goalies, "team")
    lines = normalize_team(lines, "team")

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

    if "shooterName" in df.columns:
        df.rename(columns={"shooterName": "player"}, inplace=True)
    if "teamCode" in df.columns and "team" not in df.columns:
        df.rename(columns={"teamCode": "team"}, inplace=True)

    # Drop duplicate team columns
    while list(df.columns).count("team") > 1:
        dup = [i for i, c in enumerate(df.columns) if c == "team"][1]
        df.drop(df.columns[dup], axis=1, inplace=True)

    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip().str.upper()

    # Detect shotWasOnGoal column
    og_cols = [c for c in df.columns if "shotwason" in c.lower() or "ongoal" in c.lower()]
    if not og_cols:
        print("❌ build_player_form: no shotWasOnGoal column found.")
        return pd.DataFrame()

    on_goal_col = og_cols[0]
    if on_goal_col != "shotWasOnGoal":
        df.rename(columns={on_goal_col: "shotWasOnGoal"}, inplace=True)

    df["shotWasOnGoal"] = df["shotWasOnGoal"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)

    if "game_id" not in df.columns:
        df["game_id"] = pd.factorize(df.index)[0] + 1

    df = df.sort_values(["player", "game_id"])
    grouped = df.groupby(["player", "team", "game_id"], as_index=False)["shotWasOnGoal"].sum()

    for w in [3, 5, 10, 20]:
        grouped[f"avg_{w}"] = grouped.groupby("player")["shotWasOnGoal"].transform(lambda x: x.rolling(w, 1).mean())

    grouped["baseline_20"] = grouped.groupby("player")["avg_20"].transform("mean")
    grouped["std_20"] = grouped.groupby("player")["avg_20"].transform("std").fillna(0.01)
    grouped["z_score"] = (grouped["avg_5"] - grouped["baseline_20"]) / grouped["std_20"]

    latest = grouped.groupby(["player", "team"], as_index=False).tail(1)
    print(f"✅ build_player_form: computed rolling form for {len(latest)} players.")
    return latest[["player", "team", "avg_3", "avg_5", "avg_10", "avg_20", "z_score"]]


# ---------------------------------------------------------------
# 3️⃣ Team & Goalie Context
# ---------------------------------------------------------------
def build_team_goalie_context(teams_df, goalies_df):
    if teams_df.empty:
        team_context = pd.DataFrame(columns=["team", "shotSuppression", "xGoalsFor"])
    else:
        team_context = teams_df.copy()
        found = [c for c in team_context.columns if "goal" in c.lower() and "against" in c.lower()]
        col = found[0] if found else None
        team_context["shotSuppression"] = team_context[col] if col else np.nan
        if "xGoalsFor" not in team_context.columns:
            team_context["xGoalsFor"] = np.nan
        team_context = team_context[["team", "shotSuppression", "xGoalsFor"]]

    if not goalies_df.empty:
        g = goalies_df.copy()
        g["savePct"] = 1 - (g["goals"] / g["ongoal"].replace(0, np.nan))
        g["dangerSavePct"] = 1 - (
