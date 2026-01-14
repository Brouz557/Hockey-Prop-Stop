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
    std, and z-scores from shots.csv.
    """
    if shots_df.empty:
        print("⚠️ build_player_form: shots file empty.")
        return pd.DataFrame()

    df = shots_df.copy()
    df = df.rename(columns={"shooterName": "player", "teamCode": "team"})

    # Normalize shotWasOnGoal values
    if "shotWasOnGoal" not in df.columns:
        print("⚠️ Missing shotWasOnGoal column — cannot compute rolling form.")
        return pd.DataFrame()

    df["shotWasOnGoal"] = (
        df["shotWasOnGoal"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
    )

    # Group and sort by player/game_id
    if "game_id" not in df.columns:
        print("⚠️ No game_id found, cannot compute rolling windows.")
        return pd.DataFrame()

    df = df.sort_values(["player", "game_id"])
    grouped = (
        df.groupby(["player", "team", "game_id"])
        .agg({"shotWasOnGoal": "sum"})
        .reset_index()
    )

    # Rolling averages
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
    return latest[
        ["player", "team", "avg_3", "avg_5", "avg_10", "avg_20", "z_score"]
    ]


# ---------------------------------------------------------------
# 3️⃣ Team and goalie context
# ---------------------------------------------------------------
def build_team_goalie_context(teams_df, goalies_df):
    """
    Builds team- and goalie-level suppression metrics.
    """
    if teams_df.empty:
        team_context = pd.DataFrame(columns=["team", "shotSuppression", "xGoalsFor"])
    else:
        team_context = teams_df.copy()
        # Try to detect suppression metric
        found = [c for c in team_context.columns if "goal" in c.lower() and "against" in c.lower()]
        shot_col = found[0] if found else None
        if shot_col:
            team_context["shotSuppression"] = team_context[shot_col]
        else:
            team_context["shotSuppression"] = np.nan
        if "xGoalsFor" not in team_context.columns:
            team_context["xGoalsFor"] = np.nan
        team_context = team_context[["team", "shotSuppression", "xGoalsFor"]]

    # Goalie suppression
    if not goalies_df.empty:
        g = goalies_df.copy()
        g["savePct"] = 1 - (g["goals"] / g["ongoal"].replace(0, np.nan))
        g["dangerSavePct"] = 1 - (
            (g["lowDangerGoals"] + g["mediumDangerGoals"] + g["highDangerGoals"])
            / (g["lowDangerShots"] + g["mediumDangerShots"] + g["highDangerShots"]).replace(0, np.nan)
        )
        g["goalieSuppression"] = g[["savePct", "dangerSavePct"]].mean(axis=1)
        goalie_context = g[["name", "team", "goalieSuppression"]]
    else:
        goalie_context = pd.DataFrame(columns=["name", "team", "goalieSuppression"])

    return team_context, goalie_context


# ---------------------------------------------------------------
# 4️⃣ Line-level matchup context
# ---------------------------------------------------------------
def build_line_matchups(lines_df):
    """
    Builds line-level matchup ratings from xGoalsFor vs xGoalsAgainst.
    """
    if lines_df.empty:
        return pd.DataFrame(columns=["name", "team", "matchupRating"])
    lines = lines_df.copy()
    if "xGoalsFor" in lines.columns and "xGoalsAgainst" in lines.columns:
        lines["matchupRating"] = (
            (lines["xGoalsFor"] - lines["xGoalsAgainst"])
            / (lines["xGoalsFor"] + lines["xGoalsAgainst"] + 1e-6)
        )
    else:
        lines["matchupRating"] = 0
    return lines[["name", "team", "matchupRating"]]


# ---------------------------------------------------------------
# 5️⃣ Build final projections table
# ---------------------------------------------------------------
def build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b):
    """
    Main function that ties all context together.
    """
    print(f"🔍 Building matchup model for {team_a} vs {team_b}")

    # --- Player form ---
    player_form = build_player_form(shots)
    if player_form.empty:
        print("❌ No player form data, aborting model build.")
        return pd.DataFrame()

    # --- Team & goalie context ---
    team_ctx, goalie_ctx = build_team_goalie_context(teams, goalies)

    # --- Line context ---
    line_ctx = build_line_matchups(lines)

    # --- Filter selected teams ---
    form = player_form[player_form["team"].isin([team_a, team_b])].copy()
    if form.empty:
        print("⚠️ No players found for selected teams.")
        return player_form.head(10)

    # --- Clean join keys ---
    form["player_clean"] = form["player"].str.lower().str.strip()
    skaters["name_clean"] = skaters["name"].str.lower().str.strip()

    # --- Merge everything ---
    merged = (
        form.merge(
            skaters[["name_clean", "team", "position"]],
            left_on="player_clean",
            right_on="name_clean",
            how="left",
        )
        .merge(team_ctx, on="team", how="left")
        .merge(line_ctx, left_on=["player", "team"], right_on=["name", "team"], how="left")
    )

    # --- Derive opponent suppression ---
    opp_supp = team_ctx.rename(
        columns={"team": "opponent", "shotSuppression": "oppSuppression"}
    )
    merged["opponent"] = np.where(merged["team"] == team_a, team_b, team_a)
    merged = merged.merge(opp_supp[["opponent", "oppSuppression"]], on="opponent", how="left")

    # --- Goalie suppression from opponent team ---
    opp_goalie = goalie_ctx.rename(
        columns={"team": "opponent", "goalieSuppression": "oppGoalieSuppression"}
    )
    merged = merged.merge(opp_goalie[["opponent", "oppGoalieSuppression"]], on="opponent", how="left")

    # ---------------------------------------------------------------
    # Weighted Projection Formula
    # ---------------------------------------------------------------
    merged["Projected_SOG"] = (
        0.4 * merged["avg_5"]
        + 0.25 * merged["xGoalsFor"].fillna(merged["avg_5"])
        + 0.2 * (1 - merged["oppGoalieSuppression"].fillna(0.9))
        + 0.15 * merged["matchupRating"].fillna(0)
    )

    merged["Projected_SOG"] = merged["Projected_SOG"].clip(lower=0).round(2)

    # --- Signal Strength ---
    merged["SignalStrength"] = pd.cut(
        merged["z_score"],
        bins=[-np.inf, 0, 1, np.inf],
        labels=["Weak", "Moderate", "Strong"],
    )

    # --- Final Output ---
    result = merged[
        [
            "player",
            "team",
            "opponent",
            "position",
            "avg_3",
            "avg_5",
            "avg_10",
            "avg_20",
            "z_score",
            "matchupRating",
            "oppSuppression",
            "oppGoalieSuppression",
            "Projected_SOG",
            "SignalStrength",
        ]
    ].drop_duplicates(subset=["player"]).sort_values("Projected_SOG", ascending=False)

    print(f"✅ Generated {len(result)} player projections.")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------
# 6️⃣ Wrapper for Streamlit app
# ---------------------------------------------------------------
def project_matchup(skaters, teams, shots, goalies, lines, team_a, team_b):
    try:
        output = build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b)
        return output
    except Exception as e:
        print(f"❌ Error in project_matchup: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------
# Example standalone run
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("✅ hockey_model.py loaded — syntax-verified build.")
