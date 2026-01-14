# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop — Team-based matchup model with line strength
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import zscore


# ---------------------------------------------------------------
# 1️⃣ Parse and clean uploaded files
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """
    Reads uploaded dataframes, cleans, and returns standardized versions.
    """
    skaters = file_dfs.get("skaters", pd.DataFrame())
    teams = file_dfs.get("teams", pd.DataFrame())
    shots = file_dfs.get("shots", pd.DataFrame())
    goalies = file_dfs.get("goalies", pd.DataFrame())
    lines = file_dfs.get("lines", pd.DataFrame())

    # Clean and normalize
    for df in [skaters, teams, shots, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip()
            for col in ["team", "teamCode"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.upper()
            for col in ["name", "shooterName"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

    # Build team list for dropdowns
    team_list = sorted(skaters["team"].dropna().unique().tolist()) if "team" in skaters else []

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

    if "shotWasOnGoal" not in df.columns:
        print("⚠️ Missing shotWasOnGoal column.")
        return pd.DataFrame()

    # Normalize shotWasOnGoal values
    df["shotWasOnGoal"] = (
        df["shotWasOnGoal"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
    )

    if "game_id" not in df.columns:
        print("⚠️ No game_id found.")
        return pd.DataFrame()

    df = df.sort_values(["player", "game_id"])
    grouped = (
        df.groupby(["player", "team", "game_id"])
        .agg({"shotWasOnGoal": "sum"})
        .reset_index()
    )

    # Rolling windows
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
    Prepares team and goalie suppression metrics.
    """
    if teams_df.empty:
        team_context = pd.DataFrame(columns=["team", "shotSuppression", "xGoalsFor"])
    else:
        team_context = teams_df.copy()
        found = [c for c in team_context.columns if "goal" in c.lower() and "against" in c.lower()]
        shot_col = found[0] if found else None
        team_context["shotSuppression"] = team_context[shot_col] if shot_col else np.nan
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
# 4️⃣ Compute team-based line strength
# ---------------------------------------------------------------
def build_team_line_strength(lines_df):
    """
    Computes per-team average line strength normalized from xGoalsFor/xGoalsAgainst.
    """
    if lines_df.empty:
        print("⚠️ No line data provided — skipping line strength.")
        return pd.DataFrame(columns=["team", "lineStrength"])

    df = lines_df.copy()
    if "xGoalsFor" not in df.columns or "xGoalsAgainst" not in df.columns:
        print("⚠️ Missing xGoalsFor/xGoalsAgainst in lines.csv.")
        return pd.DataFrame(columns=["team", "lineStrength"])

    df["raw_strength"] = (
        (df["xGoalsFor"] - df["xGoalsAgainst"])
        / (df["xGoalsFor"] + df["xGoalsAgainst"] + 1e-6)
    )
    team_strength = df.groupby("team")["raw_strength"].mean().reset_index()
    team_strength["lineStrength"] = (
        (team_strength["raw_strength"] - team_strength["raw_strength"].mean())
        / team_strength["raw_strength"].std()
    ).clip(-2, 2)

    print(f"🏒 Computed lineStrength for {len(team_strength)} teams.")
    return team_strength[["team", "lineStrength"]]


# ---------------------------------------------------------------
# 5️⃣ Build final projections table
# ---------------------------------------------------------------
def build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b):
    print(f"🔍 Building matchup model for {team_a} vs {team_b}")

    player_form = build_player_form(shots)
    team_ctx, goalie_ctx = build_team_goalie_context(teams, goalies)
    line_strength = build_team_line_strength(lines)

    if player_form.empty:
        print("❌ No player form data — aborting.")
        return pd.DataFrame()

    # Filter only selected teams
    form = player_form[player_form["team"].isin([team_a, team_b])].copy()
    if form.empty:
        print("⚠️ No players found for selected teams.")
        return player_form.head(25)

    # Merge with team, goalie, and line context
    merged = (
        form.merge(team_ctx, on="team", how="left")
        .merge(line_strength, on="team", how="left")
    )

    # Add opponent context
    merged["opponent"] = np.where(merged["team"] == team_a, team_b, team_a)
    opp_strength = line_strength.rename(columns={"team": "opponent", "lineStrength": "oppLineStrength"})
    merged = merged.merge(opp_strength, on="opponent", how="left")

    # Matchup adjustment — relative line strength difference
    merged["matchupAdj"] = merged["lineStrength"] - merged["oppLineStrength"]

    # Get goalie suppression for opponent
    opp_goalie = goalie_ctx.rename(columns={"team": "opponent", "goalieSuppression": "oppGoalieSuppression"})
    merged = merged.merge(opp_goalie, on="opponent", how="left")

    # ---------------------------------------------------------------
    # Weighted Projection Formula
    # ---------------------------------------------------------------
    merged["xGoalsFor"] = merged["xGoalsFor"].fillna(merged["avg_5"])
    merged["oppGoalieSuppression"] = merged["oppGoalieSuppression"].fillna(0.9)
    merged["matchupAdj"] = merged["matchupAdj"].fillna(0)

    merged["Projected_SOG"] = (
        0.4 * merged["avg_5"]
        + 0.25 * merged["xGoalsFor"]
        + 0.2 * (1 - merged["oppGoalieSuppression"])
        + 0.15 * (1 + merged["matchupAdj"])
    )

    merged["Projected_SOG"] = merged["Projected_SOG"].clip(lower=0).round(2)

    merged["SignalStrength"] = pd.cut(
        merged["z_score"], bins=[-np.inf, 0, 1, np.inf], labels=["Weak", "Moderate", "Strong"]
    )

    result = merged[
        [
            "player",
            "team",
            "opponent",
            "avg_3",
            "avg_5",
            "avg_10",
            "avg_20",
            "z_score",
            "lineStrength",
            "oppLineStrength",
            "matchupAdj",
            "oppGoalieSuppression",
            "Projected_SOG",
            "SignalStrength",
        ]
    ].sort_values("Projected_SOG", ascending=False)

    print(f"✅ Generated {len(result)} player projections.")
    if result.empty:
        print("⚠️ Returning fallback form table.")
        return player_form.head(25)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------
# 6️⃣ Wrapper for Streamlit app
# ---------------------------------------------------------------
def project_matchup(skaters, teams, shots, goalies, lines, team_a, team_b):
    try:
        return build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b)
    except Exception as e:
        print(f"❌ Error in project_matchup: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------
# Example standalone run
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("✅ hockey_model.py loaded — line-strength matchup mode.")
