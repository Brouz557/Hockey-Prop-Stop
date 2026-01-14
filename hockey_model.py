# ---------------------------------------------------------------
# hockey_model.py (patched version)
# Hockey Prop Stop — resilient multi-file matchup model
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import zscore
import re

# ---------------------------------------------------------------
# 1️⃣ Helper: normalize team names and codes
# ---------------------------------------------------------------
def normalize_team_name(name):
    """
    Standardizes team names to 3-letter abbreviations.
    Handles cases like 'Toronto', 'Tor', 'TORONTO MAPLE LEAFS' -> 'TOR'
    """
    if pd.isna(name):
        return np.nan

    s = str(name).strip().upper()
    # Common replacements
    mapping = {
        "TORONTO": "TOR", "TORONTO MAPLE LEAFS": "TOR", "TOR": "TOR", "TORO": "TOR",
        "BOSTON": "BOS", "BOS": "BOS",
        "EDMONTON": "EDM", "EDM": "EDM",
        "NEW JERSEY": "NJD", "NJ": "NJD", "N.J.": "NJD", "DEVILS": "NJD",
        "TAMPA BAY": "TBL", "TAMPA": "TBL", "TB": "TBL",
        "VEGAS": "VGK", "VGK": "VGK",
        "LOS ANGELES": "LAK", "LA": "LAK", "KINGS": "LAK",
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
        "SAN JOSE": "SJS", "SJ": "SJS", "SJS": "SJS",
        "ST LOUIS": "STL", "ST. LOUIS": "STL", "STL": "STL",
        "PITTSBURGH": "PIT", "PIT": "PIT",
        "BUFFALO": "BUF", "BUF": "BUF",
        "DETROIT": "DET", "DET": "DET"
    }
    return mapping.get(s, s[:3])


# ---------------------------------------------------------------
# 2️⃣ Parse and clean uploaded files (patched version)
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """
    Reads uploaded dataframes, cleans, aligns, and returns standardized versions.
    Handles flexible column names, missing fields, and inconsistent teams.
    """

    # Load incoming data
    skaters = file_dfs.get("skaters", pd.DataFrame())
    teams = file_dfs.get("teams", pd.DataFrame())
    shots = file_dfs.get("shots", pd.DataFrame())
    goalies = file_dfs.get("goalies", pd.DataFrame())
    lines = file_dfs.get("lines", pd.DataFrame())

    # --- Clean headers
    for df in [skaters, teams, shots, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip().str.replace(" ", "_")
            df.columns = [re.sub(r"[^A-Za-z0-9_]", "", c) for c in df.columns]
            # Normalize team field
            if any(c for c in df.columns if c.lower().startswith("team")):
                team_col = [c for c in df.columns if c.lower().startswith("team")][0]
                df["team"] = df[team_col].apply(normalize_team_name)

    # --- Specific key renames
    if "teamCode" in shots.columns:
        shots["team"] = shots["teamCode"].apply(normalize_team_name)
    if "shooterName" not in shots.columns:
        # best-effort match if naming differs
        alt = [c for c in shots.columns if "shooter" in c.lower()]
        if alt:
            shots.rename(columns={alt[0]: "shooterName"}, inplace=True)

    # --- Team list for UI dropdowns
    all_teams = []
    if not skaters.empty and "team" in skaters.columns:
        all_teams = sorted(skaters["team"].dropna().unique().tolist())
    elif not teams.empty and "team" in teams.columns:
        all_teams = sorted(teams["team"].dropna().unique().tolist())

    # Debug summary
    print("✅ parse_raw_files: loaded datasets")
    print(f"   Skaters: {len(skaters)} | Teams: {len(teams)} | Shots: {len(shots)} | Goalies: {len(goalies)} | Lines: {len(lines)}")
    print("   Teams detected:", all_teams)

    return skaters, teams, shots, goalies, lines, all_teams


# ---------------------------------------------------------------
# 3️⃣ Matchup and projection functions
# (unchanged logic from your previous build)
# ---------------------------------------------------------------
def build_player_form(shots_df):
    if shots_df.empty or "shooterName" not in shots_df.columns:
        print("⚠️ build_player_form: shots file empty or missing shooterName.")
        return pd.DataFrame()

    df = shots_df.copy()
    df = df.rename(columns={"shooterName": "player", "teamCode": "team"})

    # Standardize team names
    df["team"] = df["team"].apply(normalize_team_name)
    if "shotWasOnGoal" not in df.columns:
        print("⚠️ Missing shotWasOnGoal column.")
        return pd.DataFrame()

    df["shotWasOnGoal"] = df["shotWasOnGoal"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
    if "game_id" not in df.columns:
        print("⚠️ No game_id found.")
        return pd.DataFrame()

    df = df.sort_values(["player", "game_id"])
    grouped = (
        df.groupby(["player", "team", "game_id"])
        .agg({"shotWasOnGoal": "sum"})
        .reset_index()
    )

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
    return latest[["player", "team", "avg_3", "avg_5", "avg_10", "avg_20", "z_score"]]


def build_team_goalie_context(teams_df, goalies_df):
    if teams_df.empty:
        team_context = pd.DataFrame(columns=["team", "shotSuppression", "xGoalsFor"])
    else:
        team_context = teams_df.copy()
        found = [c for c in team_context.columns if "goal" in c.lower() and "against" in c.lower()]
        shot_col = found[0] if found else None
        team_context["shotSuppression"] = team_context[shot_col] if shot_col else np.nan
        if "xGoalsFor" not in team_context.columns:
            xgf = [c for c in team_context.columns if "xGoalsFor" in c or "xGoals" in c]
            if xgf:
                team_context["xGoalsFor"] = team_context[xgf[0]]
            else:
                team_context["xGoalsFor"] = np.nan
        team_context = team_context[["team", "shotSuppression", "xGoalsFor"]]

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


def build_team_line_strength(lines_df):
    if lines_df.empty or "xGoalsFor" not in lines_df.columns or "xGoalsAgainst" not in lines_df.columns:
        print("⚠️ No line data provided — skipping line strength.")
        return pd.DataFrame(columns=["team", "lineStrength"])

    df = lines_df.copy()
    df["raw_strength"] = (
        (df["xGoalsFor"] - df["xGoalsAgainst"]) /
        (df["xGoalsFor"] + df["xGoalsAgainst"] + 1e-6)
    )
    team_strength = df.groupby("team")["raw_strength"].mean().reset_index()
    team_strength["lineStrength"] = (
        (team_strength["raw_strength"] - team_strength["raw_strength"].mean())
        / team_strength["raw_strength"].std()
    ).clip(-2, 2)

    print(f"🏒 Computed lineStrength for {len(team_strength)} teams.")
    return team_strength[["team", "lineStrength"]]


def build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b):
    print(f"🔍 Building matchup model for {team_a} vs {team_b}")

    player_form = build_player_form(shots)
    team_ctx, goalie_ctx = build_team_goalie_context(teams, goalies)
    line_strength = build_team_line_strength(lines)

    if player_form.empty:
        print("❌ No player form data — aborting.")
        return pd.DataFrame()

    form = player_form[player_form["team"].isin([team_a, team_b])].copy()
    if form.empty:
        print("⚠️ No players found for selected teams.")
        print("DEBUG: Teams present in player_form:", player_form["team"].unique())
        return player_form.head(25)

    merged = (
        form.merge(team_ctx, on="team", how="left")
        .merge(line_strength, on="team", how="left")
    )

    merged["opponent"] = np.where(merged["team"] == team_a, team_b, team_a)
    opp_strength = line_strength.rename(columns={"team": "opponent", "lineStrength": "oppLineStrength"})
    merged = merged.merge(opp_strength, on="opponent", how="left")

    merged["matchupAdj"] = merged["lineStrength"] - merged["oppLineStrength"]

    opp_goalie = goalie_ctx.rename(columns={"team": "opponent", "goalieSuppression": "oppGoalieSuppression"})
    merged = merged.merge(opp_goalie, on="opponent", how="left")

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
        merged["z_score"], bins=[-np.inf, 0, 1, np.inf],
        labels=["Weak", "Moderate", "Strong"]
    )

    result = merged[
        ["player", "team", "opponent", "avg_3", "avg_5", "avg_10", "avg_20",
         "z_score", "lineStrength", "oppLineStrength", "matchupAdj",
         "oppGoalieSuppression", "Projected_SOG", "SignalStrength"]
    ].sort_values("Projected_SOG", ascending=False)

    print(f"✅ Generated {len(result)} player projections.")
    return result.reset_index(drop=True)


def project_matchup(skaters, teams, shots, goalies, lines, team_a, team_b):
    try:
        return build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b)
    except Exception as e:
        print(f"❌ Error in project_matchup: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("✅ hockey_model.py loaded — normalized and validated version.")
