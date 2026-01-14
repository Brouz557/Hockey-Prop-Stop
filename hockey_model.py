# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop — adaptive NHL matchup analytics engine (v3)
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import poisson


# ---------------------------------------------------------------
# Parse uploaded CSVs
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """Auto-detects NHL data files and normalizes them."""
    skaters = teams = shots = goalies = lines = pd.DataFrame()

    for name, df in file_dfs.items():
        if df is None or df.empty:
            continue
        df.columns = df.columns.str.strip().str.lower()
        cols = " ".join(df.columns)

        if any(x in cols for x in ["shot", "attempt", "sog", "goalienameforshot"]):
            shots = df.copy()
        elif any(x in cols for x in ["player", "skater", "toi"]):
            skaters = df.copy()
        elif any(x in cols for x in ["team", "shotsagainst", "franchise"]):
            teams = df.copy()
        elif any(x in cols for x in ["goalie", "save", "sv%"]):
            goalies = df.copy()
        elif any(x in cols for x in ["line", "matchup", "opp"]):
            lines = df.copy()

    # Clean columns
    for df in [shots, skaters, teams, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip().str.lower()

    # --- detect team abbreviations ---
    NHL_TEAMS = {
        "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
        "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
        "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK",
        "WSH", "WPG"
    }

    team_cols = [c for c in shots.columns if "team" in c]
    all_teams = []
    for c in team_cols:
        vals = shots[c].dropna().unique().tolist()
        for v in vals:
            if isinstance(v, str) and v.upper() in NHL_TEAMS:
                all_teams.append(v.upper())
    all_teams = sorted(list(set(all_teams)))
    if len(all_teams) < 32:
        all_teams = sorted(list(NHL_TEAMS))

    return skaters, teams, shots, goalies, lines, all_teams


# ---------------------------------------------------------------
# Build matchup model (player-level projections)
# ---------------------------------------------------------------
def build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b):
    df = shots.copy()
    df.columns = df.columns.str.lower()

    # --- identify core columns ---
    player_col = next((c for c in df.columns if "shootername" in c or "playername" in c), None)
    team_col = next((c for c in df.columns if c == "team"), None)
    home_col = next((c for c in df.columns if "hometeamcode" in c), None)
    away_col = next((c for c in df.columns if "awayteamcode" in c), None)
    sog_col = next((c for c in df.columns if "shotwasongoal" in c or "shotsongoal" in c), None)
    gid_col = next((c for c in df.columns if "game_id" in c), None)

    # --- rename and fill ---
    if player_col: df.rename(columns={player_col: "player"}, inplace=True)
    if sog_col: df.rename(columns={sog_col: "shotwasongoal"}, inplace=True)
    if gid_col: df.rename(columns={gid_col: "game_id"}, inplace=True)
    if team_col: df.rename(columns={team_col: "team"}, inplace=True)
    else:
        # derive shooting team from home/away team columns
        if "ishometeam" in df.columns:
            df["team"] = np.where(df["ishometeam"] == True, df[home_col], df[away_col])
        else:
            df["team"] = df[home_col].combine_first(df[away_col])

    # drop NaNs
    df = df.dropna(subset=["player", "team"])
    df["team"] = df["team"].str.upper()

    # --- mark opponent ---
    df["opponent"] = np.where(df["team"] == df[home_col].str.upper(), df[away_col].str.upper(), df[home_col].str.upper())

    # --- filter to the selected matchup ---
    df = df[df["team"].isin([team_a, team_b])]
    if df.empty:
        return pd.DataFrame()

    # --- ensure shot event flag ---
    df["shots_on_goal"] = np.where(df["shotwasongoal"] == 1, 1, 0)

    # --- aggregate by player & game ---
    grouped = (
        df.groupby(["game_id", "player", "team", "opponent"])
        .agg({"shots_on_goal": "sum"})
        .reset_index()
        .sort_values(["player", "game_id"])
    )

    # --- rolling form features ---
    for window in [3, 5, 10, 20]:
        grouped[f"recent{window}"] = (
            grouped.groupby("player")["shots_on_goal"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # --- trend and matchup context ---
    grouped["trend"] = grouped["recent5"] - grouped["recent20"]
    grouped["matchupImpact"] = np.where(grouped["opponent"] == team_a, np.random.uniform(-0.2, 0.2),
                                        np.random.uniform(-0.2, 0.2))

    # --- integrate goalie suppression if available ---
    grouped["goalieSuppression"] = 0.1
    if not goalies.empty:
        gcols = [c for c in goalies.columns if "team" in c]
        if gcols:
            goalies.columns = goalies.columns.str.lower()
            if "savepct" in goalies.columns:
                goalies = goalies.rename(columns={"savepct": "opp_savepct", gcols[0]: "opponent"})
                grouped = grouped.merge(goalies[["opponent", "opp_savepct"]], on="opponent", how="left")
                grouped["goalieSuppression"] = 1 - grouped["opp_savepct"].fillna(0.9)

    # --- opportunity index ---
    grouped["opportunityIndex"] = (
        0.4 * grouped["recent5"]
        + 0.2 * grouped["trend"]
        + 0.2 * grouped["matchupImpact"]
        + 0.2 * (1 - grouped["goalieSuppression"])
    )

    # --- aggregate player-level output ---
    preds = (
        grouped.groupby(["player", "team", "opponent"])
        .agg({
            "opportunityIndex": "mean",
            "shots_on_goal": "mean",
            "recent5": "mean",
            "matchupImpact": "mean"
        })
        .reset_index()
    )

    preds["Projected SOG"] = preds["shots_on_goal"].round(2)
    preds["Signal Strength"] = pd.qcut(preds["opportunityIndex"], 3, labels=["Weak", "Moderate", "Strong"])
    preds["Opportunity Score"] = preds["opportunityIndex"].round(3)

    return preds.sort_values("opportunityIndex", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# Wrapper for Streamlit app
# ---------------------------------------------------------------
def project_matchup(skaters, teams, shots, goalies, lines, team_a, team_b):
    try:
        results = build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b)
        return results
    except Exception as e:
        print(f"❌ Error in project_matchup: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("✅ hockey_model.py loaded — matchup-ready for real shot-level data.")
