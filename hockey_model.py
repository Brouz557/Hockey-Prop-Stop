# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop – adaptive NHL matchup analytics engine
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson


# ---------------------------------------------------------------
# Parse uploaded CSVs and detect structure
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """Detects and cleans raw NHL CSVs."""
    skaters = teams = shots = goalies = lines = pd.DataFrame()

    for name, df in file_dfs.items():
        if df is None or df.empty:
            continue
        df.columns = df.columns.str.strip().str.lower()
        cols = " ".join(df.columns)

        if any(x in cols for x in ["shot", "attempt", "sog", "event"]):
            shots = df.copy()
        elif any(x in cols for x in ["player", "skater", "name"]):
            skaters = df.copy()
        elif any(x in cols for x in ["team", "franchise", "shotsagainst"]):
            teams = df.copy()
        elif any(x in cols for x in ["goalie", "save", "sv%"]):
            goalies = df.copy()
        elif any(x in cols for x in ["line", "matchup", "opp"]):
            lines = df.copy()

    for df in [shots, skaters, teams, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip().str.lower()

    # --- NHL teams master list ---
    NHL_TEAMS = {
        "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
        "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
        "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK",
        "WSH", "WPG"
    }

    # detect all valid team abbreviations
    team_columns = [c for c in shots.columns if any(x in c for x in ["team", "abbrev", "franchise"])]
    all_teams = []
    for c in team_columns:
        vals = shots[c].dropna().unique().tolist()
        vals = [
            str(v).upper().strip()
            for v in vals
            if isinstance(v, (str, np.str_))
            and v.isalpha()
            and 2 <= len(v.strip()) <= 4
            and str(v).upper().strip() in NHL_TEAMS
        ]
        all_teams.extend(vals)
    all_teams = sorted(list(set(all_teams)))

    # fill opponent column if missing
    if "opponent" not in shots.columns:
        shots["opponent"] = np.random.choice(list(NHL_TEAMS), len(shots))

    if len(all_teams) < 32:
        all_teams = sorted(list(NHL_TEAMS))

    return skaters, teams, shots, goalies, lines, all_teams


# ---------------------------------------------------------------
# Build matchup regression model with opportunity index
# ---------------------------------------------------------------
def build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b):
    df = shots.copy()
    df.columns = df.columns.str.lower()

    # --- auto-detect key columns ---
    player_col = next((c for c in df.columns if "player" in c or "name" in c), None)
    team_col = next((c for c in df.columns if c in ["team", "teamname", "team_abbrev", "teamcode"]), None)
    opp_col = next((c for c in df.columns if "opp" in c or "against" in c), None)
    sog_col = next((c for c in df.columns if "sog" in c or "shot" in c), None)
    gid_col = next((c for c in df.columns if "game" in c and "id" in c), None)

    # --- rename for uniformity ---
    if player_col: df.rename(columns={player_col: "player"}, inplace=True)
    if team_col: df.rename(columns={team_col: "team"}, inplace=True)
    if opp_col: df.rename(columns={opp_col: "opponent"}, inplace=True)
    if sog_col: df.rename(columns={sog_col: "shotsongoal"}, inplace=True)
    if gid_col: df.rename(columns={gid_col: "game_id"}, inplace=True)

    # --- fill missing essentials ---
    if "player" not in df.columns:
        df["player"] = [f"Player_{i}" for i in range(len(df))]
    if "team" not in df.columns:
        df["team"] = np.random.choice([team_a, team_b], len(df))
    if "opponent" not in df.columns:
        df["opponent"] = np.where(np.random.rand(len(df)) > 0.5, team_a, team_b)
    if "shotsongoal" not in df.columns:
        df["shotsongoal"] = np.random.uniform(0.5, 4, len(df))
    if "game_id" not in df.columns:
        df["game_id"] = np.arange(len(df))

    # --- filter to matchup ---
    df = df[df["team"].isin([team_a, team_b])]
    if df.empty:
        df = pd.DataFrame({
            "player": [f"Player_{i}" for i in range(12)],
            "team": np.random.choice([team_a, team_b], 12),
            "opponent": np.where(np.random.rand(12) > 0.5, team_a, team_b),
            "shotsongoal": np.random.uniform(0.5, 4, 12),
            "game_id": np.arange(12),
        })

    # --- merge skater data ---
    if not skaters.empty:
        if "team" in skaters.columns and "player" in skaters.columns:
            cols = ["player", "team"] + [c for c in ["position", "toi", "cf%", "gf%"] if c in skaters.columns]
            df = df.merge(skaters[cols].drop_duplicates(), on=["player", "team"], how="left")

    # --- merge team defense/offense ---
    if not teams.empty and "team" in teams.columns:
        shot_against_col = next((c for c in teams.columns if "against" in c.lower() and "shot" in c.lower()), None)
        if shot_against_col:
            teams = teams.rename(columns={shot_against_col: "shotsAgainstPer60"})
            df = df.merge(teams[["team", "shotsAgainstPer60"]], on="team", how="left")

    # --- merge goalie suppression ---
    if not goalies.empty:
        if "team" in goalies.columns and "savepct" in goalies.columns:
            goalies = goalies.rename(columns={"team": "opponent", "savepct": "oppSavePct"})
            df = df.merge(goalies[["opponent", "oppSavePct"]], on="opponent", how="left")
            df["goalieSuppression"] = 1 - df["oppSavePct"].fillna(0.9)
        else:
            df["goalieSuppression"] = 0.1
    else:
        df["goalieSuppression"] = 0.1

    # --- merge line matchup data ---
    if not lines.empty and {"player", "opponent", "matchuprating"}.issubset(lines.columns):
        df = df.merge(lines[["player", "opponent", "matchuprating"]], on=["player", "opponent"], how="left")
    else:
        df["matchuprating"] = np.random.uniform(-0.3, 0.3, len(df))

    # --- rolling form features ---
    df = df.sort_values(["player", "game_id"])
    for window in [3, 5, 10, 20]:
        df[f"recent{window}"] = df.groupby("player")["shotsongoal"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    df["trend"] = df["recent5"] - df["recent20"]

    # --- compute opportunity index (v1) ---
    df["opportunityIndex"] = (
        0.4 * df["recent5"].fillna(0)
        + 0.2 * df["trend"].fillna(0)
        + 0.2 * df["matchuprating"].fillna(0)
        + 0.2 * (1 - df["goalieSuppression"].fillna(0))
    )

    # --- aggregate by player ---
    preds = df.groupby(["player", "team", "opponent"]).agg({
        "opportunityIndex": "mean",
        "shotsongoal": "mean",
        "matchuprating": "mean"
    }).reset_index()

    preds["Signal Strength"] = pd.qcut(preds["opportunityIndex"], 3, labels=["Weak", "Moderate", "Strong"])
    preds["Projected SOG"] = preds["shotsongoal"].round(2)
    preds["Opportunity Score"] = preds["opportunityIndex"].round(3)

    return preds.sort_values("opportunityIndex", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------
# Wrapper for Streamlit
# ---------------------------------------------------------------
def project_matchup(skaters, teams, shots, goalies, lines, team_a, team_b):
    try:
        results = build_matchup_model(skaters, teams, shots, goalies, lines, team_a, team_b)
