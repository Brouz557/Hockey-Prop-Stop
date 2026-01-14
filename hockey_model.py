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
    sog_col = next((c for c in df.c_
