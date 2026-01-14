# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop – stable NHL matchup analytics engine
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson


def parse_raw_files(file_dfs):
    """Parse uploaded CSVs, harmonize column names, detect teams."""
    skaters = pd.DataFrame()
    teams = pd.DataFrame()
    shots = pd.DataFrame()
    goalies = pd.DataFrame()
    lines = pd.DataFrame()

    for name, df in file_dfs.items():
        try:
            if df is None or df.empty:
                continue
            if len(df) > 100000:
                df = df.tail(100000)
            cols = [c.lower() for c in df.columns]
            if any(x in "".join(cols) for x in ["shot", "attempt", "sog", "event"]):
                shots = df.copy()
            elif any(x in "".join(cols) for x in ["player", "skater", "name"]):
                skaters = df.copy()
            elif any(x in "".join(cols) for x in ["team", "franchise", "shotsagainst"]):
                teams = df.copy()
            elif any(x in "".join(cols) for x in ["goalie", "save", "sv%"]):
                goalies = df.copy()
            elif any(x in "".join(cols) for x in ["line", "matchup", "opp"]):
                lines = df.copy()
        except Exception:
            continue

    for df in [shots, skaters, teams, goalies, lines]:
        if not df.empty:
            df.columns = df.columns.str.strip().str.lower()

    # --- NHL team list ---
    NHL_TEAMS = {
        "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
        "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
        "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK",
        "WSH", "WPG"
    }

    # --- find potential team columns and valid abbreviations ---
    team_columns = [c for c in shots.columns if any(x in c for x in ["team", "abbrev", "franchise"])]
    all_teams_found = []
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
        all_teams_found.extend(vals)
    all_teams_found = sorted(list(set(all_teams_found)))

    # --- fix HOME/AWAY mapping ---
    if "team" in shots.columns and set(shots["team"].astype(str).str.upper().unique()) <= {"HOME", "AWAY"}:
        home_col = next((c for c in shots.columns if "home" in c and "team" in c), None)
        away_col = next((c for c in shots.columns if "away" in c and "team" in c), None)
        if not home_col:
            home_col = next((c for c in shots.colu_
