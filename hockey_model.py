# ---------------------------------------------------------------
# hockey_model.py
# Hockey Prop Stop – robust, matchup-aware analytics engine
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson

# ---------------------------------------------------------------
# Safe, fast parser with HOME/AWAY detection and auto-sampling
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """Safely detects, cleans, and samples raw NHL CSVs for modeling."""
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
        except Exception as e:
            print(f"⚠️ Error parsing {name}: {e}")
            continue

    # -----------------------------------------------------------
    # Guarantee a usable shots table
    # -----------------------------------------------------------
    if shots.empty:
        print("⚠️ No shots file detected — creating placeholder.")
        shots = pd.DataFrame({
            "player": [f"Player_{i}" for i in range(20)],
            "team": np.random.choice(["CAR", "DET", "BOS", "NYR"], 20),
            "opponent": np.random.choice(["CAR", "DET", "BOS", "NYR"], 20),
            "shotsOnGoal": np.random.uniform(1, 4, 20),
            "game_id": np.arange(20)
        })
    else:
        shots.columns = shots.columns.str.strip().str.lower()
        player_col = next((c for c in shots.columns if "player" in c or "name" in c), None)
        if player_col and "player" not in shots.columns:
            shots.rename(columns={player_col: "player"}, inplace=True)

        team_col = next((c for c in shots.columns if "team" in c), None)
        if team_col and "team" not in shots.columns:
            shots.rename(columns={team_col: "team"}, inplace=True)

        opp_col = next((c for c in shots.columns if "opp" in c or "against" in c), None)
        if opp_col and "opponent" not in shots.columns:
            shots.rename(columns={opp_col: "opponent"}, inplace=True)

        sog_col = next((c for c in shots.columns if "sog" in c or "_
