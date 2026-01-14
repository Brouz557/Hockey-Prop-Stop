# ---------------------------------------------------------------
# hockey_model.py
# Core analytics engine for Hockey Prop Stop
# Matchup-aware model with exponentially weighted regression
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson
import re

# ---------------------------------------------------------------
# Smart parser for raw NHL data
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    """Detects skaters, teams, shots, goalies, lines automatically."""
    skaters = pd.DataFrame()
    teams = pd.DataFrame()
    shots = pd.DataFrame()
    goalies = pd.DataFrame()
    lines = pd.DataFrame()

    for name, df in file_dfs.items():
        if df is None or df.empty:
            continue
        cols = [c.lower() for c in df.columns]

        # --- Shots / events ---
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

    # Normalize shots columns
    if not shots.empty:
        shots.columns = shots.columns.str.strip().str.lower()
        # player
        pla
