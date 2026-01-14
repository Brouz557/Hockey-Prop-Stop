# ---------------------------------------------------------------
# hockey_model.py (debug edition)
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import re

# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------
def normalize_team_name(name):
    if pd.isna(name):
        return np.nan
    s = str(name).strip().upper()
    s = re.sub(r"[^A-Z]", "", s)
    return s[:3]

# ---------------------------------------------------------------
# Parse and clean files
# ---------------------------------------------------------------
def parse_raw_files(file_dfs):
    skaters = file_dfs.get("skaters", pd.DataFrame())
    teams = file_dfs.get("teams", pd.DataFrame())
    shots = file_dfs.get("shots", pd.DataFrame())
    goalies = file_dfs.get("goalies", pd.DataFrame())
    lines = file_dfs.get("lines", pd.DataFrame())

    for df in [skaters, teams, shots, goalies, lines]:
        if not df.empty:
