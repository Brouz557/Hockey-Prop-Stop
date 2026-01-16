# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Data Timestamp (Excel Tag + Central Time)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import contextlib
import datetime
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with goalie, line, and trend visualization
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx", "csv"])
shots_file = st.sidebar.file_uploader("SHOT DATA", type=["xlsx", "csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx", "csv"])
lines_file = st.sidebar.file_uploader("LINE DATA", type=["xlsx", "csv"])
teams_file = st.sidebar.file_uploader("TEAMS", type=["xlsx", "csv"])

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def load_file(file):
    if not file:
        return pd.DataFrame()
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception:
        return pd.DataFrame()

def safe_read(path):
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception:
        return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None:
        return load_file(file_uploader)
    return safe_read(default_path)

# ---------------------------------------------------------------
# Cache + Load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
    def find_file(filename):
        for p in base_paths:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return full
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        skaters = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
        shots = load_data(shots_file, find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
        goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
        lines = load_data(lines_file, find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
        teams = load_data(teams_file, find_file("TEAMS.xlsx") or "TEAMS.xlsx")

    return skaters, shots, goalies, lines, teams

# ---------------------------------------------------------------
# Data Timestamp Logic
# ---------------------------------------------------------------
def get_excel_lastupdated_tag(file_path):
    """Check Skaters.xlsx for 'LastUpdated:' tag."""
    try:
        df = pd.read_excel(file_path, header=None, nrows=10)
        for _, row in df.iterrows():
            for cell in row:
                if isinstance(cell, str) and cell.strip().lower().startswith("lastupdated:"):
                    return cell.split(":", 1)[1].strip()
    except Exception:
        return None
    return None

def get_file_last_modified(path):
    """Get file modification time in Central Time."""
    try:
        ts = os.path.getmtime(path)
        dt = datetime.datetime.fromtimestamp(ts, ZoneInfo("America/Chicago"))
        return dt.strftime("%B %d, %Y — %I:%M %p CT")
    except Exception:
        return None

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file
)

# Find path for skaters (for timestamp reading)
possible_paths = [
    "Skaters.xlsx",
    "./data/Skaters.xlsx",
    "/mount/src/hockey-prop-stop/data/Skaters.xlsx"
]
skaters_path = next((p for p in possible_paths if os.path.exists(p)), None)

# Determine update string
update_info = None
if skaters_path:
    excel_tag = get_excel_lastupdated_tag(skaters_path)
    if excel_tag:
        update_info = f"Last updated: {excel_tag}"
    else:
        file_time = get_file_last_modified(skaters_path)
        if file_time:
            update_info = f"Last updated: {file_time}"

if not update_info:
    update_info = "Last updated: (timestamp unavailable)"

st.success(f"✅ Data loaded successfully ({update_info}).")

# ---------------------------------------------------------------
# ✅ Continue to the rest of your model code unchanged below
# ---------------------------------------------------------------
st.write("... continue with your projection model here ...")
