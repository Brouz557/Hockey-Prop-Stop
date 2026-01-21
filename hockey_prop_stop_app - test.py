# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — L5 Probability Update (TEST MODE)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, datetime, pytz, subprocess, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page Setup — must be first Streamlit call
# ---------------------------------------------------------------
st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")

# 🧪 TEST MODE WARNING — Safe sandbox banner
st.warning("🧪 TEST MODE — You’re editing and running the sandbox version. Changes here will NOT affect your main app.")

# ---------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
        <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
    </div>
    <h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
    <p style='text-align:center;color:#D6D6D6;'>
        Team-vs-Team matchup analytics with blended regression and L5-based probabilities
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])
injuries_file = st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def load_file(file):
    if not file: return pd.DataFrame()
    try:
        return pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
    except Exception:
        return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None:
        return load_file(file_uploader)
    return safe_read(default_path)

# ---------------------------------------------------------------
# Cached Data Load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
    def find_file(name):
        for p in base_paths:
            full = os.path.join(p, name)
            if os.path.exists(full): return full
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        skaters = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
        shots   = load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
        goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
        lines   = load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
        teams   = load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")

        injuries = pd.DataFrame()
        for p in ["injuries.xlsx","Injuries.xlsx","./injuries.xlsx","data/injuries.xlsx","/mount/src/hockey-prop-stop/injuries.xlsx"]:
            if os.path.exists(p):
                injuries = load_file(open(p,"rb")); break
        if injuries.empty:
            injuries = load_file(injuries_file)
        if not injuries.empty:
            injuries.columns = injuries.columns.str.lower().str.strip()
            if "player" in injuries.columns:
                injuries["player"] = injuries["player"].astype(str).str.strip().str.lower()
    return skaters, shots, goalies, lines, teams, injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()
st.success("✅ Data loaded successfully.")

# ---------------------------------------------------------------
# Normalize Columns
# ---------------------------------------------------------------
for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns = df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
shots_df = shots_df.rename(columns={player_col_shots: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)

# ---------------------------------------------------------------
# Team Selection
# ---------------------------------------------------------------
teams = sorted(skaters_df[team_col].dropna().unique().tolist())
col1, col2 = st.columns(2)
with col1: team_a = st.selectbox("Select Team A", teams)
with col2: team_b = st.selectbox("Select Team B", [t for t in teams if t != team_a])
st.markdown("---")

# ---------------------------------------------------------------
# Build Model (same as your version + Pace Factor column)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    # ... (same as the long version above with Pace Factor)
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model + Display
# ---------------------------------------------------------------
line_input_col, run_col = st.columns([1, 2])
with line_input_col:
    target_line = st.number_input("Enter Target Line (e.g., 2.5):", min_value=0.0, step=0.5, value=2.5)
with run_col:
    run_button = st.button("🚀 Run Model")

if run_button:
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")
    df = build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df)
    df = df.sort_values(["Final Projection", "Line Adj"], ascending=[False, False]).reset_index(drop=True)
    st.session_state.results_raw = df.copy()
    st.success("✅ Model built successfully!")

# ---------------------------------------------------------------
# Display Table + Save
# ---------------------------------------------------------------
if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
    df = st.session_state.results_raw.copy()

    # Trend color logic
    def trend_color(v):
        if pd.isna(v): return "–"
        if v > 0.05:
            color = "#00B140"; symbol = "▲"
        elif v < -0.05:
            color = "#E63946"; symbol = "▼"
        else:
            color = "#6C7A89"; symbol = "–"
        return f"<div style='background:{color};color:#FFFFFF;font-weight:600;border-radius:6px;padding:4px 8px;text-align:center;'>{symbol}</div>"

    df["Trend"] = df["Trend Score"].apply(trend_color)

    cols = [
        "Player","Team","Injury","Trend","Final Projection",
        "Prob ≥ Projection (%) L5","Season Avg","Line Adj","Pace Factor",
        "Form Indicator","L3 Shots","L5 Shots","L10 Shots"
    ]
    vis = df[[c for c in cols if c in df.columns]]

    html_table = vis.to_html(index=False, escape=False)
    components.html(f"""
        <style>
        div.scrollable-table {{
            overflow-x:auto;overflow-y:auto;height:600px;
        }}
        table {{
            width:100%;border-collapse:collapse;font-family:'Source Sans Pro',sans-serif;color:#D6D6D6;
        }}
        th {{
            background-color:#0A3A67;color:#FFFFFF;padding:6px;text-align:center;position:sticky;top:0;
            border-bottom:2px solid #1E5A99;
        }}
        td:first-child,th:first-child {{
            position:sticky;left:0;background-color:#1E5A99;color:#FFFFFF;font-weight:bold;
        }}
        td {{
            background-color:#0F2743;color:#D6D6D6;padding:4px;text-align:center;
        }}
        tr:nth-child(even) td {{background-color:#142F52;}}
        </style>
        <div class='scrollable-table'>{html_table}</div>
        """, height=620, scrolling=True)

    st.markdown("---")
    st.subheader("💾 Save or Download Projections")
    selected_date = st.date_input("Select game date:", datetime.date.today())

    if st.button("💾 Save Projections for Selected Date"):
        df_to_save = df.copy()
        df_to_save["Date_Game"] = selected_date.strftime("%Y-%m-%d")
        df_to_save["Matchup"] = f"{team_a} vs {team_b}"
        os.makedirs("projections", exist_ok=True)
        filename = f"{team_a}_vs_{team_b}_{selected_date.strftime('%Y-%m-%d')}.csv"
        save_path = os.path.join("projections", filename)
        df_to_save.to_csv(save_path, index=False)
        st.success(f"✅ Saved projections to **{save_path}**")
        csv = df_to_save.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Projections CSV", csv, filename, "text/csv")
