# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Betting-Focused Hybrid Regression Model
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io
from scipy.stats import poisson
import altair as alt

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with hybrid regression and player trend visualization
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .dataframe td {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: center !important;
        line-height: 1.3em !important;
    }
    .stDataFrame { overflow-x: auto; }
    </style>
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

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def load_file(file):
    if not file:
        return pd.DataFrame()
    try:
        return pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
    except Exception:
        return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None:
        return load_file(file_uploader)
    return safe_read(default_path)

# ---------------------------------------------------------------
# Cached load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    base_paths = [".","data","/mount/src/hockey-prop-stop/data"]
    def find_file(filename):
        for p in base_paths:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return full
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        skaters = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
        shots   = load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
        goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
        lines   = load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
        teams   = load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")
    return skaters, shots, goalies, lines, teams

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()
st.success("✅ Data loaded successfully.")

# ---------------------------------------------------------------
# Normalize columns
# ---------------------------------------------------------------
skaters_df.columns = skaters_df.columns.str.lower().str.strip()
shots_df.columns   = shots_df.columns.str.lower().str.strip()
if not goalies_df.empty:
    goalies_df.columns = goalies_df.columns.str.lower().str.strip()
if not lines_df.empty:
    lines_df.columns   = lines_df.columns.str.lower().str.strip()

team_col   = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
toi_col    = "icetime"
gp_col     = "games"
sog_col    = next((c for c in shots_df.columns if "sog" in c), None)
goal_col   = next((c for c in shots_df.columns if "goal" in c), None)
game_col   = next((c for c in shots_df.columns if "game" in c and "id" in c), None)
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)

# ---------------------------------------------------------------
# Team selection
# ---------------------------------------------------------------
all_teams = sorted(skaters_df[team_col].dropna().unique().tolist())
col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A", all_teams)
with col2:
    team_b = st.selectbox("Select Team B", [t for t in all_teams if t != team_a])
st.markdown("---")

if "results" not in st.session_state:
    st.session_state.results = None
run_model = st.button("🚀 Run Model")

# ---------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------
if run_model:
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

    # (model logic unchanged — same as before, omitted here for brevity)
    # ...
    # [Keep your full existing calculation loop and table setup]
    # ...

# ---------------------------------------------------------------
# Display Results + Visualization
# ---------------------------------------------------------------
if st.session_state.results is not None:
    st.markdown("### 📊 Player Projections + Regression Insight")
    html_table = st.session_state.results.to_html(index=False, escape=False)
    st.markdown(f"<div style='overflow-x:auto'>{html_table}</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # Player Trend Visualization (5-game smoothing)
    # ---------------------------------------------------------------
    st.markdown("### 📈 Player Regression Trend Viewer")
    player_list = st.session_state.results["Player"].unique().tolist()
    selected_player = st.selectbox("Select a player to view detailed trend:", player_list)

    df_p = shots_df[shots_df["player"].str.lower() == selected_player.lower()].copy()
    if df_p.empty:
        st.warning("No shot data available for this player.")
    else:
        trend_df = (
            df_p.groupby("gameid")[["sog", "goal"]]
            .sum()
            .reset_index()
            .sort_values("gameid")
        )
        trend_df["shoot_pct"] = np.where(
            trend_df["sog"] > 0, (trend_df["goal"] / trend_df["sog"]) * 100, 0
        )
        trend_df["game_num"] = np.arange(1, len(trend_df) + 1)

        # --- 5-game moving averages (for betting responsiveness)
        trend_df["sog_ma"] = trend_df["sog"].rolling(window=5, min_periods=1).mean()
        trend_df["shoot_pct_ma"] = trend_df["shoot_pct"].rolling(window=5, min_periods=1).mean()

        st.markdown(f"**Regression Summary for {selected_player}:**")
        player_regression = st.session_state.results.loc[
            st.session_state.results["Player"] == selected_player, "Regression Indicator"
        ].values[0]
        st.markdown(f"🧭 Regression Status: **{player_regression}**")

        # --- Build Altair Chart ---
        base = alt.Chart(trend_df).encode(
            x=alt.X("game_num:Q", title="Game Number")
        )

        shots_line = base.mark_line(color="#1f77b4").encode(
            y=alt.Y("sog_ma:Q", title="Shots on Goal (5-Game Avg)")
        )

        pct_line = base.mark_line(color="#d62728", strokeDash=[4, 3]).encode(
            y=alt.Y(
                "shoot_pct_ma:Q",
                title="Shooting % (5-Game Avg)",
                axis=alt.Axis(titleColor="#d62728")
            )
        )

        # Combine both on dual Y axes
        chart = (
            alt.layer(shots_line, pct_line)
            .resolve_scale(y="independent")
            .properties(
                width=700,
                height=400,
                title=f"{selected_player} — Shots vs Shooting% (5-Game Avg)"
            )
        )

        st.altair_chart(chart, use_container_width=True)
