# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Goal Projections + Calibration
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with improved goal projections
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
    base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
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
    lines_df.columns = lines_df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
sog_col = next((c for c in shots_df.columns if "sog" in c), None)
goal_col = next((c for c in shots_df.columns if "goal" in c), None)
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)
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

    # [MODEL CODE OMITTED HERE — use your same model logic from before]
    # (same as in your working version — unchanged except below)

    # After df = pd.DataFrame(results)
    # --- Calibration and scoring tier logic (same as before) ---

    st.session_state.results = df  # make sure this exists for calibration below

# ---------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------
if st.session_state.results is not None:
    import altair as alt

    st.markdown("### 📊 Player Projections (Adjusted)")
    html_table = st.session_state.results.to_html(index=False, escape=False)
    st.markdown(f"<div style='overflow-x:auto'>{html_table}</div>", unsafe_allow_html=True)

    st.markdown("### 🔁 Interactive Table (Sortable)")
    display_cols = [c for c in ["Player","Final Projection","Projected Goals",
                                "Shooting %","Matchup Rating","Scoring Tier"]
                    if c in st.session_state.results.columns]
    st.dataframe(st.session_state.results[display_cols], use_container_width=True, hide_index=True)

    # --- Visual Separation Chart ---
    st.markdown("### 🎯 Shot vs Goal Visualization")
    chart = alt.Chart(st.session_state.results).mark_circle(size=80).encode(
        x=alt.X("Final Projection", title="Shots Projection"),
        y=alt.Y("Projected Goals", title="Goal Projection"),
        color=alt.Color("Shooting %", scale=alt.Scale(scheme="viridis")),
        tooltip=[c for c in ["Player","Team","Projected Goals","Shooting %","Scoring Tier","Final Projection"]
                 if c in st.session_state.results.columns]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    # ---------------------------------------------------------------
    # 📈 Model Calibration Chart: Actual vs Projected
    # ---------------------------------------------------------------
    st.markdown("### 🧮 Model Calibration (Projected vs Actual Goals)")

    if "goal" in shots_df.columns and "sog" in shots_df.columns:
        calib_df = (
            shots_df.groupby("player", as_index=False)
            .agg({"goal": "sum", "sog": "sum"})
        )
        calib_df["actual_goal_rate"] = np.where(calib_df["sog"] > 0, calib_df["goal"] / calib_df["sog"], 0)

        merged = st.session_state.results[["Player", "Projected Goals"]].rename(columns={"Player": "player"})
        merged = pd.merge(merged, calib_df, on="player", how="left")

        # Dynamic binning
        bin_count = 6
        bins = np.linspace(0, max(0.6, merged["Projected Goals"].max()), bin_count + 1)
        merged["goal_bin"] = pd.cut(merged["Projected Goals"], bins=bins, include_lowest=True)

        calib_summary = (
            merged.groupby("goal_bin", as_index=False)
            .agg({
                "Projected Goals": "mean",
                "actual_goal_rate": "mean",
                "player": "count"
            })
            .rename(columns={"player": "Player Count"})
        )

        calib_melt = calib_summary.melt(
            id_vars=["goal_bin", "Player Count"],
            value_vars=["Projected Goals", "actual_goal_rate"],
            var_name="Type",
            value_name="Value"
        )

        chart_calib = (
            alt.Chart(calib_melt)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X("goal_bin:N", title="Projected Goals Range"),
                y=alt.Y("Value:Q", title="Average Goals / Goal Rate"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Projected Goals", "actual_goal_rate"],
                    range=["#00B140", "#BFC0C0"]
                )),
                tooltip=["goal_bin", "Player Count", "Type", "Value"]
            )
            .properties(height=400)
        )
        st.altair_chart(chart_calib, use_container_width=True)
        st.caption("Green = Model projection | Gray = Actual scoring rate from data")
    else:
        st.info("Calibration chart unavailable (need 'goal' and 'sog' data in shots_df).")
