# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Streamlit UI for Hockey Prop Stop
# ---------------------------------------------------------------

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import importlib.util
import traceback

# ---------------------------------------------------------------
# Safe dynamic import of hockey_model.py
# ---------------------------------------------------------------
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
spec = importlib.util.spec_from_file_location("hockey_model", module_path)
hockey_model = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(hockey_model)
    parse_raw_files = getattr(hockey_model, "parse_raw_files", None)
    project_matchup = getattr(hockey_model, "project_matchup", None)

    # Diagnostic printout to verify functions are loaded
    st.write("✅ hockey_model loaded:", 
             [x for x in dir(hockey_model) if not x.startswith("__")])

    if parse_raw_files is None or project_matchup is None:
        st.error("❌ hockey_model.py loaded but expected functions not found.")
        st.stop()

except Exception as e:
    st.error("❌ Failed to load hockey_model.py.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

st.markdown(
    """
    <h1 style='text-align:center; color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Team-vs-Team matchup analytics with exponential regression weighting
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Raw NHL Data Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSVs (skaters, teams, shots, goalies, lines)",
    type=["csv"],
    accept_multiple_files=True
)

data = pd.DataFrame()
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded. Parsing raw data...")

    raw_files = {}
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, nrows=200000)
            raw_files[f.name] = df
        except Exception as e:
            st.warning(f"⚠️ Could not read {f.name}: {e}")

    skaters_df, teams_df, shots_df, goalies_df, lines_df = parse_raw_files(raw_files)
    st.write("✅ Parser completed. Detected teams:",
             sorted(shots_df["team"].dropna().unique().tolist()))

    all_teams = sorted(shots_df["team"].dropna().unique())
    if len(all_teams) < 2:
        st.warning("⚠️ Could not detect enough teams in your data.")
    else:
        st.sidebar.markdown("### 🏒 Select Matchup")
        team_a = st.sidebar.selectbox("Team A", all_teams)
        team_b = st.sidebar.selectbox("Team B", [t for t in all_teams if t != team_a])

        if st.sidebar.button("Run Model"):
            with st.spinner("Building matchup model..."):
                data = project_matchup(
                    skaters_df, teams_df, shots_df, goalies_df, lines_df, team_a, team_b
                )
            if not data.empty:
                st.success("✅ Model built and projections generated.")
            else:
                st.warning("⚠️ Model returned no results.")
else:
    st.info("Upload your CSVs to begin.")

# ---------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------
if not data.empty:
    st.markdown("### 📊 Ranked Player Projections")
    ranked = data.rename(columns={
        "player": "Player",
        "team": "Team",
        "opponent": "Opponent",
        "predictedSOG": "Projected SOG",
        "probOver2.5": "Probabi
