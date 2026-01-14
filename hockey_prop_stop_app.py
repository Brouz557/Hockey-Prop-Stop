# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop — Streamlit App with debug loader
# ---------------------------------------------------------------

import importlib.util
import os
import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Show exactly which model file is loaded
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
print("DEBUG: Loading model from path:", module_path)

spec = importlib.util.spec_from_file_location("hockey_model", module_path)
hockey_model = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(hockey_model)
    st.sidebar.success("✅ hockey_model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load hockey_model.py.\n\n{e}")
    st.stop()

parse_raw_files = hockey_model.parse_raw_files
project_matchup = hockey_model.project_matchup

st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.title("🏒 Hockey Prop Stop — Debug Mode")

st.sidebar.header("📂 Upload CSV Files")
uploaded_skaters = st.sidebar.file_uploader("NHL Skaters.csv", type=["csv"])
uploaded_teams = st.sidebar.file_uploader("NHL TEAMs.csv", type=["csv"])
uploaded_shots = st.sidebar.file_uploader("shots.csv", type=["csv"])
uploaded_goalies = st.sidebar.file_uploader("goalies.csv", type=["csv"])
uploaded_lines = st.sidebar.file_uploader("lines.csv", type=["csv"])

raw_files = {
    "skaters": pd.read_csv(uploaded_skaters) if uploaded_skaters else pd.DataFrame(),
    "teams": pd.read_csv(uploaded_teams) if uploaded_teams else pd.DataFrame(),
    "shots": pd.read_csv(uploaded_shots) if uploaded_shots else pd.DataFrame(),
    "goalies": pd.read_csv(uploaded_goalies) if uploaded_goalies else pd.DataFrame(),
    "lines": pd.read_csv(uploaded_lines) if uploaded_lines else pd.DataFrame(),
}

if all([uploaded_skaters, uploaded_teams, uploaded_shots, uploaded_goalies, uploaded_lines]):
    st.success("✅ Files uploaded.")
    skaters_df, teams_df, shots_df, goalies_df, lines_df, all_teams = parse_raw_files(raw_files)
    st.success(f"Teams detected: {', '.join(all_teams)}")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A", options=all_teams, index=0)
    with col2:
        team_b = st.selectbox("Select Team B", options=[t for t in all_teams if t != team_a], index=1)

    if st.button("🚀 Run Model"):
        st.info(f"Building model for {team_a} vs {team_b} ...")
        data = project_matchup(skaters_df, teams_df, shots_df, goalies_df, lines_df, team_a, team_b)

        if data is None or data.empty:
            st.error("⚠️ No valid projections generated. Check terminal debug logs.")
        else:
            st.success(f"✅ Generated {len(data)} projections.")
            st.dataframe(data, use_container_width=True)
else:
    st.info("📥 Upload all five CSV files to begin.")

st.caption("© Hockey Prop Stop — debug build")
