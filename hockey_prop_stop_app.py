# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop — Streamlit App (Final Learning Version)
# ---------------------------------------------------------------

import importlib.util
import os
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ---------------------------------------------------------------
# Load hockey_model.py dynamically
# ---------------------------------------------------------------
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
spec = importlib.util.spec_from_file_location("hockey_model", module_path)
hockey_model = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(hockey_model)
    st.sidebar.success("✅ hockey_model.py loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load hockey_model.py:\n\n{e}")
    st.sidebar.code(traceback.format_exc())
    st.stop()

# Shortcuts
parse_raw_files = hockey_model.parse_raw_files
project_matchup = hockey_model.project_matchup

# ---------------------------------------------------------------
# Streamlit Setup
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Hockey Prop Stop",
    layout="wide",
    page_icon="🏒"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#BFC0C0;'>
        <span style='color:#00B140;'>🏒 Hockey Prop Stop</span>
    </h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Team-vs-Team matchup analytics with adaptive regression learning
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Sidebar: Upload Data Files
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Required CSV Files")
uploaded_skaters = st.sidebar.file_uploader("NHL Skaters.csv", type=["csv"])
uploaded_teams = st.sidebar.file_uploader("NHL TEAMs.csv", type=["csv"])
uploaded_shots = st.sidebar.file_uploader("shots.csv", type=["csv"])
uploaded_goalies = st.sidebar.file_uploader("goalies.csv", type=["csv"])
uploaded_lines = st.sidebar.file_uploader("lines.csv", type=["csv"])

if all([uploaded_skaters, uploaded_teams, uploaded_shots, uploaded_goalies, uploaded_lines]):
    st.success("✅ 5 file(s) uploaded. Parsing data...")

    raw_files = {
        "skaters": pd.read_csv(uploaded_skaters),
        "teams": pd.read_csv(uploaded_teams),
        "shots": pd.read_csv(uploaded_shots),
        "goalies": pd.read_csv(uploaded_goalies),
        "lines": pd.read_csv(uploaded_lines),
    }

    skaters_df, teams_df, shots_df, goalies_df, lines_df, all_teams = parse_raw_files(raw_files)
    st.success("✅ Files parsed successfully.")

    # Team selectors
    colA, colB = st.columns(2)
    with colA:
        team_a = st.selectbox("Select Team A", options=all_teams, index=0)
    with colB:
        team_b = st.selectbox("Select Team B", options=[t for t in all_teams if t != team_a], index=1)

    if st.button("🚀 Run Model"):
        st.info(f"Building learning regression model for **{team_a} vs {team_b}**...")

        data = project_matchup(skaters_df, teams_df, shots_df, goalies_df, lines_df, team_a, team_b)

        if data is None or data.empty:
            st.error("⚠️ No valid projections generated. Check if team codes match across files.")
        else:
            st.success(f"✅ Model built successfully for {team_a} vs {team_b}.")
            st.markdown("### 📊 Player Projections")

            # Normalize column titles
            data.columns = [c.strip().title().replace("_", " ") for c in data.columns]
            if "Projected Sog" not in data.columns:
                st.error("⚠️ No 'Projected SOG' column found in results.")
                st.dataframe(data.head(), use_container_width=True)
            else:
                # Sort and show main columns
                display_cols = [
                    "Player", "Team", "Opponent",
                    "Projected Sog", "Avg 5", "Z Score",
                    "Signalstrength", "Matchupadj"
                ]
                cols_to_show = [c for c in display_cols if c in data.columns]
                st.dataframe(data[cols_to_show].sort_values("Projected Sog", ascending=False),
                             use_container_width=True)

                # ---------------------------------------------------------------
                # 📈 Visualizations
                # ---------------------------------------------------------------
                st.markdown("### 📈 Visual Insights")
                col1, col2 = st.columns(2)

                with col1:
                    if "Projected Sog" in data.columns and "Z Score" in data.columns:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.scatterplot(x="Z Score", y="Projected Sog", data=data, hue="Signalstrength", s=100)
                        ax.set_title("Z-Score vs Projected SOG")
                        st.pyplot(fig)

                with col2:
                    if "Projected Sog" in data.columns:
                        fig2, ax2 = plt.subplots(figsize=(5, 4))
                        sns.histplot(data["Projected Sog"], bins=20, kde=True, color="#00B140")
                        ax2.set_title("Projected SOG Distribution")
                        st.pyplot(fig2)

                # ---------------------------------------------------------------
                # 💾 Export
                # ---------------------------------------------------------------
                st.markdown("### 💾 Export Results")
                out = BytesIO()
                data.to_excel(out, index=False)
                st.download_button(
                    label="Download Excel",
                    data=out.getvalue(),
                    file_name=f"HockeyPropStop_{team_a}_vs_{team_b}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

else:
    st.info("📥 Upload all five CSV files to build projections.")

st.caption("© Hockey Prop Stop — adaptive NHL matchup model (v2).")
