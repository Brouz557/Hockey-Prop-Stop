# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop — Streamlit App (v2, stable)
# ---------------------------------------------------------------

import importlib.util
import os
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
    st.sidebar.success("✅ hockey_model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load hockey_model.py.\n\n{e}")
    st.stop()

# Expose functions
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
        Team-vs-Team matchup analytics with adaptive regression weighting
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Sidebar: Upload data files
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Daily Data Files")
uploaded_skaters = st.sidebar.file_uploader("NHL Skaters.csv", type=["csv"])
uploaded_teams   = st.sidebar.file_uploader("NHL TEAMs.csv", type=["csv"])
uploaded_shots   = st.sidebar.file_uploader("shots.csv", type=["csv"])
uploaded_goalies = st.sidebar.file_uploader("goalies.csv", type=["csv"])
uploaded_lines   = st.sidebar.file_uploader("lines.csv", type=["csv"])

raw_files = {
    "skaters": pd.read_csv(uploaded_skaters) if uploaded_skaters else pd.DataFrame(),
    "teams": pd.read_csv(uploaded_teams) if uploaded_teams else pd.DataFrame(),
    "shots": pd.read_csv(uploaded_shots) if uploaded_shots else pd.DataFrame(),
    "goalies": pd.read_csv(uploaded_goalies) if uploaded_goalies else pd.DataFrame(),
    "lines": pd.read_csv(uploaded_lines) if uploaded_lines else pd.DataFrame(),
}

# ---------------------------------------------------------------
# Parse files and populate team selectors
# ---------------------------------------------------------------
if all([uploaded_skaters, uploaded_teams, uploaded_shots, uploaded_goalies, uploaded_lines]):
    st.success("✅ 5 file(s) uploaded. Parsing raw data...")

    skaters_df, teams_df, shots_df, goalies_df, lines_df, all_teams = parse_raw_files(raw_files)
    st.success("✅ Files parsed successfully.")

    colA, colB = st.columns(2)
    with colA:
        team_a = st.selectbox("Select Team A", options=all_teams, index=0)
    with colB:
        team_b = st.selectbox("Select Team B", options=[t for t in all_teams if t != team_a], index=1)

    if st.button("🚀 Run Model"):
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

        data = project_matchup(skaters_df, teams_df, shots_df, goalies_df, lines_df, team_a, team_b)

        if data is None or data.empty:
            st.error("⚠️ No valid projections generated. Your files might be missing matchup data.")
        else:
            st.success(f"✅ Model built successfully for {team_a} vs {team_b}.")

            # Normalize column names
            data.columns = [c.strip().title() for c in data.columns]

            # Rename Opportunity Score → Probability (Over 2.5) for consistency
            if "Opportunity Score" in data.columns and "Probability (Over 2.5)" not in data.columns:
                data = data.rename(columns={"Opportunity Score": "Probability (Over 2.5)"})

            ranked = data.sort_values("Probability (Over 2.5)", ascending=False).reset_index(drop=True)

            # ---- Display Ranked Table ----
            st.markdown("### 📊 Ranked Player Projections")

            expected_cols = [
                "Player", "Team", "Opponent",
                "Projected Sog", "Probability (Over 2.5)",
                "Matchup Rating", "Signal Strength"
            ]
            available_cols = [c for c in expected_cols if c in ranked.columns]

            if available_cols:
                st.dataframe(ranked[available_cols], use_container_width=True)
            else:
                st.dataframe(ranked, use_container_width=True)

            # ---- Visualizations ----
            st.markdown("### 📈 Visuals")
            col1, col2 = st.columns(2)
            with col1:
                if "Projected Sog" in ranked.columns and "Probability (Over 2.5)" in ranked.columns:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.scatterplot(
                        x="Projected Sog",
                        y="Probability (Over 2.5)",
                        data=ranked,
                        hue="Signal Strength",
                        s=100
                    )
                    ax.set_title("Projected SOG vs Probability (Over 2.5)")
                    st.pyplot(fig)
            with col2:
                if "Matchup Rating" in ranked.columns:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    sns.histplot(ranked["Matchup Rating"], kde=True, color="#00B140")
                    ax2.set_title("Distribution of Matchup Ratings")
                    st.pyplot(fig2)

            # ---- Download Excel ----
            st.markdown("### 💾 Export Results")
            out = BytesIO()
            ranked.to_excel(out, index=False)
            st.download_button(
                label="Download Excel",
                data=out.getvalue(),
                file_name=f"HockeyPropStop_{team_a}_vs_{team_b}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("📥 Upload all five CSV files to begin model building.")

st.caption("© Hockey Prop Stop — adaptive NHL matchup model.")
