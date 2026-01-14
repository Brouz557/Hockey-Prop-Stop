# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop — Streamlit app (Debug + Full UI)
# ---------------------------------------------------------------

import importlib.util
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import traceback

# ---------------------------------------------------------------
# Load hockey_model.py dynamically
# ---------------------------------------------------------------
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
st.code(f"Loading model from:\n{module_path}", language="bash")
print("DEBUG: Loading model from path:", module_path)

spec = importlib.util.spec_from_file_location("hockey_model", module_path)
hockey_model = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(hockey_model)
    st.sidebar.success("✅ hockey_model loaded successfully.")
except Exception as e:
    st.sidebar.error("❌ Failed to load hockey_model.py.")
    st.exception(e)
    st.stop()

parse_raw_files = hockey_model.parse_raw_files
project_matchup = hockey_model.project_matchup

# ---------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

st.markdown(
    """
    <h1 style='text-align:center; color:#BFC0C0;'>
        <span style='color:#00B140;'>🏒 Hockey Prop Stop (Debug)</span>
    </h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Team-vs-Team matchup analytics with detailed debugging
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Sidebar uploads
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
# Parse and select teams
# ---------------------------------------------------------------
if all([uploaded_skaters, uploaded_teams, uploaded_shots, uploaded_goalies, uploaded_lines]):
    st.success("✅ 5 file(s) uploaded. Parsing raw data...")

    try:
        skaters_df, teams_df, shots_df, goalies_df, lines_df, all_teams = parse_raw_files(raw_files)
        st.success("✅ Files parsed successfully.")
        st.write("**Teams detected:**", all_teams)
        st.write("**Shots columns (first 25):**", list(shots_df.columns)[:25])
    except Exception as e:
        st.error("❌ Error during parse_raw_files.")
        st.exception(e)
        st.stop()

    colA, colB = st.columns(2)
    with colA:
        team_a = st.selectbox("Select Team A", options=all_teams, index=0)
    with colB:
        team_b = st.selectbox("Select Team B", options=[t for t in all_teams if t != team_a], index=1)

    # -----------------------------------------------------------
    # Run the model
    # -----------------------------------------------------------
    if st.button("🚀 Run Model"):
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

        try:
            data = project_matchup(skaters_df, teams_df, shots_df, goalies_df, lines_df, team_a, team_b)
        except Exception as e:
            st.error("❌ Error while running project_matchup.")
            st.exception(e)
            st.stop()

        # -------------------------------------------------------
        # Handle missing projections
        # -------------------------------------------------------
        if data is None or data.empty:
            st.error("⚠️ No valid projections generated. Check console for debug logs.")
            # Extra in-app debugging
            try:
                st.write("**Unique teams in shots:**", 
                         sorted(shots_df.get("teamCode", pd.Series(dtype=object)).dropna().unique().tolist()))
                st.write("**Unique teams in skaters:**", 
                         sorted(skaters_df.get("team", pd.Series(dtype=object)).dropna().unique().tolist()))
                if hasattr(hockey_model, "build_player_form"):
                    pf = hockey_model.build_player_form(shots_df)
                    st.write("**Player form preview:**")
                    st.dataframe(pf.head(10))
            except Exception as e:
                st.warning("Unable to render extra debug info.")
                st.exception(e)
        else:
            st.success(f"✅ Model built successfully for {team_a} vs {team_b}.")
            data.columns = [c.strip().title() for c in data.columns]
            st.markdown("### 📊 Ranked Player Projections")
            st.dataframe(data, use_container_width=True)

            # ---------------------------------------------------
            # Visualization
            # ---------------------------------------------------
            st.markdown("### 📈 Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                if "Projected_Sog" in data.columns and "Z_Score" in data.columns:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.scatterplot(
                        x="Projected_Sog", y="Z_Score",
                        data=data, hue="Signalstrength", s=100
                    )
                    ax.set_title("Projected SOG vs Z-Score")
                    st.pyplot(fig)
            with col2:
                if "Signalstrength" in data.columns:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    sns.countplot(data=data, x="Signalstrength", palette="viridis")
                    ax2.set_title("Signal Strength Distribution")
                    st.pyplot(fig2)

            # ---------------------------------------------------
            # Download results
            # ---------------------------------------------------
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
    st.info("📥 Upload all five CSV files to begin model building.")

st.caption("© Hockey Prop Stop — debug-ready build")
