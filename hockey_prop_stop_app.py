# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop — Final Streamlit App (Persistent Uploads)
# ---------------------------------------------------------------

import os
import importlib.util
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ---------------------------------------------------------------
# App Config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Hockey Prop Stop",
    layout="wide",
    page_icon="🏒",
)

st.markdown(
    """
    <h1 style='text-align:center; color:#BFC0C0;'>
        <span style='color:#00B140;'>🏒 Hockey Prop Stop</span>
    </h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Team-vs-Team calibrated matchup analytics
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Load hockey_model dynamically
# ---------------------------------------------------------------
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
spec = importlib.util.spec_from_file_location("hockey_model", module_path)
hockey_model = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(hockey_model)
    st.sidebar.success("✅ hockey_model.py loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load hockey_model.py.\n\n{e}")
    st.stop()

# ---------------------------------------------------------------
# Sidebar Uploads with Persistent Session State
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Daily Data Files")

def load_csv(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.DataFrame()

# Initialize session state for persistence
for key in ["skaters", "teams", "shots", "goalies", "lines"]:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame()

# Uploaders
uploaded_skaters = st.sidebar.file_uploader("NHL Skaters.csv", type=["csv"])
uploaded_teams   = st.sidebar.file_uploader("NHL TEAMs.csv", type=["csv"])
uploaded_shots   = st.sidebar.file_uploader("shots.csv", type=["csv"])
uploaded_goalies = st.sidebar.file_uploader("goalies.csv", type=["csv"])
uploaded_lines   = st.sidebar.file_uploader("lines.csv", type=["csv"])

# Handle uploads (persist in session)
if uploaded_skaters: st.session_state["skaters"] = load_csv(uploaded_skaters)
if uploaded_teams:   st.session_state["teams"]   = load_csv(uploaded_teams)
if uploaded_shots:   st.session_state["shots"]   = load_csv(uploaded_shots)
if uploaded_goalies: st.session_state["goalies"] = load_csv(uploaded_goalies)
if uploaded_lines:   st.session_state["lines"]   = load_csv(uploaded_lines)

# Reset button
if st.sidebar.button("🔄 Reset Uploads"):
    for key in ["skaters", "teams", "shots", "goalies", "lines"]:
        st.session_state[key] = pd.DataFrame()
    st.experimental_rerun()

# Pull cached data
skaters_df = st.session_state["skaters"]
teams_df   = st.session_state["teams"]
shots_df   = st.session_state["shots"]
goalies_df = st.session_state["goalies"]
lines_df   = st.session_state["lines"]

# ---------------------------------------------------------------
# Main Section
# ---------------------------------------------------------------
if not all([skaters_df.empty, teams_df.empty, shots_df.empty, goalies_df.empty, lines_df.empty]):
    st.success("✅ Data ready! Select matchup below.")

    # Derive available teams
    if "team" in skaters_df.columns:
        all_teams = sorted(skaters_df["team"].dropna().unique().tolist())
    else:
        all_teams = []

    if len(all_teams) < 2:
        st.error("⚠️ Not enough team data found in your uploads.")
        st.stop()

    colA, colB = st.columns(2)
    with colA:
        team_a = st.selectbox("Select Team A", options=all_teams)
    with colB:
        team_b = st.selectbox("Select Team B", options=[t for t in all_teams if t != team_a])

    if st.button("🚀 Run Simple Model"):
        st.info(f"Building calibrated model for **{team_a} vs {team_b}**...")

        result = hockey_model.simple_project_matchup(shots_df, teams_df, goalies_df, team_a, team_b)

        if result is None or result.empty:
            st.error("⚠️ No valid projections generated. Check your matchup data.")
        else:
            st.success(f"✅ Generated {len(result)} player projections.")

            # Display Table
            st.dataframe(result, use_container_width=True)

            # Visualization
            st.markdown("### 📈 Distribution of Projected SOG")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(result["Projected_SOG"], kde=True, color="#00B140", ax=ax)
            ax.set_xlabel("Projected Shots on Goal")
            st.pyplot(fig)

            # Download Option
            st.markdown("### 💾 Export Results")
            out = BytesIO()
            result.to_excel(out, index=False)
            st.download_button(
                label="Download Excel",
                data=out.getvalue(),
                file_name=f"HockeyPropStop_{team_a}_vs_{team_b}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("📥 Upload all five CSV files to begin model building.")

st.caption("© Hockey Prop Stop — calibrated NHL matchup model.")
