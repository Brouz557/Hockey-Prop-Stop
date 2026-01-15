# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop — Backtest-Enabled App
# ---------------------------------------------------------------

import os
import importlib.util
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO

# ---------------------------------------------------------------
# App Config
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

st.markdown(
    """
    <h1 style='text-align:center; color:#BFC0C0;'>
        <span style='color:#00B140;'>🏒 Hockey Prop Stop</span>
    </h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Team-vs-Team calibrated matchup analytics with backtesting
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
# Sidebar Uploads with Session Persistence
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Daily Data Files")

def load_csv(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.DataFrame()

# Initialize state
for key in ["skaters", "teams", "shots", "goalies", "lines"]:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame()

# File uploaders
uploaded_skaters = st.sidebar.file_uploader("NHL Skaters.csv", type=["csv"])
uploaded_teams   = st.sidebar.file_uploader("NHL TEAMs.csv", type=["csv"])
uploaded_shots   = st.sidebar.file_uploader("shots.csv", type=["csv"])
uploaded_goalies = st.sidebar.file_uploader("goalies.csv", type=["csv"])
uploaded_lines   = st.sidebar.file_uploader("lines.csv", type=["csv"])

# Load into cache
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

# Get data
skaters_df = st.session_state["skaters"]
teams_df   = st.session_state["teams"]
shots_df   = st.session_state["shots"]
goalies_df = st.session_state["goalies"]
lines_df   = st.session_state["lines"]

# ---------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------
if not all([skaters_df.empty, teams_df.empty, shots_df.empty, goalies_df.empty, lines_df.empty]):
    st.success("✅ Data ready! Select matchup below.")

    # Team list
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

    # Run projection
    if st.button("🚀 Run Model"):
        st.info(f"Building calibrated model for **{team_a} vs {team_b}**...")
        result = hockey_model.simple_project_matchup(shots_df, teams_df, goalies_df, team_a, team_b)

        if result is None or result.empty:
            st.error("⚠️ No valid projections generated.")
        else:
            st.success(f"✅ Generated {len(result)} player projections.")
            st.dataframe(result, use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(result["Projected_SOG"], kde=True, color="#00B140", ax=ax)
            ax.set_xlabel("Projected Shots on Goal")
            st.pyplot(fig)

            out = BytesIO()
            result.to_excel(out, index=False)
            st.download_button(
                label="💾 Download Excel",
                data=out.getvalue(),
                file_name=f"HockeyPropStop_{team_a}_vs_{team_b}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # ---------------------------------------------------------------
    # Backtest Accuracy
    # ---------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🧪 Backtest Accuracy Evaluation")

    if st.button("📊 Run Backtest Accuracy"):
        st.info("Running historical backtest...")

        # Build player rolling 5-game form
        pf = hockey_model.build_basic_form(shots_df)
        if pf.empty:
            st.error("⚠️ Could not compute player form.")
            st.stop()

        # Compute actual next-game SOG for same players
        actuals = (
            shots_df.groupby(["shooterName", "teamCode", "game_id"])["shotWasOnGoal"]
            .sum()
            .reset_index()
        )

        # Take only players that appear in player_form
        actuals = actuals.rename(columns={"shooterName": "player", "teamCode": "team"})
        actuals["team"] = actuals["team"].astype(str).str.upper()

        # Merge latest form to get most recent avg_5
        merged = pf.merge(actuals, on=["player", "team"], how="inner")
        merged = merged.rename(columns={"shotWasOnGoal": "Actual_SOG", "avg_5": "Projected_SOG"})

        if merged.empty:
            st.error("⚠️ No overlapping player data found for backtest.")
            st.stop()

        mae = mean_absolute_error(merged["Actual_SOG"], merged["Projected_SOG"])
        rmse = np.sqrt(mean_squared_error(merged["Actual_SOG"], merged["Projected_SOG"]))
        corr = merged[["Actual_SOG", "Projected_SOG"]].corr().iloc[0, 1]

        st.success(f"**MAE:** {mae:.2f} **RMSE:** {rmse:.2f} **Correlation:** {corr:.2f}")

        # Scatter plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=merged, x="Projected_SOG", y="Actual_SOG", alpha=0.6)
        ax.set_title("Projected vs Actual Shots on Goal")
        ax.axline((0, 0), slope=1, color="red", linestyle="--")
        st.pyplot(fig)

else:
    st.info("📥 Upload all five CSV files to begin model building.")

st.caption("© Hockey Prop Stop — calibrated NHL matchup model with accuracy testing.")
