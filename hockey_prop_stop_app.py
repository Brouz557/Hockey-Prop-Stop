# ---------------------------------------------------------------
# hockey_prop_stop_app.py — Stable Hockey Prop Stop (Dropdown Edition)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import hockey_model  # simple, safe import

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
# Sidebar: Upload Data Files
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
# Run only when all five files are uploaded
# ---------------------------------------------------------------
if all([uploaded_skaters, uploaded_teams, uploaded_shots, uploaded_goalies, uploaded_lines]):
    st.success("✅ 5 file(s) uploaded successfully.")

    skaters_df = raw_files["skaters"]
    teams_df = raw_files["teams"]
    shots_df = raw_files["shots"]
    goalies_df = raw_files["goalies"]
    lines_df = raw_files["lines"]

    # Team selectors
    all_teams = sorted(skaters_df["team"].dropna().unique().tolist())
    colA, colB = st.columns(2)
    with colA:
        team_a = st.selectbox("Select Team A", options=all_teams, index=0)
    with colB:
        team_b = st.selectbox("Select Team B", options=[t for t in all_teams if t != team_a], index=1)

    model_option = st.radio(
        "Choose Model Type:",
        ["Simple (L5 Only)", "Trend Weighted (L3/L5/L10/L20)"],
        horizontal=True
    )

    if st.button("🚀 Run Model"):
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

        if model_option.startswith("Simple"):
            result = hockey_model.simple_project_matchup(shots_df, teams_df, goalies_df, team_a, team_b)
        else:
            result = hockey_model.project_trend_matchup(shots_df, teams_df, goalies_df, team_a, team_b)

        if result is None or result.empty:
            st.error("⚠️ No valid projections generated.")
        else:
            st.success(f"✅ Model built successfully for {team_a} vs {team_b}.")

            # ---------------------------------------------------------------
            # 📊 Player Projection Table
            # ---------------------------------------------------------------
            st.markdown("### 📊 Ranked Player Projections")
            st.dataframe(result, use_container_width=True)

            # ---------------------------------------------------------------
            # 🧪 Player Backtest (Dropdown)
            # ---------------------------------------------------------------
            st.markdown("### 🧪 Backtest Player Accuracy")

            selected_player = st.selectbox(
                "Select a player to backtest:",
                result["player"].unique(),
                index=0
            )

            if st.button(f"📊 Run Backtest for {selected_player}"):
                st.info(f"Running backtest for **{selected_player}**...")

                bt = hockey_model.backtest_sog_accuracy(shots_df, player_name=selected_player)

                if bt is None or bt.empty:
                    st.warning("⚠️ No valid data found for that player.")
                else:
                    bt["error"] = bt["Projected_SOG"] - bt["Actual_SOG"]
                    mae = abs(bt["error"]).mean()
                    rmse = np.sqrt((bt["error"] ** 2).mean())
                    corr = bt[["Projected_SOG", "Actual_SOG"]].corr().iloc[0, 1]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Abs Error", f"{mae:.2f}")
                    col2.metric("RMSE", f"{rmse:.2f}")
                    col3.metric("Correlation", f"{corr:.2f}")

                    st.markdown(f"#### 📈 {selected_player} — Projected vs Actual SOG per Game")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(bt["game_id"], bt["Projected_SOG"], label="Projected", marker="o", color="#00B140")
                    ax.plot(bt["game_id"], bt["Actual_SOG"], label="Actual", marker="x", color="#FF4B4B")
                    ax.set_xlabel("Game ID")
                    ax.set_ylabel("Shots on Goal")
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.4)
                    st.pyplot(fig)

else:
    st.info("📥 Upload all five CSV files to begin model building.")

st.caption("© Hockey Prop Stop — adaptive NHL matchup model.")
