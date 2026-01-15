# ---------------------------------------------------------------
# hockey_prop_stop_app.py — Streamlit Front End (Final)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hockey_model

st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

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
# Sidebar: Upload Daily Data Files (CSV or Excel)
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Daily Data Files")

uploaded_skaters = st.sidebar.file_uploader("SKATERS (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_teams   = st.sidebar.file_uploader("TEAMS (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_goalies = st.sidebar.file_uploader("GOALTENDERS (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_lines   = st.sidebar.file_uploader("LINE DATA (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_shots   = st.sidebar.file_uploader("SHOT DATA (CSV or Excel)", type=["csv", "xlsx", "xls"])

def load_file(file):
    """Load CSV or Excel file safely."""
    if not file:
        return pd.DataFrame()
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error reading {file.name}: {e}")
        return pd.DataFrame()

raw_files = {
    "skaters": load_file(uploaded_skaters),
    "teams": load_file(uploaded_teams),
    "goalies": load_file(uploaded_goalies),
    "lines": load_file(uploaded_lines),
    "shots": load_file(uploaded_shots),
}

# ---------------------------------------------------------------
# Main App Logic
# ---------------------------------------------------------------
if all([uploaded_skaters, uploaded_teams, uploaded_goalies, uploaded_lines, uploaded_shots]):
    st.success("✅ All 5 files uploaded successfully.")

    skaters_df = raw_files["skaters"]
    teams_df = raw_files["teams"]
    shots_df = raw_files["shots"]
    goalies_df = raw_files["goalies"]
    lines_df = raw_files["lines"]

    # Normalize column names for robustness
    skaters_df.columns = skaters_df.columns.str.strip().str.lower()
    shots_df.columns = shots_df.columns.str.strip().str.lower()

    # Detect key columns
    player_col_skaters = next((c for c in skaters_df.columns if "player" in c or "name" in c), None)
    team_col_skaters = next((c for c in skaters_df.columns if "team" in c), None)
    player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)

    # --- Align team info from SKATERS into SHOTS ---
    if player_col_skaters and player_col_shots and team_col_skaters:
        skater_team_map = skaters_df[[player_col_skaters, team_col_skaters]].dropna()
        skater_team_map.columns = ["player", "team"]

        # Coerce join columns to string to avoid dtype mismatch
        shots_df = shots_df.rename(columns={player_col_shots: "player"})
        shots_df["player"] = shots_df["player"].astype(str).str.strip()
        skater_team_map["player"] = skater_team_map["player"].astype(str).str.strip()

        shots_df = shots_df.merge(skater_team_map, on="player", how="left")
        st.success("✅ Linked player → team from SKATERS file.")
    else:
        st.warning("⚠️ Could not align team data from SKATERS — using raw SHOT DATA.")

    # --- Build team dropdowns using SKATERS data ---
    team_col = next((c for c in skaters_df.columns if "team" in c), None)
    if not team_col:
        st.error("No 'team' column detected in SKATERS file.")
    else:
        all_teams = sorted(skaters_df[team_col].dropna().unique().tolist())
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
                st.markdown("### 📊 Ranked Player Projections")
                st.dataframe(result, use_container_width=True)

                st.markdown("### 🧪 Backtest Player Accuracy")
                selected_player = st.selectbox("Select player for backtest:", result["player"].unique(), index=0)

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

                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.plot(bt["game_id"], bt["Projected_SOG"], label="Projected", marker="o", color="#00B140")
                        ax.plot(bt["game_id"], bt["Actual_SOG"], label="Actual", marker="x", color="#FF4B4B")
                        ax.set_xlabel("Game ID")
                        ax.set_ylabel("Shots on Goal")
                        ax.legend()
                        ax.grid(True, linestyle="--", alpha=0.4)
                        st.pyplot(fig)

else:
    st.info("📥 Upload all five files (.csv or .xlsx) to begin model building.")

st.caption("© Hockey Prop Stop — adaptive NHL matchup model.")
