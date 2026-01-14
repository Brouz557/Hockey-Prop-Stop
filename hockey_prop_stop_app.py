# ---------------------------------------------------------------
# hockey_prop_stop_app.py (patched version)
# Hockey Prop Stop — Streamlit App (resilient data validator)
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
# Dynamic loader
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
normalize_team_name = hockey_model.normalize_team_name

# ---------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------
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
# Sidebar upload zone
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
# File readiness validator
# ---------------------------------------------------------------
def validate_files(file_dict):
    """Checks each uploaded file for required columns and reports issues."""
    checks = {
        "skaters": ["team", "name"],
        "teams": ["team", "xGoalsFor"],
        "shots": ["shooterName", "teamCode", "shotWasOnGoal", "game_id"],
        "goalies": ["team", "goals", "ongoal"],
        "lines": ["team", "xGoalsFor", "xGoalsAgainst"],
    }

    errors = []
    for k, df in file_dict.items():
        if df.empty:
            errors.append(f"⚠️ {k}.csv not uploaded.")
            continue
        missing = [c for c in checks[k] if c not in [col.lower() for col in df.columns.str.lower()]]
        if missing:
            errors.append(f"⚠️ {k}.csv missing expected column(s): {', '.join(missing)}")

    return errors


# ---------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------
if all([uploaded_skaters, uploaded_teams, uploaded_shots, uploaded_goalies, uploaded_lines]):
    st.success("✅ 5 file(s) uploaded. Parsing raw data...")

    validation_issues = validate_files(raw_files)
    if validation_issues:
        with st.expander("🔎 File validation issues detected:"):
            for msg in validation_issues:
                st.warning(msg)
        st.write("Attempting to continue with partial data...")
    else:
        st.sidebar.success("✅ All required columns detected.")

    skaters_df, teams_df, shots_df, goalies_df, lines_df, all_teams = parse_raw_files(raw_files)
    st.success(f"✅ Files parsed successfully — {len(all_teams)} teams detected.")

    # UI Selectors
    colA, colB = st.columns(2)
    with colA:
        team_a = st.selectbox("Select Team A", options=all_teams, index=0)
    with colB:
        team_b = st.selectbox("Select Team B", options=[t for t in all_teams if t != team_a], index=1)

    # ---------------------------------------------------------------
    # Run Model Button
    # ---------------------------------------------------------------
    if st.button("🚀 Run Model"):
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

        data = project_matchup(skaters_df, teams_df, shots_df, goalies_df, lines_df, team_a, team_b)

        if data is None or data.empty:
            st.error("⚠️ No valid projections generated — check data consistency below.")
            st.write("**Debug Info:**")
            st.write("Teams found in shots:", sorted(shots_df["teamCode"].unique()) if not shots_df.empty else "None")
            st.write("Teams found in skaters:", sorted(skaters_df["team"].unique()) if not skaters_df.empty else "None")
        else:
            st.success(f"✅ Model built successfully for {team_a} vs {team_b}.")
            data.columns = [c.strip().title() for c in data.columns]
            ranked = data.sort_values("Projected_Sog", ascending=False).reset_index(drop=True)

            # ---------------------------------------------------------------
            # Display Projections
            # ---------------------------------------------------------------
            st.markdown("### 📊 Ranked Player Projections")
            st.dataframe(
                ranked[["Player", "Team", "Opponent", "Projected_Sog", "Signalstrength"]],
                use_container_width=True,
            )

            # ---------------------------------------------------------------
            # Visualization Zone
            # ---------------------------------------------------------------
            st.markdown("### 📈 Visuals")
            col1, col2 = st.columns(2)

            with col1:
                if "Projected_Sog" in ranked.columns and "Z_Score" in ranked.columns:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.scatterplot(
                        x="Projected_Sog", y="Z_Score", data=ranked,
                        hue="Signalstrength", s=100, palette="cool"
                    )
                    ax.set_title("Projected SOG vs Player Form (Z-Score)")
                    st.pyplot(fig)

            with col2:
                if "Matchupadj" in ranked.columns:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    sns.histplot(ranked["Matchupadj"], kde=True, color="#00B140")
                    ax2.set_title("Distribution of Matchup Adjustments")
                    st.pyplot(fig2)

            # ---------------------------------------------------------------
            # Export Results
            # ---------------------------------------------------------------
            st.markdown("### 💾 Export Results")
            out = BytesIO()
            ranked.to_excel(out, index=False)
            st.download_button(
                label="Download Excel",
                data=out.getvalue(),
                file_name=f"HockeyPropStop_{team_a}_vs_{team_b}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

else:
    st.info("📥 Upload all five CSV files to begin model building.")

st.caption("© Hockey Prop Stop — validated NHL matchup model.")
