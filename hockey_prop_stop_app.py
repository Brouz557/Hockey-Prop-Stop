# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop — Simplified Streamlit App
# ---------------------------------------------------------------

import importlib.util
import os
import streamlit as st
import pandas as pd
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
    import traceback
    st.sidebar.error(f"❌ Failed to load hockey_model.py:\n\n{e}")
    st.sidebar.code(traceback.format_exc())
    st.stop()

# ---------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

st.markdown(
    """
    <h1 style='text-align:center; color:#00B140;'>🏒 Hockey Prop Stop (Simplified)</h1>
    <p style='text-align:center; color:#BFC0C0;'>Basic SOG projections using rolling form + team/goalie context</p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Sidebar: Upload files
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Required Files")
shots_file = st.sidebar.file_uploader("shots.csv", type=["csv"])
teams_file = st.sidebar.file_uploader("NHL TEAMs.csv", type=["csv"])
goalies_file = st.sidebar.file_uploader("goalies.csv", type=["csv"])

if shots_file and teams_file and goalies_file:
    shots_df = pd.read_csv(shots_file)
    teams_df = pd.read_csv(teams_file)
    goalies_df = pd.read_csv(goalies_file)

    team_list = sorted(teams_df["team"].dropna().unique().tolist())
    colA, colB = st.columns(2)
    with colA:
        team_a = st.selectbox("Select Team A", options=team_list, index=0)
    with colB:
        team_b = st.selectbox("Select Team B", options=[t for t in team_list if t != team_a], index=1)

    if st.button("🚀 Run Simple Model"):
        with st.spinner(f"Running projection for {team_a} vs {team_b}..."):
            result = hockey_model.simple_project_matchup(shots_df, teams_df, goalies_df, team_a, team_b)

        if result.empty:
            st.error("⚠️ No valid projections generated.")
        else:
            st.success(f"✅ Generated {len(result)} player projections.")
            st.dataframe(result, use_container_width=True)

            # Plot
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=result.head(15), x="Projected_SOG", y="player", hue="SignalStrength", dodge=False)
            ax.set_title(f"Top 15 Projected SOG — {team_a} vs {team_b}")
            st.pyplot(fig)

            # Download
            out = BytesIO()
            result.to_excel(out, index=False)
            st.download_button(
                "💾 Download Excel",
                data=out.getvalue(),
                file_name=f"HockeyPropStop_Simple_{team_a}_vs_{team_b}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("📥 Upload shots.csv, NHL TEAMs.csv, and goalies.csv to begin.")

st.caption("© Hockey Prop Stop — Simplified Model v1.0")
