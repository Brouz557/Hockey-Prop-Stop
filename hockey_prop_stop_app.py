# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop - Streamlit app
# Dark green / silver theme with sample data
# ---------------------------------------------------------------

import importlib.util
import sys, os, types

# --- load hockey_model.py directly no matter what path Streamlit uses ---
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
spec = importlib.util.spec_from_file_location("hockey_model", module_path)
hockey_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hockey_model)

# expose functions from the loaded module
build_model = hockey_model.build_model
project_matchup = hockey_model.project_matchup

# --- standard libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


# ---- Sample data function (for when no files uploaded) ----
def load_sample():
    np.random.seed(42)
    players = ["Aho", "Larkin", "Necas", "Burns", "Raymond", "Walman"]
    pos = ["F", "F", "F", "D", "F", "D"]
    team = ["CAR", "DET", "CAR", "CAR", "DET", "DET"]
    sog = np.random.uniform(1.0, 4.0, 6).round(2)
    prob = np.random.uniform(0.45, 0.8, 6).round(2)
    strength = ["Strong" if p > 0.7 else "Moderate" if p > 0.55 else "Weak" for p in prob]
    match = ["Favorable", "Neutral", "Favorable", "Favorable", "Unfavorable", "Unfavorable"]
    odds = (1 / prob - 1) * 100
    df = pd.DataFrame({
        "Player": players, "Team": team, "Pos": pos,
        "Projected SOG": sog, "Probability (Over)": prob,
        "Signal Strength": strength, "Matchup Favorability": match,
        "Lowest Playable Odds": odds.round(0)
    })
    return df


# ---- Streamlit setup ----
st.set_page_config(
    page_title="Hockey Prop Stop",
    layout="wide",
    page_icon="🏒"
)

# ---- Header ----
st.markdown(
    """
    <h1 style='text-align:center; color:#BFC0C0;'>
        <span style='color:#00B140;'>🏒 Hockey Prop Stop</span>
    </h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Data-driven shots-on-goal analytics dashboard
    </p>
    """,
    unsafe_allow_html=True
)

# ---- Sidebar for uploads ----
st.sidebar.header("📂 Upload Daily Data Files")
uploaded_skaters = st.sidebar.file_uploader("NHL Skaters.csv", type=["csv"])
uploaded_teams   = st.sidebar.file_uploader("NHL TEAMs.csv", type=["csv"])
uploaded_shots   = st.sidebar.file_uploader("shots.csv", type=["csv"])
uploaded_goalies = st.sidebar.file_uploader("goalies.csv", type=["csv"])
uploaded_lines   = st.sidebar.file_uploader("lines.csv", type=["csv"])

# --------------------------------------------------
# Build model and generate projections when files uploaded
# --------------------------------------------------
if all([uploaded_skaters, uploaded_teams, uploaded_shots, uploaded_goalies, uploaded_lines]):
    st.success("✅ Data uploaded successfully. Building model...")

    skaters_df = pd.read_csv(uploaded_skaters)
    teams_df   = pd.read_csv(uploaded_teams)
    shots_df   = pd.read_csv(uploaded_shots)
    goalies_df = pd.read_csv(uploaded_goalies)
    lines_df   = pd.read_csv(uploaded_lines)

    model, df = build_model(skaters_df, teams_df, shots_df, goalies_df, lines_df)
    data = project_matchup(model, df, "CAR", "DET")

    st.success("✅ Model built and projections generated.")
else:
    st.info("Showing sample data until files are uploaded.")
    data = load_sample()


# ---- Display ranked table ----
st.markdown("### 📊 Ranked Player Projections")

if "Probability (Over)" in data.columns:
    sort_col = "Probability (Over)"
else:
    sort_col = data.columns[-2] if len(data.columns) > 2 else data.columns[0]

ranked = data.sort_values(sort_col, ascending=False).reset_index(drop=True)
st.dataframe(ranked, use_container_width=True)

# ---- Visuals ----
st.markdown("### 📈 Visuals")
col1, col2 = st.columns(2)
with col1:
    if "Projected SOG" in ranked.columns and "Probability (Over)" in ranked.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(
            x="Projected SOG", y="Probability (Over)",
            data=ranked, hue="Signal Strength", s=100
        )
        ax.set_title("Projected SOG vs Probability")
        st.pyplot(fig)
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.corrcoef(np.random.rand(6, 6)), cmap="Greens", cbar=False)
    ax2.set_title("Sample Signal Heatmap")
    st.pyplot(fig2)

# ---- Download Excel ----
st.markdown("### 💾 Export Results")
out = BytesIO()
ranked.to_excel(out, index=False)
st.download_button(
    label="Download Excel",
    data=out.getvalue(),
    file_name="HockeyPropStop_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("© Hockey Prop Stop — data refreshed daily.")
