# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop - Streamlit app (smart raw-data ingestion)
# ---------------------------------------------------------------

import importlib.util
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# --- Load hockey_model.py dynamically ---
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
spec = importlib.util.spec_from_file_location("hockey_model", module_path)
hockey_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hockey_model)

from hockey_model import parse_raw_files, project_matchup

# ---------------------------------------------------------------
# Sample data fallback
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Streamlit UI setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

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

# ---------------------------------------------------------------
# Sidebar Uploads
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Your Raw Data Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSVs (any mix of skaters, teams, shots, goalies, lines)",
    type=["csv"],
    accept_multiple_files=True
)

# ---------------------------------------------------------------
# Model execution
# ---------------------------------------------------------------
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded. Parsing raw data...")

    raw_files = {}
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            raw_files[f.name] = df
        except Exception as e:
            st.warning(f"⚠️ Could not read {f.name}: {e}")

    skaters_df, teams_df, shots_df, goalies_df, lines_df = hockey_model.parse_raw_files(raw_files)
    st.info("🔍 Files parsed. Building model...")

    data = project_matchup(skaters_df, teams_df, shots_df, goalies_df, lines_df)

    if data.empty:
        st.warning("⚠️ No valid projections generated — your files may be too raw or missing required data.")
    else:
        st.success("✅ Model built and projections generated.")
else:
    st.info("Showing sample data until files are uploaded.")
    data = load_sample()

# ---------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------
st.markdown("### 📊 Ranked Player Projections")

if data.empty:
    st.warning("⚠️ No projection data to display.")
else:
    sort_col = "Probability (Over)" if "Probability (Over)" in data.columns else data.columns[0]
    ranked = data.sort_values(sort_col, ascending=False).reset_index(drop=True)
    st.dataframe(ranked, use_container_width=True)

    # ---------------------------------------------------------------
    # Visuals
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Download
    # ---------------------------------------------------------------
    st.markdown("### 💾 Export Results")
    out = BytesIO()
    ranked.to_excel(out, index=False)
    st.download_button(
        label="Download Excel",
        data=out.getvalue(),
        file_name="HockeyPropStop_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption("© Hockey Prop Stop — auto-parsing model builder")
