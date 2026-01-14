# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Streamlit UI for Hockey Prop Stop
# ---------------------------------------------------------------

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import importlib.util
import traceback

# ---------------------------------------------------------------
# Safe dynamic import of hockey_model.py
# ---------------------------------------------------------------
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
spec = importlib.util.spec_from_file_location("hockey_model", module_path)
hockey_model = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(hockey_model)
    parse_raw_files = getattr(hockey_model, "parse_raw_files", None)
    project_matchup = getattr(hockey_model, "project_matchup", None)
except Exception as e:
    st.error("❌ Failed to load hockey_model.py.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------------------------------------------------------
# NHL team logos
# ---------------------------------------------------------------
NHL_LOGO_MAP = {
    "ANA": "https://assets.nhle.com/logos/nhl/svg/ANA.svg",
    "ARI": "https://assets.nhle.com/logos/nhl/svg/ARI.svg",
    "BOS": "https://assets.nhle.com/logos/nhl/svg/BOS.svg",
    "BUF": "https://assets.nhle.com/logos/nhl/svg/BUF.svg",
    "CGY": "https://assets.nhle.com/logos/nhl/svg/CGY.svg",
    "CAR": "https://assets.nhle.com/logos/nhl/svg/CAR.svg",
    "CHI": "https://assets.nhle.com/logos/nhl/svg/CHI.svg",
    "COL": "https://assets.nhle.com/logos/nhl/svg/COL.svg",
    "CBJ": "https://assets.nhle.com/logos/nhl/svg/CBJ.svg",
    "DAL": "https://assets.nhle.com/logos/nhl/svg/DAL.svg",
    "DET": "https://assets.nhle.com/logos/nhl/svg/DET.svg",
    "EDM": "https://assets.nhle.com/logos/nhl/svg/EDM.svg",
    "FLA": "https://assets.nhle.com/logos/nhl/svg/FLA.svg",
    "LAK": "https://assets.nhle.com/logos/nhl/svg/LAK.svg",
    "MIN": "https://assets.nhle.com/logos/nhl/svg/MIN.svg",
    "MTL": "https://assets.nhle.com/logos/nhl/svg/MTL.svg",
    "NSH": "https://assets.nhle.com/logos/nhl/svg/NSH.svg",
    "NJD": "https://assets.nhle.com/logos/nhl/svg/NJD.svg",
    "NYI": "https://assets.nhle.com/logos/nhl/svg/NYI.svg",
    "NYR": "https://assets.nhle.com/logos/nhl/svg/NYR.svg",
    "OTT": "https://assets.nhle.com/logos/nhl/svg/OTT.svg",
    "PHI": "https://assets.nhle.com/logos/nhl/svg/PHI.svg",
    "PIT": "https://assets.nhle.com/logos/nhl/svg/PIT.svg",
    "SEA": "https://assets.nhle.com/logos/nhl/svg/SEA.svg",
    "SJS": "https://assets.nhle.com/logos/nhl/svg/SJS.svg",
    "STL": "https://assets.nhle.com/logos/nhl/svg/STL.svg",
    "TBL": "https://assets.nhle.com/logos/nhl/svg/TBL.svg",
    "TOR": "https://assets.nhle.com/logos/nhl/svg/TOR.svg",
    "VAN": "https://assets.nhle.com/logos/nhl/svg/VAN.svg",
    "VGK": "https://assets.nhle.com/logos/nhl/svg/VGK.svg",
    "WSH": "https://assets.nhle.com/logos/nhl/svg/WSH.svg",
    "WPG": "https://assets.nhle.com/logos/nhl/svg/WPG.svg",
}

# ---------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

st.markdown(
    """
    <h1 style='text-align:center; color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Team-vs-Team matchup analytics with exponential regression weighting
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Upload Section
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Raw NHL Data Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSVs (skaters, teams, shots, goalies, lines)",
    type=["csv"],
    accept_multiple_files=True
)

data = pd.DataFrame()
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded. Parsing raw data...")

    raw_files = {}
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, nrows=200000)
            raw_files[f.name] = df
        except Exception as e:
            st.warning(f"⚠️ Could not read {f.name}: {e}")

    skaters_df, teams_df, shots_df, goalies_df, lines_df, all_teams = parse_raw_files(raw_files)

    st.write("✅ Parser completed. Detected teams:", all_teams)

    if len(all_teams) < 2:
        st.warning("⚠️ Could not detect enough teams in your data.")
    else:
        st.sidebar.markdown("### 🏒 Select Matchup")

        def team_option(team):
            logo = NHL_LOGO_MAP.get(team, "")
            return f"<img src='{logo}' width='20'> {team}" if logo else team

        team_a = st.sidebar.selectbox("Team A", all_teams)
        team_b = st.sidebar.selectbox("Team B", [t for t in all_teams if t != team_a])

        if st.sidebar.button("Run Model"):
            with st.spinner("Building matchup model..."):
                data = project_matchup(
                    skaters_df, teams_df, shots_df, goalies_df, lines_df, team_a, team_b
                )
            if not data.empty:
                st.success(f"✅ Model built: {team_a} vs {team_b}")
            else:
                st.warning("⚠️ Model returned no results.")
else:
    st.info("Upload your CSVs to begin.")

# ---------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------
if not data.empty:
    st.markdown("### 📊 Ranked Player Projections")
    ranked = data.rename(columns={
        "player": "Player",
        "team": "Team",
        "opponent": "Opponent",
        "predictedSOG": "Projected SOG",
        "probOver2.5": "Probability (Over 2.5)"
    })

    st.dataframe(
        ranked[[
            "Player", "Team", "Opponent",
            "Projected SOG", "Probability (Over 2.5)",
            "Matchup Rating", "Signal Strength"
        ]],
        use_container_width=True
    )

    # Visuals
    st.markdown("### 📈 Visuals")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=ranked.head(10),
            x="Projected SOG", y="Player",
            hue="Signal Strength",
            dodge=False
        )
        ax.set_title("Top Projected SOG")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=ranked, x="Team", y="Projected SOG", palette="Greens")
        ax2.set_title("Projected SOG by Team")
        st.pyplot(fig2)

    # Download
    st.markdown("### 💾 Export Results")
    out = BytesIO()
    ranked.to_excel(out, index=False)
    st.download_button(
        label="Download Excel",
        data=out.getvalue(),
        file_name="HockeyPropStop_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption("© Hockey Prop Stop — robust, exponentially weighted matchup model")
