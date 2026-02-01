# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Team Abbreviation Normalization
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ": "NJD", "LA": "LAK", "SJ": "SJS", "TB": "TBL", "ARI": "ARI",
    "ANA": "ANA", "BOS": "BOS", "BUF": "BUF", "CAR": "CAR", "CBJ": "CBJ",
    "CGY": "CGY", "CHI": "CHI", "COL": "COL", "DAL": "DAL", "DET": "DET",
    "EDM": "EDM", "FLA": "FLA", "MIN": "MIN", "MTL": "MTL", "NSH": "NSH",
    "NYI": "NYI", "NYR": "NYR", "OTT": "OTT", "PHI": "PHI", "PIT": "PIT",
    "SEA": "SEA", "STL": "STL", "TOR": "TOR", "VAN": "VAN", "VGK": "VGK",
    "WSH": "WSH", "WPG": "WPG"
}

st.set_page_config(page_title="Puck Shotz Hockey Analytics", layout="wide", page_icon="🏒")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Data Files")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])
injuries_file= st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------
def load_file(f):
    if not f: return pd.DataFrame()
    try:
        return pd.read_excel(f) if f.name.lower().endswith(".xlsx") else pd.read_csv(f)
    except Exception: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_all(sk_f, sh_f, gl_f, ln_f, tm_f, ij_f):
    sk = load_file(sk_f)
    sh = load_file(sh_f)
    gl = load_file(gl_f)
    ln = load_file(ln_f)
    tm = load_file(tm_f)
    ij = load_file(ij_f)
    
    for df in [sk, sh, gl, ln, tm]:
        if not df.empty: df.columns = df.columns.str.lower().str.strip()
    return sk, sh, gl, ln, tm, ij

skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file)

# ---------------------------------------------------------------
# Build Model Logic
# ---------------------------------------------------------------
def build_model(team_a, team_b):
    # (Placeholder for your complex model logic)
    # This function uses the uploaded data to calculate projections, 
    # consistency scores, and matchup heat.
    # ...
    results = pd.DataFrame() # This would be your calculated output
    return results

# ---------------------------------------------------------------
# Run Section (Corrected from Screenshot)
# ---------------------------------------------------------------
if skaters_df.empty or shots_df.empty:
    st.info("Please upload data files to begin.")
else:
    team_col = next((c for c in skaters_df.columns if "team" in c), "team")
    
    if st.button("🚀 Run Model"):
        all_games = []
        for t in skaters_df[team_col].unique():
            # In your script, this typically takes two teams for a matchup
            df = build_model(t, t) 
            if not df.empty:
                all_games.append(df)
        
        if all_games:
            # Combine all results and sort by score as shown in your screenshot
            final = pd.concat(all_games).sort_values("FC 3SOG Score", ascending=False)
            st.dataframe(final, use_container_width=True)
            st.session_state.results = final
        else:
            st.warning("No valid data generated. Check your input file columns.")

# ---------------------------------------------------------------
# Display logic for filtered results / CSV downloads
# ---------------------------------------------------------------
if "results" in st.session_state:
    st.download_button("💾 Download Results", st.session_state.results.to_csv(index=False), "puck_shotz_results.csv")
