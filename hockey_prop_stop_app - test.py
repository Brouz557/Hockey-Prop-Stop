# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (Instant Filter + Logos + Injuries + xG)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components
# ---------------------------------------------------------------
# Team Abbreviation Normalization (ESPN -> Data)
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ":  "NJD","LA":  "LAK","SJ":  "SJS","TB":  "TBL","ARI": "ARI","ANA": "ANA",
    "BOS": "BOS","BUF": "BUF","CAR": "CAR","CBJ": "CBJ","CGY": "CGY","CHI": "CHI",
    "COL": "COL","DAL": "DAL","DET": "DET","EDM": "EDM","FLA": "FLA","MIN": "MIN",
    "MTL": "MTL","NSH": "NSH","NYI": "NYI","NYR": "NYR","OTT": "OTT","PHI": "PHI",
    "PIT": "PIT","SEA": "SEA","STL": "STL","TOR": "TOR","VAN": "VAN","VGK": "VGK",
    "WSH": "WSH","WPG": "WPG"
}

st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("Production Version")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups with inline logos, instant filters, injuries, and expected goals.</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])
injuries_file= st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def load_file(f):
    if not f: return pd.DataFrame()
    try:
        return pd.read_excel(f) if f.name.lower().endswith(".xlsx") else pd.read_csv(f)
    except Exception:
        return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None:
        return load_file(file_uploader)
    return safe_read(default_path)

@st.cache_data(show_spinner=False)
def load_all(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    base_paths=[".","data","/mount/src/hockey-prop-stop/data"]
    def find_file(name):
        for p in base_paths:
            fp=os.path.join(p,name)
            if os.path.exists(fp): return fp
        return None
    skaters=load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
    shots  =load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
    goalies=load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
    lines  =load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
    teams  =load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")

    injuries=pd.DataFrame()
    if injuries_file:
        injuries=load_file(injuries_file)
    if not injuries.empty:
        injuries.columns=injuries.columns.str.lower().str.strip()
        if "player" in injuries.columns:
            injuries["player"]=injuries["player"].astype(str).str.strip().str.lower()
    return skaters,shots,goalies,lines,teams,injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file)

if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing data. Upload required files.")
    st.stop()
st.success("✅ Data loaded successfully.")

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns=df.columns.str.lower().str.strip()

team_col=next((c for c in skaters_df.columns if "team" in c),None)
player_col="name" if "name" in skaters_df.columns else None
shots_df["player"]=shots_df["player"].astype(str).str.strip()
game_col=next((c for c in shots_df.columns if "game" in c and "id" in c),None)

# ---------------------------------------------------------------
# Run Button / Line Input
# ---------------------------------------------------------------
col_run,col_line=st.columns([3,1])
with col_run: run_model=st.button("🚀 Run Model (All Games)")
with col_line:
    line_test=st.number_input("Line to Test",0.0,10.0,3.5,0.5,key="line_test")
    st.session_state.line_test_val=line_test

# ---------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------
if run_model:
    st.success("Model ran — lights active")

if "results" in st.session_state:
    df=st.session_state.results.copy()

    # >>> ADDED: Line-aware Model Score + Signal
    test_line = st.session_state.line_test_val

    def model_score_line(row):
        score=0.0
        diff=row["Final Projection"]-test_line
        if diff>=1.0: score+=1.5
        elif diff>=0.4: score+=1.0
        prob=row.get("Prob ≥ Line (%)",0)
        if prob>=60: score+=1.0
        elif prob>=55: score+=0.5
        l10=[int(x) for x in str(row["L10 Shots"]).split(",") if x.strip().isdigit()]
        if sum(s>=test_line for s in l10)>=5: score+=1.0
        if row["Line Adj"]>=1.05: score+=1.0
        try:
            if float(row["Exp Goals (xG)"])>=0.30: score+=1.0
        except:
            pass
        return round(score,2)

    def model_light(score):
        if score>=4.0: return "<span style='color:#00FF00;font-weight:bold;'>🟢</span>"
        elif score>=2.5: return "<span style='color:#FFD700;font-weight:bold;'>🟡</span>"
        else: return "<span style='color:#FF4B4B;font-weight:bold;'>🔴</span>"

    df["Model Score"]=df.apply(model_score_line,axis=1)
    df["Signal"]=df["Model Score"].apply(model_light)

    cols=[
        "Signal","Model Score","Player","Team","Final Projection","Prob ≥ Line (%)",
        "Line Adj","Exp Goals (xG)","Form Indicator","L10 Shots"
    ]

    html_table=df[cols].to_html(index=False,escape=False)
    components.html(f"<div style='overflow-x:auto'>{html_table}</div>",height=700)
