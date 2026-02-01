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
    "NJ":"NJD","LA":"LAK","SJ":"SJS","TB":"TBL",
    "ARI":"ARI","ANA":"ANA","BOS":"BOS","BUF":"BUF","CAR":"CAR",
    "CBJ":"CBJ","CGY":"CGY","CHI":"CHI","COL":"COL","DAL":"DAL",
    "DET":"DET","EDM":"EDM","FLA":"FLA","MIN":"MIN","MTL":"MTL",
    "NSH":"NSH","NYI":"NYI","NYR":"NYR","OTT":"OTT","PHI":"PHI",
    "PIT":"PIT","SEA":"SEA","STL":"STL","TOR":"TOR","VAN":"VAN",
    "VGK":"VGK","WSH":"WSH","WPG":"WPG"
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
    return load_file(file_uploader) if file_uploader else safe_read(default_path)

@st.cache_data(show_spinner=False)
def load_all(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    skaters = load_data(skaters_file, "Skaters.xlsx")
    shots   = load_data(shots_file, "SHOT DATA.xlsx")
    goalies = load_data(goalies_file, "GOALTENDERS.xlsx")
    lines   = load_data(lines_file, "LINE DATA.xlsx")
    teams   = load_data(teams_file, "TEAMS.xlsx")
    injuries= load_data(injuries_file, "Injuries.xlsx")
    if not injuries.empty:
        injuries.columns=injuries.columns.str.lower().str.strip()
        injuries["player"]=injuries["player"].astype(str).str.lower().str.strip()
    return skaters,shots,goalies,lines,teams,injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file)

if skaters_df.empty or shots_df.empty:
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty:
        df.columns=df.columns.str.lower().str.strip()

team_col=next(c for c in skaters_df.columns if "team" in c)
player_col="name"
shots_df=shots_df.rename(columns={next(c for c in shots_df.columns if "player" in c or "name" in c):"player"})
shots_df["player"]=shots_df["player"].astype(str)
game_col=next(c for c in shots_df.columns if "game" in c and "id" in c)

# ---------------------------------------------------------------
# Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_todays_games():
    r=requests.get("https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard")
    games=[]
    for e in r.json().get("events",[]):
        c=e["competitions"][0]["competitors"]
        games.append({
            "away":TEAM_ABBREV_MAP.get(c[0]["team"]["abbreviation"]),
            "home":TEAM_ABBREV_MAP.get(c[1]["team"]["abbreviation"]),
            "away_logo":c[0]["team"]["logo"],
            "home_logo":c[1]["team"]["logo"]
        })
    return games

games=get_todays_games()

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
col_run,col_line=st.columns([3,1])
with col_run:
    run_model=st.button("🚀 Run Model")
with col_line:
    line_test=st.number_input("Line to Test",0.0,10.0,3.5,0.5)
    st.session_state.line_test_val=line_test

# ---------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a,team_b):
    results=[]
    roster=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    grouped={n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    for r in roster.itertuples():
        player,team=r.name,r.team
        df_p=grouped.get(player.lower(),pd.DataFrame())
        if df_p.empty: continue

        sog=df_p.groupby(game_col)["sog"].sum().tolist()
        if len(sog)<3: continue

        l3,l5,l10=np.mean(sog[-3:]),np.mean(sog[-5:]),np.mean(sog[-10:])
        lam=round(0.55*l10+0.3*l5+0.15*l3,2)

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":lam,
            "Season Avg":round(np.mean(sog),2)
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model:
    all_dfs=[]
    for g in games:
        df=build_model(g["away"],g["home"])
        if not df.empty:
            all_dfs.append(df)

    df=pd.concat(all_dfs,ignore_index=True)

    # ✅ EDGE VS LINE
    test_line=st.session_state.line_test_val
    df["Edge vs Line"]=(df["Final Projection"]-test_line).round(2)

    # Probability
    df["Prob ≥ Line (%)"]=df["Final Projection"].apply(
        lambda x:round((1-poisson.cdf(test_line-1,mu=max(x,0.01)))*100,1)
    )

    # Sort by strongest edge
    df=df.sort_values(["Edge vs Line","Final Projection"],ascending=[False,False])

    st.session_state.results=df

# ---------------------------------------------------------------
# Display
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results
    cols=[
        "Player","Team","Final Projection",
        "Edge vs Line","Prob ≥ Line (%)","Season Avg"
    ]
    csv=df[cols].to_csv(index=False).encode("utf-8")

    st.download_button("💾 Download CSV",csv,"puck_shotz_results.csv","text/csv")
    components.html(df[cols].to_html(index=False),height=650,scrolling=True)
