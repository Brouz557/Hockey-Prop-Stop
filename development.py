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
    "NJ": "NJD", "LA": "LAK", "SJ": "SJS", "TB": "TBL",
    "ARI": "ARI", "ANA": "ANA", "BOS": "BOS", "BUF": "BUF",
    "CAR": "CAR", "CBJ": "CBJ", "CGY": "CGY", "CHI": "CHI",
    "COL": "COL", "DAL": "DAL", "DET": "DET", "EDM": "EDM",
    "FLA": "FLA", "MIN": "MIN", "MTL": "MTL", "NSH": "NSH",
    "NYI": "NYI", "NYR": "NYR", "OTT": "OTT", "PHI": "PHI",
    "PIT": "PIT", "SEA": "SEA", "STL": "STL", "TOR": "TOR",
    "VAN": "VAN", "VGK": "VGK", "WSH": "WSH", "WPG": "WPG"
}

st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("🧪 TEST MODE — Sandbox version. Changes here won’t affect your main app.")

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
    for p in ["injuries.xlsx","Injuries.xlsx","data/injuries.xlsx"]:
        if os.path.exists(p):
            injuries=load_file(open(p,"rb"));break
    if injuries.empty:
        injuries=load_file(injuries_file)

    if not injuries.empty:
        injuries.columns=injuries.columns.str.lower().str.strip()
        injuries["player"]=injuries["player"].astype(str).str.lower().str.strip()

    return skaters,shots,goalies,lines,teams,injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file
)

if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing data. Upload required files.")
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    df.columns=df.columns.str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"
shots_df = shots_df.rename(columns={next(c for c in shots_df.columns if "player" in c):"player"})
shots_df["player"]=shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c and "id" in c)

# ---------------------------------------------------------------
# Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_todays_games():
    url="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    r=requests.get(url,timeout=10)
    games=[]
    for e in r.json().get("events",[]):
        c=e["competitions"][0]["competitors"]
        games.append({
            "away": TEAM_ABBREV_MAP.get(c[0]["team"]["abbreviation"], c[0]["team"]["abbreviation"]),
            "home": TEAM_ABBREV_MAP.get(c[1]["team"]["abbreviation"], c[1]["team"]["abbreviation"]),
            "away_logo": c[0]["team"]["logo"],
            "home_logo": c[1]["team"]["logo"]
        })
    return games

games=get_todays_games()
if not games:
    st.stop()

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
col_run,col_line=st.columns([3,1])
with col_run: run_model=st.button("🚀 Run Model (All Games)")
with col_line:
    line_test=st.number_input("Line to Test",0.0,10.0,3.5,0.5)
    st.session_state.setdefault("line_test_val",line_test)
    if line_test!=st.session_state.line_test_val:
        st.session_state.line_test_val=line_test

# ---------------------------------------------------------------
# MODEL (WITH CORSI INTEGRATION)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a,team_b):
    results=[]

    skaters=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster=skaters[[player_col,team_col,"on ice corsi"]].drop_duplicates()

    league_player_corsi=skaters_df["on ice corsi"].mean()
    league_team_cp=teams_df["corsi%"].mean()

    for r in roster.itertuples(index=False):
        player,team,player_corsi=r
        dfp=shots_df[shots_df["player"].str.lower()==player.lower()]
        if dfp.empty: continue

        sog_vals=dfp.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals)<3: continue

        l3,l5,l10=np.mean(sog_vals[-3:]),np.mean(sog_vals[-5:]),np.mean(sog_vals[-10:])
        baseline=0.55*l10+0.3*l5+0.15*l3
        trend=(l5-l10)/l10 if l10>0 else 0

        player_corsi_factor=np.clip(player_corsi/league_player_corsi,0.85,1.20)

        opp=team_b if team==team_a else team_a
        team_cp=teams_df.loc[teams_df["team"]==team,"corsi%"].mean()
        opp_cp=teams_df.loc[teams_df["team"]==opp,"corsi%"].mean()
        pace_factor=np.clip(((team_cp+opp_cp)/2)/league_team_cp,0.92,1.08)

        lam=baseline*(1+0.15*(player_corsi_factor-1))*pace_factor
        lam=np.clip(lam,baseline*0.6,baseline*1.4)

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":round(lam,2),
            "Trend Score":round(trend,3),
            "Season Avg":round(np.mean(sog_vals),2),
            "Line Adj":round(player_corsi_factor,2)
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model:
    tables=[]
    for g in games:
        df=build_model(g["away"],g["home"])
        if not df.empty:
            df["Matchup"]=f"{g['away']}@{g['home']}"
            tables.append(df)

    if tables:
        st.session_state.results=pd.concat(tables,ignore_index=True)
        st.success("✅ Model built")

# ---------------------------------------------------------------
# Display
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results.copy()
    test_line=st.session_state.line_test_val
    df["Prob ≥ Line (%)"]=df["Final Projection"].apply(
        lambda x:round((1-poisson.cdf(test_line-1,mu=max(x,0.01)))*100,1)
    )
    st.dataframe(df.sort_values("Final Projection",ascending=False),use_container_width=True)
