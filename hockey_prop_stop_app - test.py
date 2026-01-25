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
    "NJ":"NJD","LA":"LAK","SJ":"SJS","TB":"TBL","ARI":"ARI","ANA":"ANA","BOS":"BOS",
    "BUF":"BUF","CAR":"CAR","CBJ":"CBJ","CGY":"CGY","CHI":"CHI","COL":"COL","DAL":"DAL",
    "DET":"DET","EDM":"EDM","FLA":"FLA","MIN":"MIN","MTL":"MTL","NSH":"NSH","NYI":"NYI",
    "NYR":"NYR","OTT":"OTT","PHI":"PHI","PIT":"PIT","SEA":"SEA","STL":"STL","TOR":"TOR",
    "VAN":"VAN","VGK":"VGK","WSH":"WSH","WPG":"WPG"
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
    except:
        return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except:
        return pd.DataFrame()

def load_data(file_uploader, default_path):
    return load_file(file_uploader) if file_uploader else safe_read(default_path)

@st.cache_data(show_spinner=False)
def load_all(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    skaters=load_data(skaters_file,"Skaters.xlsx")
    shots=load_data(shots_file,"SHOT DATA.xlsx")
    goalies=load_data(goalies_file,"GOALTENDERS.xlsx")
    lines=load_data(lines_file,"LINE DATA.xlsx")
    teams=load_data(teams_file,"TEAMS.xlsx")
    injuries=load_data(injuries_file,"")
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

st.success("✅ Data loaded successfully.")

for d in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    d.columns=d.columns.str.lower().str.strip()

team_col=next(c for c in skaters_df.columns if "team" in c)
player_col="name"
shots_df=shots_df.rename(columns={next(c for c in shots_df.columns if "player" in c or "name" in c):"player"})
shots_df["player"]=shots_df["player"].astype(str).str.strip()
game_col=next(c for c in shots_df.columns if "game" in c and "id" in c)

# ---------------------------------------------------------------
# Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_games():
    r=requests.get("https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",timeout=10)
    games=[]
    for e in r.json().get("events",[]):
        c=e["competitions"][0]["competitors"]
        games.append({
            "away":TEAM_ABBREV_MAP.get(c[0]["team"]["abbreviation"],c[0]["team"]["abbreviation"]),
            "home":TEAM_ABBREV_MAP.get(c[1]["team"]["abbreviation"],c[1]["team"]["abbreviation"]),
            "away_logo":c[0]["team"]["logo"],
            "home_logo":c[1]["team"]["logo"]
        })
    return games

games=get_games()

# ---------------------------------------------------------------
# Run Button
# ---------------------------------------------------------------
col_run,col_line=st.columns([3,1])
with col_run:
    run_model=st.button("🚀 Run Model (All Games)")
with col_line:
    line_test=st.number_input("Line to Test",0.0,10.0,3.5,0.5)
    st.session_state.line_test_val=line_test

# ---------------------------------------------------------------
# Build Model (UNCHANGED)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results=[]
    skaters=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster=skaters[[player_col,team_col]].rename(columns={player_col:"player",team_col:"team"}).drop_duplicates("player")
    grouped={n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    for row in roster.itertuples(index=False):
        player,team=row.player,row.team
        df_p=grouped.get(player.lower(),pd.DataFrame())
        if df_p.empty or "sog" not in df_p.columns: continue

        sog_vals=df_p.groupby(game_col)["sog"].sum().tolist()
        if not sog_vals: continue

        last3,last5,last10=sog_vals[-3:],sog_vals[-5:],sog_vals[-10:]
        baseline=0.55*np.mean(last10)+0.3*np.mean(last5)+0.15*np.mean(last3)
        lam=baseline

        if "goal" in df_p.columns:
            g=df_p.groupby(game_col).agg({"sog":"sum","goal":"sum"})
            shooting_pct=(g["goal"].mean()/g["sog"].mean()) if g["sog"].mean()>0 else 0
            xg=shooting_pct*lam
        else:
            shooting_pct,xg=np.nan,np.nan

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":round(lam,2),
            "Season Avg":round(np.mean(sog_vals),2),
            "Line Adj":1.0,
            "Exp Goals (xG)":round(xg,3),
            "Shooting %":round(shooting_pct*100,2),
            "L3 Shots":", ".join(map(str,last3)),
            "L5 Shots":", ".join(map(str,last5)),
            "L10 Shots":", ".join(map(str,last10))
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# RUN MODEL (RESTORED)
# ---------------------------------------------------------------
if run_model:
    all_tables=[]
    for m in games:
        df=build_model(m["away"],m["home"],skaters_df,shots_df,goalies_df,lines_df,teams_df,injuries_df)
        if not df.empty:
            all_tables.append(df)

    if all_tables:
        st.session_state.results=pd.concat(all_tables,ignore_index=True)
        st.success("✅ Model built successfully.")
        st.experimental_rerun()

# ---------------------------------------------------------------
# DISPLAY + FILTER
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results.copy()

    df["Shooting %"]=pd.to_numeric(df["Shooting %"],errors="coerce")
    df["Exp Goals (xG)"]=pd.to_numeric(df["Exp Goals (xG)"],errors="coerce")

    # 🔥 YOUR FILTER
    df=df[
        (df["Final Projection"]>1.5) &
        (df["Line Adj"]>0.94) &
        (df["Shooting %"]>5) &
        (df["Exp Goals (xG)"]>0.29)
    ]

    st.dataframe(df, use_container_width=True)
