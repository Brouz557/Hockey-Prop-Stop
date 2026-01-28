# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (Tabs + Full Styling)
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
    "NJ":"NJD","LA":"LAK","SJ":"SJS","TB":"TBL","ARI":"ARI","ANA":"ANA","BOS":"BOS",
    "BUF":"BUF","CAR":"CAR","CBJ":"CBJ","CGY":"CGY","CHI":"CHI","COL":"COL","DAL":"DAL",
    "DET":"DET","EDM":"EDM","FLA":"FLA","MIN":"MIN","MTL":"MTL","NSH":"NSH","NYI":"NYI",
    "NYR":"NYR","OTT":"OTT","PHI":"PHI","PIT":"PIT","SEA":"SEA","STL":"STL","TOR":"TOR",
    "VAN":"VAN","VGK":"VGK","WSH":"WSH","WPG":"WPG"
}

st.set_page_config(page_title="Puck Shotz Hockey Analytics", layout="wide", page_icon="🏒")
st.warning("Production Version")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background:#123A63; padding:14px; border-radius:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#E6F0FF;'>Puck Shotz Hockey Analytics</h1>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# File Uploads
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files")
skaters_file = st.sidebar.file_uploader("Skaters", ["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("Shot Data", ["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("Goalies", ["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("Lines", ["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("Teams", ["xlsx","csv"])
injuries_file= st.sidebar.file_uploader("Injuries", ["xlsx","csv"])

def load_file(f):
    if f is None: return pd.DataFrame()
    return pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)

skaters_df = load_file(skaters_file)
shots_df   = load_file(shots_file)
goalies_df = load_file(goalies_file)
lines_df   = load_file(lines_file)
teams_df   = load_file(teams_file)
injuries_df= load_file(injuries_file)

if skaters_df.empty or shots_df.empty:
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"
shots_df = shots_df.rename(columns={next(c for c in shots_df.columns if "player" in c or "name" in c):"player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c)

# ---------------------------------------------------------------
# ESPN Games
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_games():
    url="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    data=requests.get(url,timeout=10).json()
    games=[]
    for e in data.get("events",[]):
        a,h=e["competitions"][0]["competitors"]
        games.append({
            "away":TEAM_ABBREV_MAP.get(a["team"]["abbreviation"],a["team"]["abbreviation"]),
            "home":TEAM_ABBREV_MAP.get(h["team"]["abbreviation"],h["team"]["abbreviation"]),
            "away_logo":a["team"]["logo"],
            "home_logo":h["team"]["logo"]
        })
    return games

games = get_games()

# ---------------------------------------------------------------
# Line Input
# ---------------------------------------------------------------
line = st.number_input("🎯 Line to Test", 0.0, 10.0, 3.5, 0.5)

# ---------------------------------------------------------------
# BUILD MODEL (FULL FEATURES)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b):
    results=[]
    roster = skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    grouped = {n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    for _,r in roster.iterrows():
        player = r[player_col]
        team = r[team_col]
        dfp = grouped.get(player.lower())
        if dfp is None or "sog" not in dfp.columns: continue

        sog_vals = dfp.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals)<3: continue

        l3,l5,l10 = np.mean(sog_vals[-3:]),np.mean(sog_vals[-5:]),np.mean(sog_vals[-10:])
        baseline = 0.55*l10 + 0.3*l5 + 0.15*l3
        trend = (l5-l10)/l10 if l10>0 else 0
        form = "🟢 Above" if trend>0.05 else "🔴 Below" if trend<-0.05 else "⚪ Neutral"

        lam = baseline
        prob = 1-poisson.cdf(line-1, mu=max(lam,0.01))
        odds = -100*(prob/(1-prob)) if prob>=0.5 else 100*((1-prob)/prob)

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":round(lam,2),
            "Prob ≥ Line (%)":round(prob*100,1),
            "Playable Odds":f"{'+' if odds>0 else ''}{int(odds)}",
            "Season Avg":round(np.mean(sog_vals),2),
            "Trend":trend,
            "Form":form,
            "L3":", ".join(map(str,sog_vals[-3:])),
            "L5":", ".join(map(str,sog_vals[-5:])),
            "L10":", ".join(map(str,sog_vals[-10:])),
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# RUN
# ---------------------------------------------------------------
if st.button("🚀 Run Model"):
    all=[]
    for g in games:
        df=build_model(g["away"],g["home"])
        if not df.empty:
            df["Matchup"]=f"{g['away']}@{g['home']}"
            all.append(df)
    st.session_state.results=pd.concat(all,ignore_index=True)
    st.session_state.games=games

# ---------------------------------------------------------------
# DISPLAY — TABS + STYLED HTML
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results
    games=st.session_state.games

    cols=st.columns(3)
    for i,g in enumerate(games):
        with cols[i%3]:
            if st.button(f"{g['away']} @ {g['home']}"):
                st.session_state.sel=f"{g['away']}@{g['home']}"

    if "sel" not in st.session_state:
        st.stop()

    ta,tb = st.session_state.sel.split("@")
    tabs = st.tabs([ta,tb])

    def render(team,tab):
        with tab:
            tdf=df[(df["Team"]==team)&(df["Matchup"]==st.session_state.sel)]
            html_table = tdf.to_html(index=False)
            components.html(f"""
            <style>
            table {{width:100%;background:#F5F7FA;color:#000}}
            th {{background:#DDE6F0}}
            </style>
            {html_table}
            """,height=700)

    render(ta,tabs[0])
    render(tb,tabs[1])
