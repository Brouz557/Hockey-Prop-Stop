# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — STABLE TAB VERSION
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

# ---------------------------------------------------------------
# Load data EXACTLY like your working version
# ---------------------------------------------------------------
def safe_read(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)

def find_file(name):
    for p in [".", "data", "/mount/src/hockey-prop-stop/data"]:
        fp = os.path.join(p, name)
        if os.path.exists(fp):
            return fp
    return None

@st.cache_data(show_spinner=False)
def load_all():
    return (
        safe_read(find_file("Skaters.xlsx")),
        safe_read(find_file("SHOT DATA.xlsx")),
        safe_read(find_file("GOALTENDERS.xlsx")),
        safe_read(find_file("LINE DATA.xlsx")),
        safe_read(find_file("TEAMS.xlsx")),
    )

skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all()

if skaters_df.empty or shots_df.empty:
    st.error("Data files not found.")
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    df.columns = df.columns.str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"
shots_df = shots_df.rename(columns={next(c for c in shots_df.columns if "player" in c or "name" in c):"player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c)

# ---------------------------------------------------------------
# ESPN Matchups
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
# Run model (UNCHANGED CORE LOGIC)
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
        if dfp is None or "sog" not in dfp.columns:
            continue

        sog_vals = dfp.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals) < 3:
            continue

        l3,l5,l10 = np.mean(sog_vals[-3:]),np.mean(sog_vals[-5:]),np.mean(sog_vals[-10:])
        baseline = 0.55*l10 + 0.3*l5 + 0.15*l3

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":round(baseline,2),
            "Season Avg":round(np.mean(sog_vals),2),
            "L3":", ".join(map(str,sog_vals[-3:])),
            "L5":", ".join(map(str,sog_vals[-5:])),
            "L10":", ".join(map(str,sog_vals[-10:])),
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# RUN BUTTON
# ---------------------------------------------------------------
if st.button("🚀 Run Model (All Games)"):
    out=[]
    for g in games:
        df=build_model(g["away"],g["home"])
        if not df.empty:
            df["Matchup"]=f"{g['away']}@{g['home']}"
            out.append(df)
    st.session_state.results=pd.concat(out,ignore_index=True)
    st.session_state.games=games

# ---------------------------------------------------------------
# DISPLAY — TEAM TABS (SAFE)
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

    ta,tb=st.session_state.sel.split("@")
    tabs=st.tabs([ta,tb])

    def render(team,tab):
        with tab:
            tdf=df[(df["Team"]==team)&(df["Matchup"]==st.session_state.sel)]
            st.dataframe(tdf,use_container_width=True,height=650)

    render(ta,tabs[0])
    render(tb,tabs[1])
