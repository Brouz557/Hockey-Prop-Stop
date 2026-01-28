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
# Helper functions
# ---------------------------------------------------------------
def load_file(f):
    if not f: return pd.DataFrame()
    return pd.read_excel(f) if f.name.lower().endswith(".xlsx") else pd.read_csv(f)

def safe_read(path):
    if not path or not os.path.exists(path): return pd.DataFrame()
    return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)

def load_data(f, default):
    return load_file(f) if f is not None else safe_read(default)

@st.cache_data(show_spinner=False)
def load_all():
    return (
        load_data(skaters_file,"Skaters.xlsx"),
        load_data(shots_file,"SHOT DATA.xlsx"),
        load_data(goalies_file,"GOALTENDERS.xlsx"),
        load_data(lines_file,"LINE DATA.xlsx"),
        load_data(teams_file,"TEAMS.xlsx"),
        load_data(injuries_file,"injuries.xlsx")
    )

skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all()
if skaters_df.empty or shots_df.empty:
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    df.columns = df.columns.str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"
shots_df = shots_df.rename(columns={next(c for c in shots_df.columns if "player" in c or "name" in c):"player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c)

# ---------------------------------------------------------------
# BUILD MODEL (RESTORED — UNCHANGED)
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
        if len(sog_vals)<3: continue

        l3,l5,l10=np.mean(sog_vals[-3:]),np.mean(sog_vals[-5:]),np.mean(sog_vals[-10:])
        baseline=(0.55*l10)+(0.3*l5)+(0.15*l3)
        lam=baseline

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":round(lam,2),
            "Season Avg":round(np.mean(sog_vals),2),
            "L3":", ".join(map(str,sog_vals[-3:])),
            "L5":", ".join(map(str,sog_vals[-5:])),
            "L10":", ".join(map(str,sog_vals[-10:]))
        })
    return pd.DataFrame(results)

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

games=get_games()

# ---------------------------------------------------------------
# Run model
# ---------------------------------------------------------------
if st.button("🚀 Run Model (All Games)"):
    all_tables=[]
    for g in games:
        df=build_model(g["away"],g["home"],skaters_df,shots_df,goalies_df,lines_df,teams_df,injuries_df)
        if not df.empty:
            df["Matchup"]=f"{g['away']}@{g['home']}"
            all_tables.append(df)
    if all_tables:
        st.session_state.results=pd.concat(all_tables,ignore_index=True)
        st.session_state.matchups=games
        st.success("Model built successfully")

# ---------------------------------------------------------------
# DISPLAY — TEAM TABS (WORKING)
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results
    games=st.session_state.matchups

    cols=st.columns(3)
    for i,g in enumerate(games):
        with cols[i%3]:
            if st.button(f"{g['away']} @ {g['home']}"):
                st.session_state.selected_match=f"{g['away']}@{g['home']}"

    if "selected_match" not in st.session_state:
        st.stop()

    team_a,team_b=st.session_state.selected_match.split("@")
    tabs=st.tabs([team_a,team_b])

    def render(team,tab):
        with tab:
            tdf=df[(df["Team"]==team)&(df["Matchup"]==st.session_state.selected_match)]
            st.dataframe(tdf,use_container_width=True,height=650)

    render(team_a,tabs[0])
    render(team_b,tabs[1])
