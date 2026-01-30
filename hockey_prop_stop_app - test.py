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
# Team Abbreviation Normalization (ESPN -> Data)
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ":"NJD","LA":"LAK","SJ":"SJS","TB":"TBL"
}

st.set_page_config(page_title="Puck Shotz Hockey Analytics", layout="wide", page_icon="🏒")
st.warning("Production Version")

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files")
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
    if not f:
        return pd.DataFrame()
    return pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)

def safe_read(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)

def load_data(file_uploader, default_path):
    return load_file(file_uploader) if file_uploader else safe_read(default_path)

@st.cache_data(show_spinner=False)
def load_all(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    skaters = load_data(skaters_file, "Skaters.xlsx")
    shots   = load_data(shots_file,   "SHOT DATA.xlsx")
    goalies = load_data(goalies_file, "GOALTENDERS.xlsx")
    lines   = load_data(lines_file,   "LINE DATA.xlsx")
    teams   = load_data(teams_file,   "TEAMS.xlsx")
    injuries= load_data(injuries_file,"injuries.xlsx")

    for df in [skaters, shots, goalies, lines, teams, injuries]:
        if not df.empty:
            df.columns = df.columns.str.lower().str.strip()

    return skaters, shots, goalies, lines, teams, injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file
)

if skaters_df.empty or shots_df.empty:
    st.stop()

team_col   = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"
shots_df   = shots_df.rename(columns={c:"player" for c in shots_df.columns if "player" in c or "name" in c})
game_col   = next(c for c in shots_df.columns if "game" in c and "id" in c)

# ---------------------------------------------------------------
# ESPN Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_games():
    url="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    r=requests.get(url,timeout=10).json()
    games=[]
    for e in r.get("events",[]):
        c=e["competitions"][0]["competitors"]
        games.append({
            "away":TEAM_ABBREV_MAP.get(c[0]["team"]["abbreviation"],c[0]["team"]["abbreviation"]),
            "home":TEAM_ABBREV_MAP.get(c[1]["team"]["abbreviation"],c[1]["team"]["abbreviation"]),
            "away_logo":c[0]["team"]["logo"],
            "home_logo":c[1]["team"]["logo"]
        })
    return games

games = get_games()
if not games:
    st.stop()

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
run_model = st.button("🚀 Run Model")
line_test = st.number_input("Line to Test",0.0,10.0,3.5,0.5)

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b):
    results=[]
    skaters = skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster  = skaters[[player_col,team_col]].rename(columns={player_col:"player",team_col:"team"})
    grouped = {n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    for row in roster.itertuples(index=False):
        player, team = row.player, row.team
        df_p = grouped.get(player.lower())
        if df_p is None or "sog" not in df_p.columns:
            continue

        sog_vals = df_p.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals) < 3:
            continue

        last3,last5,last10 = sog_vals[-3:],sog_vals[-5:],sog_vals[-10:]
        l3,l5,l10 = map(np.mean,[last3,last5,last10])

        baseline = (0.35*l10)+(0.30*l5)+(0.35*l3)
        trend = (l5-l10)/l10 if l10>0 else 0

        # ---------------- HUNTER MODE ----------------
        form_boost = 1.0
        if (l3 > l5 > l10) and trend > 0:
            form_boost += min(0.25, trend*1.25)

        baseline *= form_boost
        # --------------------------------------------

        lam = baseline
        prob = 1-poisson.cdf(line_test-1,mu=max(lam,0.01))

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":round(lam,2),
            "Hunter":"🔥" if form_boost>1 else "",
            "Prob ≥ Line (%)":round(prob*100,1),
            "L3":round(l3,2),
            "L5":round(l5,2),
            "L10":round(l10,2)
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if run_model:
    out=[]
    for g in games:
        df = build_model(g["away"],g["home"])
        if not df.empty:
            df["Matchup"]=f'{g["away"]}@{g["home"]}'
            out.append(df)

    if out:
        final=pd.concat(out).sort_values("Final Projection",ascending=False)
        st.dataframe(final,use_container_width=True)
        st.download_button("Download CSV",final.to_csv(index=False),"puck_shotz.csv")
