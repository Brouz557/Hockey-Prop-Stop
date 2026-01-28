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
    "NJ": "NJD","LA": "LAK","SJ": "SJS","TB": "TBL","ARI": "ARI","ANA": "ANA","BOS": "BOS",
    "BUF": "BUF","CAR": "CAR","CBJ": "CBJ","CGY": "CGY","CHI": "CHI","COL": "COL","DAL": "DAL",
    "DET": "DET","EDM": "EDM","FLA": "FLA","MIN": "MIN","MTL": "MTL","NSH": "NSH","NYI": "NYI",
    "NYR": "NYR","OTT": "OTT","PHI": "PHI","PIT": "PIT","SEA": "SEA","STL": "STL","TOR": "TOR",
    "VAN": "VAN","VGK": "VGK","WSH": "WSH","WPG": "WPG"
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
# Helpers
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
    return load_file(file_uploader) if file_uploader is not None else safe_read(default_path)

@st.cache_data(show_spinner=False)
def load_all():
    return (
        load_data(skaters_file, "Skaters.xlsx"),
        load_data(shots_file, "SHOT DATA.xlsx"),
        load_data(goalies_file, "GOALTENDERS.xlsx"),
        load_data(lines_file, "LINE DATA.xlsx"),
        load_data(teams_file, "TEAMS.xlsx"),
        load_data(injuries_file, "injuries.xlsx")
    )

skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all()
if skaters_df.empty or shots_df.empty:
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    df.columns = df.columns.str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"
shots_df = shots_df.rename(columns={next(c for c in shots_df.columns if "player" in c or "name" in c): "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c)

# ---------------------------------------------------------------
# ESPN Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_games():
    url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    data = requests.get(url, timeout=10).json()
    games = []
    for e in data.get("events", []):
        a, h = e["competitions"][0]["competitors"]
        games.append({
            "away": TEAM_ABBREV_MAP.get(a["team"]["abbreviation"], a["team"]["abbreviation"]),
            "home": TEAM_ABBREV_MAP.get(h["team"]["abbreviation"], h["team"]["abbreviation"]),
            "away_logo": a["team"]["logo"],
            "home_logo": h["team"]["logo"]
        })
    return games

games = get_games()

# ---------------------------------------------------------------
# Run model (unchanged)
# ---------------------------------------------------------------
if st.button("🚀 Run Model (All Games)"):
    results = []
    for g in games:
        for _, r in skaters_df[skaters_df[team_col].isin([g["away"], g["home"]])].iterrows():
            df_p = shots_df[shots_df["player"].str.lower() == r[player_col].lower()]
            if df_p.empty: continue
            sog = df_p.groupby(game_col)["sog"].sum().tolist()
            if len(sog) < 3: continue
            lam = np.mean(sog[-10:])
            results.append({
                "Player": r[player_col],
                "Team": r[team_col],
                "Matchup": f"{g['away']}@{g['home']}",
                "Final Projection": round(lam,2),
                "Season Avg": round(np.mean(sog),2),
                "L3": ", ".join(map(str, sog[-3:])),
                "L5": ", ".join(map(str, sog[-5:])),
                "L10": ", ".join(map(str, sog[-10:])),
            })
    st.session_state.results = pd.DataFrame(results)
    st.session_state.matchups = games
    st.success("Model built")

# ---------------------------------------------------------------
# DISPLAY — TEAM TABS (NEW)
# ---------------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state.results
    games = st.session_state.matchups

    cols = st.columns(3)
    for i, g in enumerate(games):
        with cols[i % 3]:
            if st.button(f"{g['away']} @ {g['home']}"):
                st.session_state.selected_match = f"{g['away']}@{g['home']}"

    if "selected_match" not in st.session_state:
        st.stop()

    team_a, team_b = st.session_state.selected_match.split("@")
    tabs = st.tabs([team_a, team_b])

    def render(team, tab):
        with tab:
            tdf = (
                df[(df["Team"] == team) & (df["Matchup"] == st.session_state.selected_match)]
                .sort_values("Final Projection", ascending=False)
            )
            st.dataframe(tdf, use_container_width=True, height=650)

    render(team_a, tabs[0])
    render(team_b, tabs[1])
