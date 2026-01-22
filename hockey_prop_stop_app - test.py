# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (with Goals + Trends)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests
from scipy.stats import poisson
import streamlit.components.v1 as components
from datetime import datetime

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
<p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups.</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file  = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file    = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file  = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file    = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file    = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])
injuries_file = st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def load_file(f):
    if not f:
        return pd.DataFrame()
    try:
        return pd.read_excel(f) if f.name.lower().endswith(".xlsx") else pd.read_csv(f)
    except Exception:
        return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None:
        return load_file(file_uploader)
    return safe_read(default_path)

@st.cache_data(show_spinner=False)
def load_all(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    skaters  = load_data(skaters_file, "Skaters.xlsx")
    shots    = load_data(shots_file, "SHOT DATA.xlsx")
    goalies  = load_data(goalies_file, "GOALTENDERS.xlsx")
    lines    = load_data(lines_file, "LINE DATA.xlsx")
    teams    = load_data(teams_file, "TEAMS.xlsx")
    injuries = load_data(injuries_file, "injuries.xlsx")
    return skaters, shots, goalies, lines, teams, injuries

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

# normalize column names
for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

# 🔹 REQUIRED FIX: shots file uses "name" → normalize to "player"
if "name" in shots_df.columns and "player" not in shots_df.columns:
    shots_df = shots_df.rename(columns={"name": "player"})

# ---------------------------------------------------------------
# ESPN Matchups (FIXED)
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_todays_games():
    today = datetime.utcnow().strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={today}"

    r = requests.get(
        url,
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    r.raise_for_status()
    data = r.json()

    games = []
    for e in data.get("events", []):
        comps = e.get("competitions", [{}])[0].get("competitors", [])
        if len(comps) >= 2:
            home = next(c for c in comps if c.get("homeAway") == "home")
            away = next(c for c in comps if c.get("homeAway") == "away")
            games.append({
                "away": away["team"]["abbreviation"],
                "home": home["team"]["abbreviation"]
            })
    return games

games = get_todays_games()
st.caption(f"ESPN games found: {len(games)}")
if not games:
    st.warning("No games found today (ESPN may not have published the slate yet).")

# ---------------------------------------------------------------
# MODEL FUNCTION (DEFINED BEFORE USE)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results = []

    team_col   = next((c for c in skaters_df.columns if "team" in c), None)
    player_col = "name" if "name" in skaters_df.columns else "player"
    game_col   = next((c for c in shots_df.columns if "game" in c and "id" in c), None)

    skaters = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
    roster  = skaters[[player_col, team_col]].drop_duplicates()

    for _, row in roster.iterrows():
        player = row[player_col]
        team   = row[team_col]

        df_p = shots_df[shots_df["player"].str.lower() == str(player).lower()]
        if df_p.empty or "sog" not in df_p.columns:
            continue

        agg   = df_p.groupby(game_col)["sog"].sum().reset_index()
        shots = agg["sog"].tolist()
        if not shots:
            continue

        lam  = np.mean(shots[-5:])
        prob = 1 - poisson.cdf(2.5, mu=max(lam, 0.01))

        results.append({
            "Player": player,
            "Team": team,
            "Final Projection": round(lam, 2),
            "Prob ≥ 3 (%)": round(prob * 100, 1)
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Button
# ---------------------------------------------------------------
col_run, _ = st.columns([3, 1])
with col_run:
    run_model = st.button("🚀 Run Model (All Games)")

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model and games:
    all_tables = []

    for g in games:
        df = build_model(
            g["away"], g["home"],
            skaters_df, shots_df, goalies_df,
            lines_df, teams_df, injuries_df
        )
        if not df.empty:
            df["Matchup"] = f'{g["away"]}@{g["home"]}'
            all_tables.append(df)

    if all_tables:
        st.session_state.results = pd.concat(all_tables, ignore_index=True)
        st.success("✅ Model complete")

elif run_model and not games:
    st.warning("Cannot run model — no games available.")

# ---------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------
if "results" in st.session_state:
    components.html(
        st.session_state.results.to_html(index=False),
        height=650,
        scrolling=True
    )