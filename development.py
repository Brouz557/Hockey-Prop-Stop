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
# AUTO LOAD DATA (NO UPLOADERS)
# ---------------------------------------------------------------
DATA_PATHS = {
    "skaters":   "Skaters.xlsx",
    "shots":     "SHOT DATA.xlsx",
    "goalies":   "GOALTENDERS.xlsx",
    "lines":     "LINE DATA.xlsx",
    "teams":     "TEAMS.xlsx",
    "injuries":  "injuries.xlsx"
}

def must_load(path):
    if not os.path.exists(path):
        st.error(f"❌ Missing required file: {path}")
        st.stop()
    return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)

skaters_df  = must_load(DATA_PATHS["skaters"])
shots_df    = must_load(DATA_PATHS["shots"])
goalies_df  = must_load(DATA_PATHS["goalies"])
lines_df    = must_load(DATA_PATHS["lines"])
teams_df    = must_load(DATA_PATHS["teams"])
injuries_df = must_load(DATA_PATHS["injuries"]) if os.path.exists(DATA_PATHS["injuries"]) else pd.DataFrame()

st.success("✅ Data auto-loaded successfully.")

# ---------------------------------------------------------------
# Normalize Columns
# ---------------------------------------------------------------
for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    df.columns = df.columns.str.lower().str.strip()

if not injuries_df.empty:
    injuries_df.columns = injuries_df.columns.str.lower().str.strip()
    injuries_df["player"] = injuries_df["player"].astype(str).str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"

shots_df = shots_df.rename(
    columns={next(c for c in shots_df.columns if "player" in c or "name" in c): "player"}
)
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c and "id" in c)

# ---------------------------------------------------------------
# Matchups (ESPN)
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_todays_games():
    url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    r = requests.get(url, timeout=10)
    games = []
    for e in r.json().get("events", []):
        c = e["competitions"][0]["competitors"]
        games.append({
            "away": TEAM_ABBREV_MAP.get(c[0]["team"]["abbreviation"], c[0]["team"]["abbreviation"]),
            "home": TEAM_ABBREV_MAP.get(c[1]["team"]["abbreviation"], c[1]["team"]["abbreviation"]),
            "away_logo": c[0]["team"]["logo"],
            "home_logo": c[1]["team"]["logo"]
        })
    return games

games = get_todays_games()
if not games:
    st.warning("No games found today.")
    st.stop()

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
col_run, col_line = st.columns([3,1])
with col_run:
    run_model = st.button("🚀 Run Model (All Games)")
with col_line:
    line_test = st.number_input("Line to Test", 0.0, 10.0, 3.5, 0.5)
    st.session_state.setdefault("line_test_val", line_test)
    if line_test != st.session_state.line_test_val:
        st.session_state.line_test_val = line_test

# ---------------------------------------------------------------
# MODEL (WITH CORSI)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b):

    results = []

    skaters = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
    roster = skaters[[player_col, team_col, "on ice corsi"]].drop_duplicates()

    league_player_corsi = skaters_df["on ice corsi"].mean()
    league_team_corsi_pct = teams_df["corsi%"].mean()

    for r in roster.itertuples(index=False):
        player, team, player_corsi = r
        dfp = shots_df[shots_df["player"].str.lower() == player.lower()]
        if dfp.empty:
            continue

        sog_vals = dfp.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals) < 3:
            continue

        l3, l5, l10 = np.mean(sog_vals[-3:]), np.mean(sog_vals[-5:]), np.mean(sog_vals[-10:])
        baseline = 0.55*l10 + 0.3*l5 + 0.15*l3
        trend = (l5-l10)/l10 if l10 > 0 else 0

        player_corsi_factor = np.clip(player_corsi / league_player_corsi, 0.85, 1.20)

        opp = team_b if team == team_a else team_a
        team_cp = teams_df.loc[teams_df["team"] == team, "corsi%"].mean()
        opp_cp = teams_df.loc[teams_df["team"] == opp, "corsi%"].mean()
        pace_factor = np.clip(((team_cp + opp_cp)/2) / league_team_corsi_pct, 0.92, 1.08)

        lam = baseline
        lam *= (1 + 0.15*(player_corsi_factor - 1))
        lam *= pace_factor
        lam = np.clip(lam, baseline*0.6, baseline*1.4)

        results.append({
            "Player": player,
            "Team": team,
            "Final Projection": round(lam,2),
            "Trend Score": round(trend,3),
            "Season Avg": round(np.mean(sog_vals),2),
            "Line Adj": round(player_corsi_factor,2)
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model:
    tables = []
    for g in games:
        df = build_model(g["away"], g["home"])
        if not df.empty:
            df["Matchup"] = f"{g['away']}@{g['home']}"
            tables.append(df)

    if tables:
        st.session_state.results = pd.concat(tables, ignore_index=True)
        st.success("✅ Model built successfully.")

# ---------------------------------------------------------------
# Display
# ---------------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state.results.copy()
    test_line = st.session_state.line_test_val

    df["Prob ≥ Line (%)"] = df["Final Projection"].apply(
        lambda x: round((1 - poisson.cdf(test_line - 1, mu=max(x, 0.01))) * 100, 1)
    )

    st.dataframe(
        df.sort_values("Final Projection", ascending=False),
        use_container_width=True
    )
