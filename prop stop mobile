# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Mobile Friendly
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import requests, os, html, json
from scipy.stats import poisson

# ---------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Puck Shotz – Mobile",
    layout="centered",
    page_icon="🏒"
)

# ---------------------------------------------------------------
# Team Abbreviation Normalization
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ": "NJD",
    "LA": "LAK",
    "SJ": "SJS",
    "TB": "TBL"
}

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <h2 style="text-align:center;">🏒 Puck Shotz</h2>
    <p style="text-align:center;color:#9DB6D8;">
    Mobile-first NHL Shots on Goal Analytics
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Sidebar Uploads
# ---------------------------------------------------------------
st.sidebar.header("📂 Data Files")

skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("Shots", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("Goalies", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def load_file(f):
    if f is None:
        return pd.DataFrame()
    return pd.read_excel(f) if f.name.endswith("xlsx") else pd.read_csv(f)

@st.cache_data(show_spinner=False)
def load_data():
    skaters = load_file(skaters_file)
    shots   = load_file(shots_file)
    goalies = load_file(goalies_file)
    return skaters, shots, goalies

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df = load_data()

if skaters_df.empty or shots_df.empty:
    st.info("Upload Skaters and Shots data to begin.")
    st.stop()

# Normalize columns
for df in [skaters_df, shots_df, goalies_df]:
    df.columns = df.columns.str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name" if "name" in skaters_df.columns else skaters_df.columns[0]

skaters_df[team_col] = (
    skaters_df[team_col]
    .astype(str).str.upper().str.strip()
    .replace(TEAM_ABBREV_MAP)
)

shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c)

# ---------------------------------------------------------------
# ESPN Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_todays_games():
    url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    data = requests.get(url, timeout=10).json()
    games = []

    for e in data.get("events", []):
        comps = e.get("competitions", [{}])[0].get("competitors", [])
        if len(comps) == 2:
            a, h = comps
            games.append({
                "away": TEAM_ABBREV_MAP.get(a["team"]["abbreviation"], a["team"]["abbreviation"]),
                "home": TEAM_ABBREV_MAP.get(h["team"]["abbreviation"], h["team"]["abbreviation"]),
                "away_logo": a["team"]["logo"],
                "home_logo": h["team"]["logo"]
            })
    return games

games = get_todays_games()
if not games:
    st.warning("No games today.")
    st.stop()

# ---------------------------------------------------------------
# Run Button
# ---------------------------------------------------------------
run_model = st.button("🚀 Run Model")

# ---------------------------------------------------------------
# Core SOG Model (same bones)
# ---------------------------------------------------------------
def build_model(team_a, team_b):
    results = []
    roster = skaters_df[skaters_df[team_col].isin([team_a, team_b])]

    grouped = {
        n.lower(): g for n, g in shots_df.groupby(shots_df["player"].str.lower())
    }

    for _, row in roster.iterrows():
        player = row[player_col]
        team   = row[team_col]

        df_p = grouped.get(player.lower())
        if df_p is None or "sog" not in df_p.columns:
            continue

        sog_vals = df_p.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals) < 3:
            continue

        l3 = np.mean(sog_vals[-3:])
        l5 = np.mean(sog_vals[-5:])
        l10 = np.mean(sog_vals[-10:])

        lam = 0.55*l10 + 0.30*l5 + 0.15*l3

        prob = 1 - poisson.cdf(2, mu=max(lam, 0.01))

        results.append({
            "Player": player,
            "Team": team,
            "Final Projection": round(lam, 2),
            "Prob ≥ 3 (%)": round(prob * 100, 1),
            "L5": ", ".join(map(str, sog_vals[-5:])),
            "L10": ", ".join(map(str, sog_vals[-10:]))
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run + Store Results
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
        st.session_state.games = games
        st.success("Model complete")

# ---------------------------------------------------------------
# Mobile Matchup View
# ---------------------------------------------------------------
if "results" in st.session_state:
    results = st.session_state.results

    st.markdown("### Matchups")

    for g in st.session_state.games:
        matchup = f"{g['away']}@{g['home']}"

        if st.button(matchup, use_container_width=True):
            st.session_state.selected_matchup = matchup

    if "selected_matchup" in st.session_state:
        team_a, team_b = st.session_state.selected_matchup.split("@")

        tab_a, tab_b = st.tabs([team_a, team_b])

        def render_team(team, tab):
            with tab:
                df_t = results[
                    (results["Team"] == team) &
                    (results["Matchup"] == st.session_state.selected_matchup)
                ]

                if df_t.empty:
                    st.info("No players")
                    return

                for _, r in df_t.iterrows():
                    st.markdown(
                        f"""
                        <div style="
                            border:1px solid #1E5A99;
                            border-radius:12px;
                            padding:10px;
                            margin-bottom:10px;
                            background:#0F2743;
                        ">
                          <b>{r['Player']} – {r['Team']}</b><br>
                          <span style="color:#9DB6D8;">
                          Final Proj: {r['Final Projection']}<br>
                          Prob ≥ 3: {r['Prob ≥ 3 (%)']}%<br>
                          L5: {r['L5']}<br>
                          L10: {r['L10']}
                          </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        render_team(team_a, tab_a)
        render_team(team_b, tab_b)
