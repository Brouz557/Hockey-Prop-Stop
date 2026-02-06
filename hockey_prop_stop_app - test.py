# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Mobile (ORIGINAL STABLE)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Puck Shotz Hockey Analytics (Mobile)",
    layout="wide",
    page_icon="🏒"
)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67;
            padding:14px; border-radius:8px; margin-bottom:12px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png'
       width='200'>
</div>
<h3 style='text-align:center;color:#1E5A99;margin-top:0;'>
    Puck Shotz Hockey Analytics
</h3>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def safe_read(path):
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def find_file(name):
    for p in [".", "data", "/mount/src/hockey-prop-stop/data"]:
        fp = os.path.join(p, name)
        if os.path.exists(fp):
            return fp
    return None

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
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
    st.error("Missing required data files.")
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    df.columns = df.columns.str.lower().str.strip()

# ---------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------
team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name" if "name" in skaters_df.columns else skaters_df.columns[0]

shots_player_col = next(c for c in shots_df.columns if "player" in c or "name" in c)
shots_df = shots_df.rename(columns={shots_player_col: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()

game_col = next(c for c in shots_df.columns if "game" in c)

# ---------------------------------------------------------------
# ESPN Matchups (ORIGINAL)
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ": "NJD",
    "LA": "LAK",
    "SJ": "SJS",
    "TB": "TBL"
}

@st.cache_data(ttl=300)
def get_games():
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

games = get_games()
if not games:
    st.warning("No games found today.")
    st.stop()

# ---------------------------------------------------------------
# Line test
# ---------------------------------------------------------------
st.session_state.setdefault("line_test_val", 3.5)
line_test = st.number_input(
    "Line to Test",
    min_value=0.0,
    max_value=10.0,
    step=0.5,
    value=st.session_state.line_test_val
)
st.session_state.line_test_val = line_test

# ---------------------------------------------------------------
# Run model (ORIGINAL)
# ---------------------------------------------------------------
if st.button("Run Model (All Games)", use_container_width=True):
    results = []

    for g in games:
        team_a, team_b = g["away"], g["home"]
        roster = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
        grouped = {n: d for n, d in shots_df.groupby("player")}

        for _, r in roster.iterrows():
            player = str(r[player_col])
            team = r[team_col]

            df_p = grouped.get(player)
            if df_p is None or "sog" not in df_p.columns:
                continue

            sog_vals = df_p.groupby(game_col)["sog"].sum().tolist()
            if len(sog_vals) < 3:
                continue

            l3, l5, l10 = np.mean(sog_vals[-3:]), np.mean(sog_vals[-5:]), np.mean(sog_vals[-10:])
            baseline = 0.55*l10 + 0.3*l5 + 0.15*l3

            results.append({
                "Player": player,
                "Team": team,
                "Matchup": f"{team_a}@{team_b}",
                "Final Projection": round(baseline,2),
                "Season Avg": round(np.mean(sog_vals),2),
                "L3": ", ".join(map(str, sog_vals[-3:])),
                "L5": ", ".join(map(str, sog_vals[-5:])),
                "L10": ", ".join(map(str, sog_vals[-10:])),
            })

    st.session_state.base_results = pd.DataFrame(results)
    st.success("Model built successfully")

# ---------------------------------------------------------------
# DISPLAY (ORIGINAL)
# ---------------------------------------------------------------
if "base_results" in st.session_state:
    df = st.session_state.base_results

    if "selected_match" not in st.session_state:
        st.session_state.selected_match = f"{games[0]['away']}@{games[0]['home']}"

    st.markdown("## Matchups")
    cols = st.columns(3)
    for i, g in enumerate(games):
        with cols[i % 3]:
            if st.button(f"{g['away']} @ {g['home']}", use_container_width=True):
                st.session_state.selected_match = f"{g['away']}@{g['home']}"

    team_a, team_b = st.session_state.selected_match.split("@")
    tabs = st.tabs([team_a, team_b])

    def team_logo(team):
        for g in games:
            if g["away"] == team:
                return g["away_logo"]
            if g["home"] == team:
                return g["home_logo"]
        return ""

    def render(team, tab):
        with tab:
            tdf = (
                df[(df["Team"] == team) & (df["Matchup"] == st.session_state.selected_match)]
                .drop_duplicates("Player")
                .sort_values("Final Projection", ascending=False)
            )

            for _, r in tdf.iterrows():
                components.html(
                    f"""
                    <div style="background:#0F2743;border:1px solid #1E5A99;
                                border-radius:16px;padding:16px;margin-bottom:16px;color:#fff;">
                        <div style="display:flex;align-items:center;margin-bottom:6px;">
                            <img src="{team_logo(team)}" style="width:32px;height:32px;margin-right:10px;">
                            <b>{r['Player']} – {r['Team']}</b>
                        </div>
                        <div>Final Projection: <b>{r['Final Projection']}</b></div>
                        <div>Season Avg: {r['Season Avg']}</div>
                        <div>L3: {r['L3']}</div>
                        <div>L5: {r['L5']}</div>
                        <div>L10: {r['L10']}</div>
                    </div>
                    """,
                    height=300
                )

    render(team_a, tabs[0])
    render(team_b, tabs[1])
