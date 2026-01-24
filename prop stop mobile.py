# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Mobile Cards (BASELINE)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests
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
# Header (WORKING VERSION)
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
# Team Abbreviation Normalization
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ": "NJD",
    "LA": "LAK",
    "SJ": "SJS",
    "TB": "TBL"
}

# ---------------------------------------------------------------
# Auto-load data
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

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name" if "name" in skaters_df.columns else skaters_df.columns[0]

shots_player_col = next(c for c in shots_df.columns if "player" in c or "name" in c)
shots_df = shots_df.rename(columns={shots_player_col: "player"})
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
# Run model button
# ---------------------------------------------------------------
if st.button("🚀 Run Model (All Games)", use_container_width=True):
    st.session_state.run_model = True

# ---------------------------------------------------------------
# Build model (UNCHANGED)
# ---------------------------------------------------------------
def build_model(team_a, team_b):
    results = []
    roster = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
    grouped = {n.lower(): g for n, g in shots_df.groupby(shots_df["player"].str.lower())}

    for _, r in roster.iterrows():
        player = str(r[player_col])
        team = r[team_col]
        df_p = grouped.get(player.lower())

        if df_p is None or "sog" not in df_p.columns:
            continue

        sog_vals = df_p.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals) < 3:
            continue

        l3, l5, l10 = np.mean(sog_vals[-3:]), np.mean(sog_vals[-5:]), np.mean(sog_vals[-10:])
        lam = 0.55*l10 + 0.3*l5 + 0.15*l3

        results.append({
            "Player": player,
            "Team": team,
            "Final Projection": round(lam, 2),
            "L3": ", ".join(map(str, sog_vals[-3:])),
            "L5": ", ".join(map(str, sog_vals[-5:])),
            "L10": ", ".join(map(str, sog_vals[-10:])),
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run model
# ---------------------------------------------------------------
if st.session_state.get("run_model"):
    tables = []
    for g in games:
        df = build_model(g["away"], g["home"])
        if not df.empty:
            df["Matchup"] = f"{g['away']}@{g['home']}"
            tables.append(df)

    if tables:
        st.session_state.results = pd.concat(tables, ignore_index=True)

# ---------------------------------------------------------------
# DISPLAY — MATCHUP BUTTONS + EXTENDED CARD VIEW
# ---------------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state.results

    st.markdown("## Matchups")

    cols = st.columns(3)
    for i, g in enumerate(games):
        matchup_id = f"{g['away']}@{g['home']}"
        with cols[i % 3]:
            if st.button(f"{g['away']} @ {g['home']}", use_container_width=True):
                st.session_state.selected_match = matchup_id

            st.markdown(
                f"""
                <div style="display:flex;justify-content:center;gap:8px;margin-top:4px;">
                    <img src="{g['away_logo']}" height="22">
                    <img src="{g['home_logo']}" height="22">
                </div>
                """,
                unsafe_allow_html=True
            )

    if "selected_match" not in st.session_state:
        st.stop()

    team_a, team_b = st.session_state.selected_match.split("@")

    # Selected matchup logos (WORKING)
    sel = next(g for g in games if g["away"] == team_a and g["home"] == team_b)
    components.html(
        f"""
        <div style="display:flex;justify-content:center;align-items:center;
                    gap:14px;margin:14px 0 20px 0;font-size:18px;
                    font-weight:700;color:#FFFFFF;">
            <img src="{sel['away_logo']}" height="32">
            <span>{team_a} @ {team_b}</span>
            <img src="{sel['home_logo']}" height="32">
        </div>
        """,
        height=70
    )

    tabs = st.tabs([team_a, team_b])

    def render_team(team, tab):
        with tab:
            team_df = df[
                (df["Team"] == team) &
                (df["Matchup"] == st.session_state.selected_match)
            ].drop_duplicates(subset=["Player"])

            for _, r in team_df.iterrows():
                components.html(
                    f"""
                    <div style="
                        background:#0F2743;
                        border:1px solid #1E5A99;
                        border-radius:16px;
                        padding:16px;
                        margin-bottom:16px;
                        color:#FFFFFF;
                    ">
                        <div style="font-size:16px;font-weight:700;">
                            {r.get('Player','')} – {r.get('Team','')}
                        </div>
                        <div>Final Projection: <b>{r.get('Final Projection','')}</b></div>
                        <div>L3: {r.get('L3','')}</div>
                        <div>L5: {r.get('L5','')}</div>
                        <div>L10: {r.get('L10','')}</div>
                    </div>
                    """,
                    height=220
                )

    render_team(team_a, tabs[0])
    render_team(team_b, tabs[1])
