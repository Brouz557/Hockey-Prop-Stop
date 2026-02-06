# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Mobile (OPERABLE BASELINE)
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
        safe_read(find_file("LINE DATA.xlsx")),
    )

skaters_df, shots_df, lines_df = load_all()
if skaters_df.empty or shots_df.empty:
    st.error("Missing required data files.")
    st.stop()

for df in [skaters_df, shots_df, lines_df]:
    df.columns = df.columns.str.lower().str.strip()

# ---------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------
team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name" if "name" in skaters_df.columns else skaters_df.columns[0]
pos_col = next(c for c in skaters_df.columns if c in ["position","pos","primary position"])

shots_player_col = next(c for c in shots_df.columns if "player" in c or "name" in c)
shots_df = shots_df.rename(columns={shots_player_col: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.lower().str.strip()

game_col = next(c for c in shots_df.columns if "game" in c)
shots_df["opponent"] = shots_df["opponent"].astype(str).str.upper().str.strip()

# ---------------------------------------------------------------
# Team normalization
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "VEG": "VGK",
    "NJ": "NJD",
    "LA": "LAK",
    "SJ": "SJS",
    "TB": "TBL"
}
shots_df["opponent"] = shots_df["opponent"].replace(TEAM_ABBREV_MAP)

# ---------------------------------------------------------------
# Build player last names
# ---------------------------------------------------------------
skaters_df["player_last"] = (
    skaters_df[player_col]
    .astype(str)
    .str.lower()
    .str.split()
    .str[-1]
)

# ---------------------------------------------------------------
# Build typical line roles (SAFE VERSION)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_player_line_roles(lines_df):
    if lines_df.empty:
        return pd.DataFrame()

    req = {"line pairings", "team", "games"}
    if not req.issubset(lines_df.columns):
        return pd.DataFrame()

    l = lines_df.copy()
    l["games"] = pd.to_numeric(l["games"], errors="coerce").fillna(0)

    l = (
        l.groupby(["team", "line pairings"], as_index=False)
        .agg({"games": "sum"})
        .rename(columns={"games": "line_usage"})
    )

    l["line_rank"] = (
        l.groupby("team")["line_usage"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    rows = []
    for _, r in l.iterrows():
        for p in str(r["line pairings"]).lower().split():
            rows.append({
                "team": r["team"],
                "player_last": p,
                "line_rank": r["line_rank"],
                "line_usage": r["line_usage"]
            })

    pl = pd.DataFrame(rows)
    if pl.empty:
        return pd.DataFrame()

    # keep dominant line only
    pl = (
        pl.sort_values("line_usage", ascending=False)
        .drop_duplicates(["team", "player_last"])
    )

    return pl

player_line_roles = build_player_line_roles(lines_df)

# ---------------------------------------------------------------
# Build opponent × position × line profile (STRICT)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_opponent_role_profile(shots_df, skaters_df, player_line_roles):
    pg = (
        shots_df
        .groupby(["player", game_col, "opponent"], as_index=False)["sog"]
        .sum()
    )

    pg = pg[pg["sog"] >= 3]

    pg = pg.merge(
        skaters_df[[player_col, team_col, pos_col, "player_last"]],
        left_on="player",
        right_on=player_col,
        how="left"
    )

    pg["position"] = pg[pos_col].replace({
        "LW": "L", "RW": "R", "LD": "D", "RD": "D"
    })

    pg = pg.merge(
        player_line_roles,
        on=["team", "player_last"],
        how="left"
    )

    profiles = {}

    for opp in pg["opponent"].dropna().unique():
        vs = pg[pg["opponent"] == opp]

        last_games = (
            vs[[game_col]]
            .drop_duplicates()
            .tail(20)[game_col]
            .tolist()
        )

        recent = vs[vs[game_col].isin(last_games)]

        summary = (
            recent
            .groupby(["position", "line_rank"])["player"]
            .nunique()
            .to_dict()
        )

        profiles[opp] = summary

    return profiles

opponent_role_profiles = build_opponent_role_profile(
    shots_df, skaters_df, player_line_roles
)

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

def team_logo(team):
    for g in games:
        if g["away"] == team:
            return g["away_logo"]
        if g["home"] == team:
            return g["home_logo"]
    return ""

# ---------------------------------------------------------------
# DISPLAY (CONTEXT ONLY)
# ---------------------------------------------------------------
st.markdown("## Matchups")
cols = st.columns(3)

if "selected_match" not in st.session_state:
    st.session_state.selected_match = f"{games[0]['away']}@{games[0]['home']}"

for i, g in enumerate(games):
    with cols[i % 3]:
        if st.button(f"{g['away']} @ {g['home']}", use_container_width=True):
            st.session_state.selected_match = f"{g['away']}@{g['home']}"

team_a, team_b = st.session_state.selected_match.split("@")
tabs = st.tabs([team_a, team_b])

def render(team, tab):
    with tab:
        opp = team_b if team == team_a else team_a
        prof = opponent_role_profiles.get(opp, {})

        def rp(pos, line):
            return prof.get((pos, line), 0)

        components.html(
            f"""
            <div style="background:#0F2743;border:1px solid #1E5A99;
                        border-radius:16px;padding:16px;color:#fff;">
                <b>VS {opp}</b><br>
                D1 – {rp("D",1)} &nbsp; D2 – {rp("D",2)}<br>
                R1 – {rp("R",1)} &nbsp; R2 – {rp("R",2)}<br>
                L1 – {rp("L",1)} &nbsp; L2 – {rp("L",2)}<br>
                C1 – {rp("C",1)} &nbsp; C2 – {rp("C",2)}
            </div>
            """,
            height=220
        )

render(team_a, tabs[0])
render(team_b, tabs[1])
