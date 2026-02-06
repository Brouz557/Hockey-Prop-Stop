# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Mobile (POSITION × LINE)
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
# Normalization maps
# ---------------------------------------------------------------
POSITION_MAP = {"LW": "L", "RW": "R", "C": "C", "D": "D"}
TEAM_ABBREV_MAP = {"LA": "LAK", "TB": "TBL", "NY": "NYR", "NJ": "NJD", "SJ": "SJS"}

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

shots_df["opponent"] = (
    shots_df["opponent"]
    .astype(str)
    .str.upper()
    .replace(TEAM_ABBREV_MAP)
)

# ---------------------------------------------------------------
# Player last names
# ---------------------------------------------------------------
skaters_df["player_last"] = (
    skaters_df[player_col]
    .astype(str)
    .str.lower()
    .str.split()
    .str[-1]
)

# ---------------------------------------------------------------
# Build typical line numbers from LINE DATA
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_player_line_map(lines_df):
    l = lines_df.copy()
    l["games"] = pd.to_numeric(l["games"], errors="coerce").fillna(0)

    l = (
        l.groupby(["team","line pairings"], as_index=False)
        .agg({"games":"sum"})
        .rename(columns={"games":"usage"})
    )

    l["line"] = (
        l.groupby("team")["usage"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    rows = []
    for _, r in l.iterrows():
        for name in str(r["line pairings"]).lower().split():
            rows.append({
                "team": r["team"],
                "player_last": name,
                "line": r["line"]
            })

    return (
        pd.DataFrame(rows)
        .drop_duplicates(["team","player_last"])
    )

player_line_map = build_player_line_map(lines_df)

# ---------------------------------------------------------------
# Build opponent POSITION × LINE profile
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_opponent_role_profile(shots_df, skaters_df, player_line_map):

    # sum SOG per player-game
    pg = (
        shots_df
        .groupby(["player", game_col, "opponent"], as_index=False)["sog"]
        .sum()
    )
    pg["hit_3p"] = pg["sog"] >= 3

    # add skater info
    sk = skaters_df[[player_col, team_col, pos_col, "player_last"]].copy()
    sk.columns = ["player", "team", "position", "player_last"]
    sk["player"] = sk["player"].astype(str).str.lower().str.strip()
    sk["position"] = (
        sk["position"]
        .astype(str)
        .str.upper()
        .map(POSITION_MAP)
    )

    pg = pg.merge(sk, on="player", how="left")
    pg = pg.merge(player_line_map, on=["team","player_last"], how="left")

    pg["line"] = pg["line"].fillna(0).astype(int)
    pg = pg[pg["position"].isin(["C","L","R","D"])]

    profiles = {}

    for opp in pg["opponent"].unique():
        last_games = (
            pg[pg["opponent"] == opp]
            [[game_col]]
            .drop_duplicates()
            .sort_values(game_col)
            .tail(20)[game_col]
            .tolist()
        )

        sub = pg[
            (pg["opponent"] == opp) &
            (pg[game_col].isin(last_games)) &
            (pg["hit_3p"])
        ]

        profiles[opp] = (
            sub.groupby(["position","line"])["player"]
            .nunique()
            .to_dict()
        )

    return profiles

opponent_role_profiles = build_opponent_role_profile(
    shots_df, skaters_df, player_line_map
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
    st.stop()

def team_logo(team):
    for g in games:
        if g["away"] == team:
            return g["away_logo"]
        if g["home"] == team:
            return g["home_logo"]
    return ""

# ---------------------------------------------------------------
# Run model (unchanged)
# ---------------------------------------------------------------
if st.button("Run Model (All Games)", use_container_width=True):
    results = []
    grouped = {n:d for n,d in shots_df.groupby("player")}

    for g in games:
        team_a, team_b = g["away"], g["home"]
        roster = skaters_df[skaters_df[team_col].isin([team_a, team_b])]

        for _, r in roster.iterrows():
            player = str(r[player_col])
            team = r[team_col]
            pos = POSITION_MAP.get(str(r[pos_col]).upper(), "")

            df_p = grouped.get(player.lower())
            if df_p is None:
                continue

            sog_vals = df_p.groupby(game_col)["sog"].sum().tolist()
            if len(sog_vals) < 3:
                continue

            l3, l5, l10 = np.mean(sog_vals[-3:]), np.mean(sog_vals[-5:]), np.mean(sog_vals[-10:])
            lam = 0.55*l10 + 0.3*l5 + 0.15*l3

            results.append({
                "Player": player,
                "Position": pos,
                "Team": team,
                "Matchup": f"{team_a}@{team_b}",
                "Final Projection": round(lam,2),
                "Season Avg": round(np.mean(sog_vals),2),
                "L3": ", ".join(map(str,sog_vals[-3:])),
                "L5": ", ".join(map(str,sog_vals[-5:])),
                "L10": ", ".join(map(str,sog_vals[-10:])),
            })

    st.session_state.base_results = pd.DataFrame(results)
    st.success("Model built")

# ---------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------
if "base_results" in st.session_state:
    df = st.session_state.base_results
    st.session_state.setdefault("selected_match", f"{games[0]['away']}@{games[0]['home']}")

    cols = st.columns(3)
    for i,g in enumerate(games):
        with cols[i%3]:
            if st.button(f"{g['away']} @ {g['home']}", use_container_width=True):
                st.session_state.selected_match = f"{g['away']}@{g['home']}"

    team_a, team_b = st.session_state.selected_match.split("@")
    tabs = st.tabs([team_a, team_b])

    def render(team, tab):
        with tab:
            tdf = (
                df[(df["Team"] == team) & (df["Matchup"] == st.session_state.selected_match)]
                .drop_duplicates("Player")
                .sort_values("Final Projection", ascending=False)
            )

            for _, r in tdf.iterrows():
                opp = team_b if team == team_a else team_a
                prof = opponent_role_profiles.get(opp, {})

                def render_rows():
                    rows = []
                    for (p,l),v in sorted(prof.items()):
                        rows.append(f"{p}{l}: {v}")
                    return "<br>".join(rows) if rows else "—"

                components.html(
                    f"""
                    <div style="background:#0F2743;border:1px solid #1E5A99;
                                border-radius:16px;padding:16px;margin-bottom:16px;
                                color:#fff;display:flex;justify-content:space-between;">
                        <div style="width:65%;">
                            <div style="display:flex;align-items:center;margin-bottom:6px;">
                                <img src="{team_logo(team)}"
                                     style="width:32px;height:32px;margin-right:10px;">
                                <b>{r['Player']} ({r['Position']}) – {r['Team']}</b>
                            </div>
                            Final Projection: <b>{r['Final Projection']}</b><br>
                            Season Avg: {r['Season Avg']}<br>
                            L3: {r['L3']}<br>
                            L5: {r['L5']}<br>
                            L10: {r['L10']}
                        </div>
                        <div style="width:30%;border-left:1px solid #1E5A99;padding-left:10px;">
                            <b>VS {opp}</b><br>
                            {render_rows()}
                        </div>
                    </div>
                    """,
                    height=380
                )

    render(team_a, tabs[0])
    render(team_b, tabs[1])
