# ---------------------------------------------------------------
# 🏒 Puck Shotz — Opponent Line × Position Engine (FINAL)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Puck Shotz — Opponent Pressure",
    layout="wide",
    page_icon="🏒"
)

st.title("Opponent Line × Position SOG Pressure")

# ---------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------
POSITION_MAP = {
    "LW": "L",
    "RW": "R",
    "C": "C",
    "D": "D"
}

TEAM_ABBREV_MAP = {
    "LA": "LAK",
    "TB": "TBL",
    "NY": "NYR",
    "NJ": "NJD",
    "SJ": "SJS"
}

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def safe_read(path):
    try:
        return pd.read_excel(path) if path and os.path.exists(path) else pd.DataFrame()
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
skaters_df = safe_read(find_file("Skaters.xlsx"))
shots_df   = safe_read(find_file("SHOT DATA.xlsx"))
lines_df   = safe_read(find_file("LINE DATA.xlsx"))

if skaters_df.empty or shots_df.empty or lines_df.empty:
    st.error("Missing required data files.")
    st.stop()

for df in [skaters_df, shots_df, lines_df]:
    df.columns = df.columns.str.lower().str.strip()

# ---------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------
player_col = "name" if "name" in skaters_df.columns else skaters_df.columns[0]
team_col   = next(c for c in skaters_df.columns if "team" in c)
pos_col    = next(c for c in skaters_df.columns if c in ["position","pos","primary position"])
game_col   = next(c for c in shots_df.columns if "game" in c)

shots_player_col = next(c for c in shots_df.columns if "player" in c or "name" in c)
shots_df = shots_df.rename(columns={shots_player_col: "player"})

# ---------------------------------------------------------------
# Normalize SHOT DATA
# ---------------------------------------------------------------
shots_df["player"] = shots_df["player"].str.lower().str.strip()
shots_df["opponent"] = (
    shots_df["opponent"]
    .astype(str)
    .str.upper()
    .replace(TEAM_ABBREV_MAP)
)

# ---------------------------------------------------------------
# Normalize SKATERS
# ---------------------------------------------------------------
skaters_df["player"] = skaters_df[player_col].str.lower().str.strip()
skaters_df["position"] = (
    skaters_df[pos_col]
    .astype(str)
    .str.upper()
    .map(POSITION_MAP)
)
skaters_df["player_last"] = skaters_df[player_col].str.lower().str.split().str[-1]

# ---------------------------------------------------------------
# Build typical line map (ICE TIME proxy via games)
# ---------------------------------------------------------------
lines_df["games"] = pd.to_numeric(lines_df["games"], errors="coerce").fillna(0)

line_usage = (
    lines_df
    .groupby(["team", "line pairings"], as_index=False)
    .agg({"games": "sum"})
)

line_usage["line"] = (
    line_usage
    .groupby("team")["games"]
    .rank(method="first", ascending=False)
    .astype(int)
)

rows = []
for _, r in line_usage.iterrows():
    for name in str(r["line pairings"]).lower().split():
        rows.append({
            "team": r["team"],
            "player_last": name,
            "line": r["line"]
        })

player_line_map = (
    pd.DataFrame(rows)
    .drop_duplicates(["team", "player_last"])
)

# ---------------------------------------------------------------
# Build PLAYER–GAME SOG vs OPPONENT
# ---------------------------------------------------------------
player_game = (
    shots_df
    .groupby(["player", game_col, "opponent"], as_index=False)["sog"]
    .sum()
)

player_game["hit_3p"] = player_game["sog"] >= 3

# ---------------------------------------------------------------
# Enrich with ROLE
# ---------------------------------------------------------------
player_game = (
    player_game
    .merge(
        skaters_df[["player", team_col, "position", "player_last"]],
        on="player",
        how="left"
    )
    .merge(
        player_line_map,
        on=["team", "player_last"],
        how="left"
    )
)

# Drop unresolved roles
player_game = player_game.dropna(subset=["position", "line"])
player_game["line"] = player_game["line"].astype(int)

# ---------------------------------------------------------------
# Build opponent profile (LAST 10 GAMES)
# ---------------------------------------------------------------
profiles = {}

for opp in sorted(player_game["opponent"].unique()):

    last_games = (
        player_game[player_game["opponent"] == opp]
        [[game_col]]
        .drop_duplicates()
        .sort_values(game_col)
        .tail(10)[game_col]
        .tolist()
    )

    sub = player_game[
        (player_game["opponent"] == opp) &
        (player_game[game_col].isin(last_games)) &
        (player_game["hit_3p"])
    ]

    profiles[opp] = (
        sub
        .groupby(["line", "position"])["player"]
        .nunique()
        .to_dict()
    )

# ---------------------------------------------------------------
# UI — VALIDATION PANEL
# ---------------------------------------------------------------
opp = st.selectbox("Select Opponent", sorted(profiles.keys()))
prof = profiles.get(opp, {})

def render_panel():
    rows = []
    for p in ["C","L","R","D"]:
        max_line = 3 if p == "D" else 4
        for l in range(1, max_line + 1):
            rows.append(f"L{l}{p}: {prof.get((l,p), 0)}")
    return "<br>".join(rows)

st.markdown(f"### VS {opp}")

st.markdown(render_panel(), unsafe_allow_html=True)
