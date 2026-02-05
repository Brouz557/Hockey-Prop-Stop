import streamlit as st
import pandas as pd
import os

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Shot Matchup Exporter", layout="wide")
st.title("🏒 Shot Matchup Exporter")
st.caption("Game-by-Game Shot Breakdown (3+ SOG Only)")

# --------------------------------------------------
# Safe repo loader
# --------------------------------------------------
def load_repo_file(filename):
    for path in [filename, f"./{filename}", f"data/{filename}"]:
        if os.path.exists(path):
            return pd.read_excel(path)
    st.error(f"❌ Could not find {filename} in repo.")
    st.stop()

# --------------------------------------------------
# Load SHOT DATA
# --------------------------------------------------
shots = load_repo_file("SHOT DATA.xlsx")

# --------------------------------------------------
# Load SKATERS (team lookup only)
# --------------------------------------------------
skaters = load_repo_file("Skaters.xlsx")

# --------------------------------------------------
# Normalize headers
# --------------------------------------------------
shots.columns = shots.columns.str.lower().str.strip()
skaters.columns = skaters.columns.str.lower().str.strip()

# --------------------------------------------------
# LOCKED column mapping (SHOT DATA)
# --------------------------------------------------
shots = shots.rename(columns={
    "name": "player",
    "position": "position",
    "gameid": "game_id",
    "sog": "sog",
    "opponent": "opponent"
})

required = ["player", "position", "game_id", "sog", "opponent"]
missing = [c for c in required if c not in shots.columns]
if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.write("Available columns:", list(shots.columns))
    st.stop()

# --------------------------------------------------
# Clean / cast SHOT DATA
# --------------------------------------------------
shots["player"] = shots["player"].astype(str).str.strip()
shots["position"] = shots["position"].astype(str).str.strip()
shots["opponent"] = shots["opponent"].astype(str).str.strip()
shots["game_id"] = shots["game_id"].astype(str)
shots["sog"] = pd.to_numeric(shots["sog"], errors="coerce").fillna(0)

# --------------------------------------------------
# 🔒 SAFE TEAM EXTRACTION FROM SKATERS
# --------------------------------------------------
player_col = "name" if "name" in skaters.columns else None
team_col = next((c for c in skaters.columns if "team" in c), None)

if not player_col or not team_col:
    st.error("❌ Could not identify player/team columns in Skaters.xlsx")
    st.write("Skaters columns:", list(skaters.columns))
    st.stop()

skaters = skaters.rename(columns={
    player_col: "player",
    team_col: "team"
})

skaters["player"] = skaters["player"].astype(str).str.strip()
skaters["team"] = skaters["team"].astype(str).str.strip()

player_team = skaters[["player", "team"]].drop_duplicates()

# --------------------------------------------------
# 🔗 MERGE TEAM INTO SHOTS (GUARANTEED SAFE)
# --------------------------------------------------
shots = shots.merge(
    player_team,
    on="player",
    how="left"
)

if "team" not in shots.columns:
    st.error("❌ Team column failed to merge into shots.")
    st.stop()

shots["team"] = shots["team"].fillna("UNK")

# --------------------------------------------------
# Controls
# --------------------------------------------------
last_n = st.slider(
    "Last N games per player",
    min_value=3,
    max_value=20,
    value=10
)

# --------------------------------------------------
# Sort by recency (GAMEID proxy)
# --------------------------------------------------
shots = shots.sort_values("game_id")

# --------------------------------------------------
# Keep last N games PER PLAYER
# --------------------------------------------------
shots["game_rank"] = shots.groupby("player").cumcount(ascending=False)
recent = shots[shots["game_rank"] < last_n].copy()

# ==================================================
# LEVEL 1: GAME × PLAYER × POSITION (+ TEAM)
# ==================================================
game_player_df = (
    recent
    .groupby(
        ["game_id", "opponent", "team", "player", "position"],
        as_index=False
    )
    .agg(
        shots_in_game=("sog", "sum")
    )
)

# -------------------------
# 🔒 FILTER: 3+ SHOTS ONLY
# -------------------------
game_player_df = game_player_df[game_player_df["shots_in_game"] >= 3]

game_player_df = game_player_df.sort_values(
    ["game_id", "shots_in_game"],
    ascending=[True, False]
)

# ==================================================
# LEVEL 2: GAME × OPPONENT × POSITION (+ TEAM)
# ==================================================
game_position_df = (
    game_player_df
    .groupby(
        ["game_id", "opponent", "team", "position"],
        as_index=False
    )
    .agg(
        total_shots_allowed=("shots_in_game", "sum")
    )
)

game_position_df = game_position_df.sort_values(
    ["game_id", "total_shots_allowed"],
    ascending=[True, False]
)

# --------------------------------------------------
# Display
# --------------------------------------------------
tab1, tab2 = st.tabs(
    ["🎯 Game → Player (3+ SOG)", "🔥 Game → Position (3+ SOG Players Only)"]
)

with tab1:
    st.subheader("Game-by-Game Player Shot Totals (≥ 3 SOG)")
    st.dataframe(game_player_df, use_container_width=True)

    st.download_button(
        "⬇️ Export Game-Player CSV",
        game_player_df.to_csv(index=False).encode("utf-8"),
        "game_player_shots_3plus.csv",
        "text/csv"
    )

with tab2:
    st.subheader("Game-by-Game Opponent Shot Allowance by Position (3+ SOG Only)")
    st.dataframe(game_position_df, use_container_width=True)

    st.download_button(
        "⬇️ Export Game-Position CSV",
        game_position_df.to_csv(index=False).encode("utf-8"),
        "game_position_shots_3plus.csv",
        "text/csv"
    )
