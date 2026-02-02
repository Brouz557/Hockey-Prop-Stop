import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Shot Matchup Exporter", layout="wide")
st.title("🏒 Shot Matchup Exporter")

# -------------------------
# Safe repo loader
# -------------------------
def load_repo_file(filename):
    for path in [filename, f"data/{filename}", f"./{filename}"]:
        if os.path.exists(path):
            return pd.read_excel(path)
    st.error(f"❌ File not found: {filename}")
    st.stop()

shots = load_repo_file("SHOT DATA.xlsx")

# -------------------------
# Normalize columns
# -------------------------
shots.columns = shots.columns.str.lower().str.strip()

# -------------------------
# REQUIRED COLUMN MAPPING (LOCKED)
# -------------------------
shots = shots.rename(columns={
    "name": "player",
    "position": "position",
    "gameid": "game_id",
    "sog": "sog",
    "opponent": "opponent",
    "team": "team"
})

required = ["player", "position", "game_id", "sog", "opponent"]
missing = [c for c in required if c not in shots.columns]
if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.write("Available columns:", list(shots.columns))
    st.stop()

# -------------------------
# Clean types
# -------------------------
shots["player"] = shots["player"].astype(str).str.strip()
shots["position"] = shots["position"].astype(str).str.strip()
shots["opponent"] = shots["opponent"].astype(str).str.strip()
shots["game_id"] = shots["game_id"].astype(str)

shots["sog"] = pd.to_numeric(shots["sog"], errors="coerce").fillna(0)

# -------------------------
# Controls
# -------------------------
last_n = st.slider("Last N games per player", 3, 20, 10)

# -------------------------
# Sort by recency (GAMEID works)
# -------------------------
shots = shots.sort_values("game_id")

# -------------------------
# Keep last N games PER PLAYER
# -------------------------
shots["game_rank"] = shots.groupby("player").cumcount(ascending=False)
recent = shots[shots["game_rank"] < last_n].copy()

# -------------------------
# Build export table
# -------------------------
export_df = (
    recent
    .groupby(["opponent", "player", "position"], as_index=False)
    .agg(
        shots_game_ids=("game_id", lambda x: ",".join(sorted(set(x)))),
        total_shots=("sog", "sum"),
        games=("game_id", "nunique")
    )
)

export_df["shots_per_game"] = (
    export_df["total_shots"] / export_df["games"]
).round(2)

export_df = export_df[
    ["opponent", "player", "position", "shots_game_ids", "total_shots", "shots_per_game"]
].sort_values(
    ["total_shots", "shots_per_game"],
    ascending=False
)

# -------------------------
# Display + Export
# -------------------------
st.subheader("📊 Shot Matchups (Last Games)")
st.dataframe(export_df, use_container_width=True)

st.download_button(
    "⬇️ Export CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    "shot_matchups_last_games.csv",
    "text/csv"
)
