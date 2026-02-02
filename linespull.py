import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Shot Matchup Exporter", layout="wide")

st.title("🏒 Shot Matchup Exporter (Repo Safe)")

# -------------------------
# Safe file loading
# -------------------------
def load_repo_file(filename):
    for path in [filename, f"data/{filename}", f"./{filename}"]:
        if os.path.exists(path):
            return pd.read_excel(path)
    st.error(f"❌ File not found: {filename}")
    st.stop()

shots = load_repo_file("SHOT DATA.xlsx")
skaters = load_repo_file("Skaters.xlsx")

# -------------------------
# Normalize columns
# -------------------------
shots.columns = shots.columns.str.lower().str.strip()
skaters.columns = skaters.columns.str.lower().str.strip()

shots["player"] = shots["player"].astype(str).str.strip()
skaters["name"] = skaters["name"].astype(str).str.strip()

# -------------------------
# Controls
# -------------------------
last_n = st.slider("Number of recent games per player", 3, 20, 10)

# -------------------------
# Attach positions
# -------------------------
shots = shots.merge(
    skaters[["name", "position"]],
    left_on="player",
    right_on="name",
    how="left"
).drop(columns="name")

# -------------------------
# Sort by recency
# -------------------------
if "date" in shots.columns:
    shots["date"] = pd.to_datetime(shots["date"])
    shots = shots.sort_values("date")
else:
    shots = shots.sort_values("game_id")

# -------------------------
# Keep last N games per player
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
        shots_game_ids=("game_id", lambda x: ",".join(map(str, sorted(set(x))))),
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

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Export CSV",
    csv,
    "shot_matchups_last_games.csv",
    "text/csv"
)
