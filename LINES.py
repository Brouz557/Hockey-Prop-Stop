import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta, datetime

# ----------------------------------------------------
# App Config
# ----------------------------------------------------
st.set_page_config(
    page_title="NST Current Lines Builder",
    page_icon="🏒",
    layout="wide"
)

st.title("🏒 Natural Stat Trick – Current Lines Builder")
st.caption(
    "Determines current lines using each team’s most recent completed game "
    "(Natural Stat Trick game CSVs)"
)

# ----------------------------------------------------
# Constants
# ----------------------------------------------------
SEASON = "20252026"
OUTPUT_FILE = "CURRENT_LINES.xlsx"

# ----------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------
st.sidebar.header("⚙️ Settings")

days_back = st.sidebar.selectbox(
    "How many days back to search for games",
    [1, 2, 3, 4, 5, 7],
    index=2,
    help="Fallback window if some teams haven’t played recently"
)

run_button = st.sidebar.button("🚀 Build Current Lines")

# ----------------------------------------------------
# Helper: Discover recent completed games (ESPN scoreboard)
# ----------------------------------------------------
@st.cache_data(ttl=300)
def get_recent_completed_games(days_back: int):
    games = []

    for i in range(days_back):
        d = date.today() - timedelta(days=i + 1)
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/"
            f"hockey/nhl/scoreboard?dates={d:%Y%m%d}"
        )

        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            continue

        data = r.json()
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {})

            if not status.get("completed"):
                continue

            teams = [
                c["team"]["abbreviation"]
                for c in comp.get("competitors", [])
            ]

            games.append({
                "game_id": event["id"],
                "date": d,
                "teams": teams
            })

    return games


def map_latest_game_per_team(games):
    latest = {}

    games_sorted = sorted(
        games,
        key=lambda x: x["date"],
        reverse=True
    )

    for g in games_sorted:
        for team in g["teams"]:
            if team not in latest:
                latest[team] = g["game_id"]

    return latest

# ----------------------------------------------------
# Helper: Load NST Game CSV
# ----------------------------------------------------
def load_nst_game_csv(game_id: str):
    url = (
        "https://www.naturalstattrick.com/game.php"
        f"?season={SEASON}"
        f"&game={game_id}"
        "&view=limited"
        "&csv=y"
    )
    return pd.read_csv(url)

# ----------------------------------------------------
# Main Pipeline
# ----------------------------------------------------
if run_button:
    st.info("🔍 Finding most recent completed games…")

    games = get_recent_completed_games(days_back)

    if not games:
        st.error("No completed games found in selected window.")
        st.stop()

    team_to_game = map_latest_game_per_team(games)

    st.success(f"Found recent games for {len(team_to_game)} teams")

    rows = []

    progress = st.progress(0)
    total = len(team_to_game)

    for i, (team, game_id) in enumerate(team_to_game.items(), start=1):
        progress.progress(i / total)

        try:
            df = load_nst_game_csv(game_id)
        except Exception:
            continue

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        if "team" not in df.columns:
            continue

        # ------------------------------------------------
        # 🔑 CRITICAL FIX: detect NST line columns
        # NST uses f1/f2/f3 and d1/d2 (not "player1")
        # ------------------------------------------------
        player_cols = [
            c for c in df.columns
            if c.startswith(("player", "f", "d"))
            and df[c].dtype == object
        ]

        if not player_cols:
            continue

        def build_line(row):
            names = []
            for c in player_cols:
                val = row[c]
                if pd.notna(val) and isinstance(val, str):
                    names.append(val.split()[-1].lower())
            return " ".join(names)

        df["line pairings"] = df.apply(build_line, axis=1)

        team_df = df[df["team"] == team]

        for _, r in team_df.iterrows():
            rows.append({
                "team": team,
                "line pairings": r["line pairings"],
                "toi": r.get("toi", 0),
                "source_game": game_id
            })

    progress.empty()

    if not rows:
        st.error("No line data extracted from NST.")
        st.stop()

    raw = pd.DataFrame(rows)

    # ------------------------------------------------
    # Aggregate final CURRENT_LINES table
    # ------------------------------------------------
    out = (
        raw
        .groupby(["team", "line pairings"], as_index=False)
        .agg(
            games=("source_game", "nunique"),
            toi=("toi", "sum")
        )
        .sort_values(["team", "toi"], ascending=[True, False])
    )

    out.to_excel(OUTPUT_FILE, index=False)

    # ------------------------------------------------
    # Display Results
    # ------------------------------------------------
    st.success(f"✅ {OUTPUT_FILE} created successfully")
    st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M}")

    st.subheader("📊 Current Lines (Most Recent Game Usage)")
    st.dataframe(out, use_container_width=True)

    st.subheader("🔎 Debug: Team → Source Game")
    debug = (
        raw[["team", "source_game"]]
        .drop_duplicates()
        .sort_values("team")
    )
    st.dataframe(debug, use_container_width=True)

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.caption(
    "Current lines based on actual on-ice usage · "
    "Natural Stat Trick game CSVs · deterministic & reproducible"
)
