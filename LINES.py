import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta, datetime

# ====================================================
# STREAMLIT CONFIG
# ====================================================
st.set_page_config(
    page_title="NST Current Lines Builder",
    page_icon="🏒",
    layout="wide"
)

st.title("🏒 Natural Stat Trick – Current Lines Builder")
st.caption(
    "Builds current forward lines and defense pairs using each team’s "
    "most recent completed game (NST Forward Lines + Defense Lines reports)."
)

# ====================================================
# CONSTANTS
# ====================================================
SEASON = "20252026"
OUTPUT_FILE = "CURRENT_LINES.xlsx"
STRENGTH = "5v5"

# ====================================================
# SIDEBAR CONTROLS
# ====================================================
st.sidebar.header("⚙️ Controls")

days_back = st.sidebar.selectbox(
    "Search window (days back)",
    [1, 2, 3, 4, 5, 7],
    index=2,
    help="How far back to look for each team’s most recent game"
)

run_button = st.sidebar.button("🚀 Build Current Lines")

# ====================================================
# GAME DISCOVERY (ESPN SCOREBOARD)
# ====================================================
@st.cache_data(ttl=300)
def get_recent_completed_games(days_back: int):
    games = []

    for i in range(days_back):
        game_date = date.today() - timedelta(days=i + 1)
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/"
            f"hockey/nhl/scoreboard?dates={game_date:%Y%m%d}"
        )

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except Exception:
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
                "date": game_date,
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

# ====================================================
# NST CSV LOADERS (CORRECT REPORTS)
# ====================================================
def load_forward_lines(game_id: str):
    url = (
        "https://www.naturalstattrick.com/game.php"
        f"?season={SEASON}"
        f"&game={game_id}"
        "&report=forwardlines"
        f"&strength={STRENGTH}"
        "&csv=y"
    )
    return pd.read_csv(url)


def load_defense_lines(game_id: str):
    url = (
        "https://www.naturalstattrick.com/game.php"
        f"?season={SEASON}"
        f"&game={game_id}"
        "&report=defenselines"
        f"&strength={STRENGTH}"
        "&csv=y"
    )
    return pd.read_csv(url)

# ====================================================
# MAIN PIPELINE
# ====================================================
if run_button:
    st.info("🔍 Discovering most recent completed games…")

    games = get_recent_completed_games(days_back)

    if not games:
        st.error("No completed games found.")
        st.stop()

    team_to_game = map_latest_game_per_team(games)
    st.success(f"Found recent games for {len(team_to_game)} teams")

    rows = []
    progress = st.progress(0)
    total = len(team_to_game)

    for idx, (team, game_id) in enumerate(team_to_game.items(), start=1):
        progress.progress(idx / total)

        # -----------------------------
        # FORWARD LINES
        # -----------------------------
        try:
            fwd = load_forward_lines(game_id)
            fwd.columns = fwd.columns.str.lower().str.strip()
        except Exception:
            fwd = pd.DataFrame()

        if not fwd.empty and {"team", "f1", "f2", "f3"}.issubset(fwd.columns):
            fwd_team = fwd[fwd["team"] == team]

            for _, r in fwd_team.iterrows():
                line = " ".join([
                    r["f1"].split()[-1].lower(),
                    r["f2"].split()[-1].lower(),
                    r["f3"].split()[-1].lower()
                ])

                rows.append({
                    "team": team,
                    "unit_type": "F",
                    "line_pairings": line,
                    "toi": r.get("toi", 0),
                    "source_game": game_id
                })

        # -----------------------------
        # DEFENSE PAIRS
        # -----------------------------
        try:
            dfd = load_defense_lines(game_id)
            dfd.columns = dfd.columns.str.lower().str.strip()
        except Exception:
            dfd = pd.DataFrame()

        if not dfd.empty and {"team", "d1", "d2"}.issubset(dfd.columns):
            dfd_team = dfd[dfd["team"] == team]

            for _, r in dfd_team.iterrows():
                pair = " ".join([
                    r["d1"].split()[-1].lower(),
                    r["d2"].split()[-1].lower()
                ])

                rows.append({
                    "team": team,
                    "unit_type": "D",
                    "line_pairings": pair,
                    "toi": r.get("toi", 0),
                    "source_game": game_id
                })

    progress.empty()

    if not rows:
        st.error("No line data extracted from NST.")
        st.stop()

    raw = pd.DataFrame(rows)

    # ====================================================
    # AGGREGATE FINAL CURRENT_LINES TABLE
    # ====================================================
    current_lines = (
        raw
        .groupby(["team", "unit_type", "line_pairings"], as_index=False)
        .agg(
            games=("source_game", "nunique"),
            toi=("toi", "sum")
        )
        .sort_values(["team", "unit_type", "toi"], ascending=[True, True, False])
    )

    current_lines.to_excel(OUTPUT_FILE, index=False)

    # ====================================================
    # DISPLAY OUTPUT
    # ====================================================
    st.success(f"✅ {OUTPUT_FILE} created successfully")
    st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M}")

    st.subheader("📊 Current Lines (Most Recent Game Usage)")
    st.dataframe(current_lines, use_container_width=True)

    st.subheader("🔎 Debug: Team → Source Game")
    debug = (
        raw[["team", "source_game"]]
        .drop_duplicates()
        .sort_values("team")
    )
    st.dataframe(debug, use_container_width=True)

# ====================================================
# FOOTER
# ====================================================
st.markdown("---")
st.caption(
    "Current lines based on actual on-ice usage · "
    "Natural Stat Trick Forward Lines + Defense Lines reports · deterministic"
)
