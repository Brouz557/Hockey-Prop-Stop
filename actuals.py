import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# --------------------------------------------------
# DATE (YESTERDAY, ESPN LEAGUE DAY - EASTERN TIME)
# --------------------------------------------------
ET = pytz.timezone("US/Eastern")
DATE = (datetime.now(ET) - timedelta(days=1)).strftime("%Y%m%d")

SCOREBOARD_URL = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={DATE}"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

st.set_page_config(page_title="NHL Actual SOG (All Games)", layout="wide")
st.title("🏒 NHL Actual Shots on Goal — All Games (Yesterday)")
st.caption(f"Brute-force pull for {DATE}")

# --------------------------------------------------
# GET ALL GAME IDS
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def get_game_ids():
    r = requests.get(SCOREBOARD_URL, timeout=10)
    events = r.json().get("events", [])
    return [e["id"] for e in events]

# --------------------------------------------------
# PULL SOG FOR ONE GAME
# --------------------------------------------------
def pull_game_sog(game_id):
    r = requests.get(SUMMARY_URL.format(game_id), timeout=10)
    data = r.json()
    rows = []

    for team in data.get("boxscore", {}).get("players", []):
        team_abbr = team.get("team", {}).get("abbreviation")

        for group in team.get("statistics", []):
            if group.get("name") == "goalies":
                continue

            for athlete in group.get("athletes", []):
                player = athlete.get("athlete", {}).get("displayName")
                for stat in athlete.get("stats", []):
                    if isinstance(stat, dict) and stat.get("name") == "shotsOnGoal":
                        rows.append({
                            "date": DATE,
                            "game_id": game_id,
                            "team": team_abbr,
                            "player": player,
                            "actual_sog": stat.get("value", 0)
                        })
                        break

    return rows

# --------------------------------------------------
# RUN
# --------------------------------------------------
if st.button("📥 Pull All Games from Yesterday"):
    with st.spinner("Pulling ESPN boxscores..."):
        game_ids = get_game_ids()
        st.write(f"Games found: {len(game_ids)}")

        all_rows = []
        games_with_stats = 0

        for gid in game_ids:
            rows = pull_game_sog(gid)
            if rows:
                games_with_stats += 1
                all_rows.extend(rows)

        df = pd.DataFrame(all_rows)

        st.write(f"Games with skater stats: {games_with_stats}")

        if df.empty:
            st.warning("No skater stats published yet for this league day.")
        else:
            st.success(f"Pulled {len(df)} skater rows")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download Actuals CSV",
                csv,
                file_name=f"nhl_actual_sog_{DATE}.csv",
                mime="text/csv"
            )
