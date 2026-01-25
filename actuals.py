import streamlit as st
import requests
import pandas as pd
from datetime import datetime

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

@st.cache_data(show_spinner=False)
def get_todays_games():
    r = requests.get(SCOREBOARD_URL, timeout=10)
    data = r.json()
    games = []

    for event in data.get("events", []):
        game_id = event.get("id")
        date = event.get("date", "")[:10]
        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        teams = {c["homeAway"]: c["team"]["abbreviation"] for c in competitors}

        games.append({
            "game_id": game_id,
            "date": date,
            "home": teams.get("home"),
            "away": teams.get("away")
        })
    return games

def get_boxscore_sog(game_id, game_date):
    r = requests.get(SUMMARY_URL.format(game_id), timeout=10)
    data = r.json()
    rows = []

    for team in data.get("boxscore", {}).get("teams", []):
        team_abbr = team.get("team", {}).get("abbreviation")

        for cat in team.get("statistics", []):
            if cat.get("name") != "skaters":
                continue

            for ath in cat.get("athletes", []):
                stats = ath.get("stats", [])
                if len(stats) < 6:
                    continue

                rows.append({
                    "date": game_date,
                    "game_id": game_id,
                    "team": team_abbr,
                    "player": ath.get("athlete", {}).get("displayName"),
                    "sog": int(stats[5])
                })
    return rows

def build_actuals():
    games = get_todays_games()
    all_rows = []

    for g in games:
        rows = get_boxscore_sog(g["game_id"], g["date"])
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)

# ---------------- STREAMLIT UI ----------------
st.header("📊 Pull NHL Actual Shots on Goal (ESPN)")

if st.button("📥 Pull Actuals for Today’s Games"):
    with st.spinner("Pulling ESPN box scores..."):
        actuals_df = build_actuals()
        st.session_state.actuals = actuals_df
        st.success(f"Pulled {len(actuals_df)} skater rows")

if "actuals" in st.session_state:
    st.dataframe(st.session_state.actuals, use_container_width=True)

    csv = st.session_state.actuals.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Actuals CSV",
        csv,
        file_name="nhl_actual_sog.csv",
        mime="text/csv"
    )
