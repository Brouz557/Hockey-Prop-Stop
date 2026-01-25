import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
YESTERDAY = (datetime.utcnow() - timedelta(days=1))
DATE_STR = YESTERDAY.strftime("%Y%m%d")
DATE_DISPLAY = YESTERDAY.strftime("%Y-%m-%d")

SCOREBOARD_URL = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={DATE_STR}"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

st.set_page_config(page_title="NHL Actual SOG (Yesterday)", layout="wide")
st.title("📊 NHL Actual Shots on Goal — Yesterday")
st.caption(f"Pulling finalized ESPN skater stats for {DATE_DISPLAY}")

# -------------------------------------------------
# GET YESTERDAY GAMES
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def get_yesterday_games():
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

# -------------------------------------------------
# PULL BOX SCORE SOG (ROBUST)
# -------------------------------------------------
def get_boxscore_sog(game_id, game_date):
    r = requests.get(SUMMARY_URL.format(game_id), timeout=10)
    data = r.json()

    rows = []

    for team_block in data.get("boxscore", {}).get("players", []):
        team = team_block.get("team", {}).get("abbreviation")

        for group in team_block.get("statistics", []):
            # Skip goalies explicitly
            if group.get("name") == "goalies":
                continue

            for athlete in group.get("athletes", []):
                player = athlete.get("athlete", {}).get("displayName")
                stats = athlete.get("stats", [])

                sog = None
                for stat in stats:
                    if isinstance(stat, dict) and stat.get("name") == "shotsOnGoal":
                        sog = stat.get("value")
                        break

                if sog is not None:
                    rows.append({
                        "date": game_date,
                        "team": team,
                        "player": player,
                        "actual_sog": int(sog)
                    })

    return rows

# -------------------------------------------------
# BUILD ACTUALS
# -------------------------------------------------
def build_actuals():
    games = get_yesterday_games()
    all_rows = []

    for g in games:
        all_rows.extend(
            get_boxscore_sog(g["game_id"], g["date"])
        )

    return pd.DataFrame(all_rows), games

# -------------------------------------------------
# UI ACTION
# -------------------------------------------------
if st.button("📥 Pull Yesterday’s Actuals"):
    with st.spinner("Pulling ESPN skater box scores..."):
        actuals_df, games = build_actuals()
        st.session_state.actuals = actuals_df
        st.session_state.games = games

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
if "games" in st.session_state:
    st.subheader("🏒 Games Included")
    for g in st.session_state.games:
        st.write(f"{g['away']} @ {g['home']}")

if "actuals" in st.session_state:
    df = st.session_state.actuals

    if df.empty:
        st.error("No skater stats returned (unexpected for yesterday).")
    else:
        st.success(f"Pulled {len(df)} skater rows")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Yesterday Actuals (CSV)",
            data=csv,
            file_name=f"nhl_actual_sog_{DATE_STR}.csv",
            mime="text/csv"
        )
