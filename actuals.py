import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
st.set_page_config(page_title="NHL Actual SOG Exporter", layout="wide")
st.title("🏒 NHL Actuals Exporter (All Games)")
st.caption("Pulls Goals, Assists, Shots on Goal, and TOI from ESPN for all games on a selected date")

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={}"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

# --------------------------------------------------
# DATE SELECTOR
# --------------------------------------------------
game_date = st.date_input(
    "Select game date",
    value=datetime.utcnow() - timedelta(days=1)
)

date_str = game_date.strftime("%Y%m%d")

# --------------------------------------------------
# GET ALL GAME IDS FOR DATE
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def get_game_ids_for_date(date_str):
    r = requests.get(SCOREBOARD_URL.format(date_str), timeout=10)
    data = r.json()
    return [event["id"] for event in data.get("events", [])]

# --------------------------------------------------
# PULL BOXSCORE FOR ONE GAME (FINAL + BULLETPROOF)
# --------------------------------------------------
def pull_boxscore(game_id):
    r = requests.get(SUMMARY_URL.format(game_id), timeout=10)
    data = r.json()

    rows = []

    for team in data.get("boxscore", {}).get("players", []):
        team_abbr = team.get("team", {}).get("abbreviation")

        for group in team.get("statistics", []):
            # Skip goalies
            if group.get("name") == "goalies":
                continue

            labels = group.get("labels", [])

            for athlete in group.get("athletes", []):
                stats = athlete.get("stats", [])

                row = {
                    "team": team_abbr,
                    "player": athlete.get("athlete", {}).get("displayName"),
                }

                found = False

                for i, label in enumerate(labels):
                    if i >= len(stats):
                        continue

                    value = stats[i]

                    # ---- ESPN QUIRK HANDLING ----
                    if label == "S":
                        row["sog"] = value
                        found = True

                    elif label == "SOG" and "sog" not in row:
                        row["sog"] = value
                        found = True

                    elif label == "G":
                        row["goals"] = value
                        found = True

                    elif label == "A":
                        row["assists"] = value
                        found = True

                    elif label == "TOI":
                        row["toi"] = value
                        found = True

                if found:
                    rows.append(row)

    return pd.DataFrame(rows)

# --------------------------------------------------
# RUN ALL GAMES
# --------------------------------------------------
if st.button("📊 Pull ALL Games for Selected Date"):
    with st.spinner("Pulling ESPN box scores..."):
        game_ids = get_game_ids_for_date(date_str)
        st.write(f"Games found: {len(game_ids)}")

        all_dfs = []
        games_with_data = 0

        for gid in game_ids:
            df_game = pull_boxscore(gid)
            if not df_game.empty:
                df_game["game_id"] = gid
                df_game["date"] = date_str
                all_dfs.append(df_game)
                games_with_data += 1

        if not all_dfs:
            st.error("No skater stats found for any games.")
        else:
            final_df = pd.concat(all_dfs, ignore_index=True)
            st.success(f"Pulled skater stats from {games_with_data} games")

            st.dataframe(final_df, use_container_width=True)

            csv = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download Actuals CSV",
                csv,
                file_name=f"nhl_actuals_{date_str}.csv",
                mime="text/csv"
            )
