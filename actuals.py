import streamlit as st
import requests
import pandas as pd

# -------------------------------------------------
# ESPN ENDPOINTS
# -------------------------------------------------
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

st.set_page_config(
    page_title="NHL Actual Shots on Goal (ESPN)",
    layout="wide"
)

st.title("📊 NHL Actual Shots on Goal (ESPN)")
st.caption("Pulls box scores ONLY for games marked FINAL")

# -------------------------------------------------
# GET TODAY'S FINAL GAMES ONLY
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def get_final_games_today():
    r = requests.get(SCOREBOARD_URL, timeout=10)
    data = r.json()

    games = []

    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {}).get("name")

        if status != "STATUS_FINAL":
            continue

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
# PULL BOX SCORE SOG (CURRENT ESPN SCHEMA)
# -------------------------------------------------
def get_boxscore_sog(game_id, game_date):
    r = requests.get(SUMMARY_URL.format(game_id), timeout=10)
    data = r.json()

    rows = []

    boxscore = data.get("boxscore", {})
    players_blocks = boxscore.get("players", [])

    for team_block in players_blocks:
        team_abbr = team_block.get("team", {}).get("abbreviation")

        for stat_group in team_block.get("statistics", []):
            if stat_group.get("name") != "skaters":
                continue

            for athlete in stat_group.get("athletes", []):
                player_name = athlete.get("athlete", {}).get("displayName")

                sog = None
                for stat in athlete.get("stats", []):
                    if stat.get("name") == "shotsOnGoal":
                        sog = stat.get("value")
                        break

                if sog is None:
                    continue

                rows.append({
                    "date": game_date,
                    "game_id": game_id,
                    "team": team_abbr,
                    "player": player_name,
                    "sog": int(sog)
                })

    return rows

# -------------------------------------------------
# BUILD ACTUALS TABLE
# -------------------------------------------------
def build_actuals():
    games = get_final_games_today()
    all_rows = []

    for g in games:
        all_rows.extend(
            get_boxscore_sog(g["game_id"], g["date"])
        )

    return pd.DataFrame(all_rows), games

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.divider()

if st.button("📥 Pull Actuals for FINAL Games Today"):
    with st.spinner("Pulling ESPN box scores..."):
        actuals_df, final_games = build_actuals()
        st.session_state.actuals = actuals_df
        st.session_state.final_games = final_games

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------
if "final_games" in st.session_state:
    st.subheader("✅ Final Games Found")

    if not st.session_state.final_games:
        st.warning("No games are FINAL yet today.")
    else:
        for g in st.session_state.final_games:
            st.write(f"{g['away']} @ {g['home']}")

if "actuals" in st.session_state:
    df = st.session_state.actuals

    if df.empty:
        st.error("Games are FINAL, but no skater stats were returned.")
    else:
        st.success(f"Pulled {len(df)} skater rows")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download Actuals CSV",
            csv,
            file_name="nhl_actual_sog.csv",
            mime="text/csv"
        )
