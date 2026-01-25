import streamlit as st
import requests
import pandas as pd

# -------------------------------------------------
# ESPN ENDPOINTS
# -------------------------------------------------
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

st.set_page_config(page_title="NHL Actual Shots on Goal (ESPN)", layout="wide")
st.title("📊 NHL Actual Shots on Goal (ESPN)")
st.caption("Uses ESPN boxscore.players schema (current)")

# -------------------------------------------------
# GET FINAL GAMES ONLY
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
# PULL BOX SCORE SOG (CORRECT SCHEMA)
# -------------------------------------------------
def get_boxscore_sog(game_id, game_date):
    r = requests.get(SUMMARY_URL.format(game_id), timeout=10)
    data = r.json()

    rows = []

    for team_block in data.get("boxscore", {}).get("players", []):
        team_abbr = team_block.get("team", {}).get("abbreviation")

        for stat_group in team_block.get("statistics", []):
            if stat_group.get("name") != "skaters":
                continue

            for athlete in stat_group.get("athletes", []):
                player_name = athlete.get("athlete", {}).get("displayName")

                sog = None
                for stat in athlete.get("stats", []):
                    if stat.get("name") == "shotsOnGoal":
                        sog = stat.get("
