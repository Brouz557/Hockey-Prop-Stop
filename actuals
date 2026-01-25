import requests
import pandas as pd
from datetime import datetime

# -------------------------------
# ESPN ENDPOINTS
# -------------------------------
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

# -------------------------------
# STEP 1: GET TODAY'S GAMES
# -------------------------------
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

# -------------------------------
# STEP 2: GET BOX SCORE (SOG)
# -------------------------------
def get_boxscore_sog(game_id, game_date):
    url = SUMMARY_URL.format(game_id)
    r = requests.get(url, timeout=10)
    data = r.json()

    rows = []

    boxscore = data.get("boxscore", {})
    teams = boxscore.get("teams", [])

    for team in teams:
        team_abbr = team.get("team", {}).get("abbreviation")

        for category in team.get("statistics", []):
            if category.get("name") != "skaters":
                continue

            for athlete in category.get("athletes", []):
                stats = athlete.get("stats", [])
                if len(stats) < 6:
                    continue

                try:
                    rows.append({
                        "date": game_date,
                        "game_id": game_id,
                        "team": team_abbr,
                        "player": athlete.get("athlete", {}).get("displayName"),
                        "sog": int(stats[5])  # Shots on Goal index
                    })
                except:
                    continue

    return rows

# -------------------------------
# STEP 3: BUILD ACTUALS TABLE
# -------------------------------
def build_actuals():
    games = get_todays_games()
    all_rows = []

    for g in games:
        print(f"Pulling game {g['away']} @ {g['home']} ({g['game_id']})")
        rows = get_boxscore_sog(g["game_id"], g["date"])
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    actuals_df = build_actuals()
    actuals_df.to_csv("nhl_actual_sog.csv", index=False)
    print("Saved nhl_actual_sog.csv")
