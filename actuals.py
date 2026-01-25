import requests
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
DATE = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")

SCOREBOARD_URL = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={DATE}"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

# -----------------------------
# STEP 1: GET YESTERDAY GAMES
# -----------------------------
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

# -----------------------------
# STEP 2: PULL ACTUAL SOG
# -----------------------------
def get_boxscore_sog(game_id, game_date):
    r = requests.get(SUMMARY_URL.format(game_id), timeout=10)
    data = r.json()

    rows = []

    for team_block in data.get("boxscore", {}).get("players", []):
        team = team_block.get("team", {}).get("abbreviation")

        for group in team_block.get("statistics", []):
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

# -----------------------------
# STEP 3: BUILD ACTUALS
# -----------------------------
games = get_yesterday_games()
all_rows = []

for g in games:
    all_rows.extend(get_boxscore_sog(g["game_id"], g["date"]))

actuals_df = pd.DataFrame(all_rows)

# -----------------------------
# SAVE
# -----------------------------
out_file = f"nhl_actual_sog_{DATE}.csv"
actuals_df.to_csv(out_file, index=False)
print(f"Saved {len(actuals_df)} rows → {out_file}")
