import pandas as pd
import requests
from datetime import date, timedelta

SEASON = "20252026"

def get_recent_games(days_back=3):
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
            if comp.get("status", {}).get("type", {}).get("completed"):
                games.append({
                    "game_id": event["id"],
                    "date": d,
                    "teams": [
                        c["team"]["abbreviation"]
                        for c in comp.get("competitors", [])
                    ]
                })

    return games


def get_latest_game_per_team(games):
    latest = {}

    for g in sorted(games, key=lambda x: x["date"], reverse=True):
        for team in g["teams"]:
            if team not in latest:
                latest[team] = g["game_id"]

    return latest


def load_nst_game_csv(game_id):
    url = (
        "https://www.naturalstattrick.com/game.php"
        f"?season={SEASON}"
        f"&game={game_id}"
        "&view=limited"
        "&csv=y"
    )
    return pd.read_csv(url)
