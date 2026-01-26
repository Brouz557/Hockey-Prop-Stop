# export_moneypuck.py
# -----------------------------------------
# MoneyPuck Data Exporter
# This script pulls data and saves it to /data
# -----------------------------------------

import pandas as pd
import requests
from pathlib import Path

BASE_URL = "https://api.moneypuck.com/v2"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "PuckShotz-Exporter/1.0"
}

def fetch(endpoint: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{endpoint}"
    print(f"Fetching {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def main():
    # Player game-level data (shots, goals, TOI, etc.)
    player_games = fetch("playerGameStats")
    player_games.to_parquet(DATA_DIR / "player_games.parquet", index=False)

    # Goalie data
    goalies = fetch("goalieStats")
    goalies.to_parquet(DATA_DIR / "goalies.parquet", index=False)

    # Team data (pace, shots for/against, etc.)
    teams = fetch("teamStats")
    teams.to_parquet(DATA_DIR / "teams.parquet", index=False)

    print("✅ MoneyPuck export complete")

if __name__ == "__main__":
    main()
