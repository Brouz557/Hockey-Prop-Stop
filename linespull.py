import streamlit as st
import requests
import pandas as pd

BASE_URL = "https://www.dailyfaceoff.com"
API_TEMPLATE = BASE_URL + "/api/teams/{team}/line-combinations"

st.set_page_config(page_title="DailyFaceoff JSON Line Scraper", layout="wide")
st.title("🏒 DailyFaceoff Line & Goalie Scraper (JSON API)")

st.markdown("""
Scrapes structured JSON from DailyFaceoff’s internal API for all teams.
Includes:
- Forward lines (Line 1–4)
- Defensive pairings
- Power play / PK if available
- Goalies + statuses
""")

def fetch_team_slugs():
    """Get all NHL team slugs from DailyFaceoff teams page JSON."""
    r = requests.get(BASE_URL + "/teams")
    soup = r.text

    # The team page loads but contains slugs in the API data
    # Best reliable way: parse via separate JSON endpoint
    teams_api = requests.get(BASE_URL + "/api/teams").json()
    return [t["slug"] for t in teams_api]

def fetch_team_data(slug):
    url = API_TEMPLATE.format(team=slug)
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def parse_team_json(team_json):
    lines = []
    goalies = []

    team_name = team_json["team"]["name"]

    # Forward & Defensive Units
    for unit in team_json.get("lines", []):
        line_number = unit.get("position", "")
        lineup_type = unit.get("type", "").upper()

        for player in unit.get("players", []):
            lines.append({
                "team": team_name,
                "unit_type": lineup_type,
                "line_number": line_number,
                "player": player.get("fullName", "")
            })

    # Goalie Info
    for g in team_json.get("goalies", []):
        goalies.append({
            "team": team_name,
            "goalie": g.get("player", {}).get("fullName", ""),
            "status": g.get("status", "")
        })

    return lines, goalies

if st.button("📡 Fetch All Teams Line Data"):
    slugs = fetch_team_slugs()
    all_lines = []
    all_goalies = []

    progress = st.progress(0)
    status = st.empty()

    for i, slug in enumerate(slugs):
        status.write(f"Fetching {slug}…")
        data = fetch_team_data(slug)
        if data:
            lines, goalies = parse_team_json(data)
            all_lines.extend(lines)
            all_goalies.extend(goalies)
        progress.progress((i+1)/len(slugs))

    lines_df = pd.DataFrame(all_lines)
    goalies_df = pd.DataFrame(all_goalies)

    st.write("Lines rows:", len(lines_df))
    st.write("Goalies rows:", len(goalies_df))

    st.dataframe(lines_df.head(50), use_container_width=True)
    st.dataframe(goalies_df.head(50), use_container_width=True)

    st.download_button("⬇️ Download Lines CSV", lines_df.to_csv(index=False), file_name="dailyfaceoff_lines.csv")
    st.download_button("⬇️ Download Goalies CSV", goalies_df.to_csv(index=False), file_name="dailyfaceoff_goalies.csv")
