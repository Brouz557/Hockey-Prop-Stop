import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="NHL Box Score Exporter", layout="wide")
st.title("🏒 NHL Box Score → Table Export")

st.caption("Enter an ESPN NHL game ID and export skater box score data")

game_id = st.text_input(
    "Enter ESPN Game ID",
    placeholder="e.g. 401803161"
)

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

def pull_boxscore(game_id):
    r = requests.get(SUMMARY_URL.format(game_id), timeout=10)
    data = r.json()

    rows = []

    for team in data.get("boxscore", {}).get("players", []):
        team_abbr = team.get("team", {}).get("abbreviation")

        for group in team.get("statistics", []):
            if group.get("name") == "goalies":
                continue

            for athlete in group.get("athletes", []):
                row = {
                    "team": team_abbr,
                    "player": athlete.get("athlete", {}).get("displayName")
                }

                for stat in athlete.get("stats", []):
                    if isinstance(stat, dict):
                        row[stat["name"]] = stat.get("value")

                rows.append(row)

    return pd.DataFrame(rows)

if st.button("📊 Pull Box Score"):
    if not game_id.strip():
        st.warning("Please enter a game ID.")
    else:
        with st.spinner("Pulling box score..."):
            df = pull_boxscore(game_id)

        if df.empty:
            st.error("No skater stats found for this game.")
        else:
            st.success(f"Pulled {len(df)} skaters")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download Box Score CSV",
                csv,
                file_name=f"boxscore_{game_id}.csv",
                mime="text/csv"
            )
