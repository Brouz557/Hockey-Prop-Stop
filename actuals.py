import streamlit as st
import requests
import pandas as pd

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
st.set_page_config(page_title="NHL Box Score Exporter", layout="wide")
st.title("🏒 NHL Box Score Exporter")
st.caption("Exports Goals, Assists, Shots on Goal, and TOI from ESPN NHL box scores")

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

# --------------------------------------------------
# INPUT
# --------------------------------------------------
game_id = st.text_input(
    "Enter ESPN NHL Game ID",
    placeholder="e.g. 401803161"
)

# --------------------------------------------------
# BOX SCORE PULL FUNCTION (FINAL + CORRECT)
# --------------------------------------------------
def pull_boxscore(game_id):
    r = requests.get(SUMMARY_URL.format(game_id), timeout=10)
    data = r.json()

    rows = []

    # Exact ESPN NHL labels -> clean column names
    STAT_MAP = {
        "G": "goals",
        "A": "assists",
        "SOG": "sog",
        "TOI": "toi"
    }

    for team in data.get("boxscore", {}).get("players", []):
        team_abbr = team.get("team", {}).get("abbreviation")

        for group in team.get("statistics", []):
            # Skip goalie stats
            if group.get("name") == "goalies":
                continue

            labels = group.get("labels", [])

            for athlete in group.get("athletes", []):
                stats = athlete.get("stats", [])

                row = {
                    "team": team_abbr,
                    "player": athlete.get("athlete", {}).get("displayName"),
                }

                found_stat = False

                for i, label in enumerate(labels):
                    if label in STAT_MAP and i < len(stats):
                        row[STAT_MAP[label]] = stats[i]
                        found_stat = True

                # Only keep players with at least one stat
                if found_stat:
                    rows.append(row)

    return pd.DataFrame(rows)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if st.button("📊 Pull Box Score"):
    if not game_id.strip():
        st.warning("Please enter a valid ESPN game ID.")
    else:
        with st.spinner("Pulling box score from ESPN..."):
            df = pull_boxscore(game_id)

        if df.empty:
            st.error("No skater stats found for this game.")
        else:
            st.success(f"Pulled {len(df)} skaters")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Box Score CSV",
                data=csv,
                file_name=f"boxscore_{game_id}.csv",
                mime="text/csv"
            )
