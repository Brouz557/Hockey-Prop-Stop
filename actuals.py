import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

DATE = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")

SCOREBOARD = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={DATE}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={}"

st.title("NHL Actual Shots on Goal (Yesterday)")
st.caption("Click once → download CSV")

@st.cache_data(show_spinner=False)
def pull_actuals():
    games = requests.get(SCOREBOARD).json().get("events", [])
    rows = []

    for e in games:
        gid = e["id"]
        summary = requests.get(SUMMARY.format(gid)).json()

        for team in summary.get("boxscore", {}).get("players", []):
            t = team["team"]["abbreviation"]

            for grp in team.get("statistics", []):
                if grp.get("name") == "goalies":
                    continue

                for a in grp.get("athletes", []):
                    for stat in a.get("stats", []):
                        if isinstance(stat, dict) and stat.get("name") == "shotsOnGoal":
                            rows.append({
                                "date": DATE,
                                "team": t,
                                "player": a["athlete"]["displayName"],
                                "actual_sog": stat["value"]
                            })

    return pd.DataFrame(rows)

if st.button("📥 Pull Yesterday Actuals"):
    df = pull_actuals()
    st.success(f"Pulled {len(df)} rows")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "💾 Download CSV",
        df.to_csv(index=False),
        file_name=f"nhl_actual_sog_{DATE}.csv",
        mime="text/csv"
    )
