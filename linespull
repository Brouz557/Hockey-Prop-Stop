import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.dailyfaceoff.com"

headers = {
    "User-Agent": "Mozilla/5.0"
}

st.set_page_config(page_title="DailyFaceoff Line Scraper", layout="wide")
st.title("🏒 DailyFaceoff Line Combination Scraper")

st.markdown("""
This app scrapes **all NHL team line combinations and goalie statuses**
from DailyFaceoff and exports them to CSV.
""")

# --------------------------------------------------
# Scraper Function
# --------------------------------------------------
def run_scraper():
    teams_page = requests.get(f"{BASE_URL}/teams", headers=headers)
    soup = BeautifulSoup(teams_page.text, "html.parser")

    team_links = []
    for a in soup.select("a[href^='/teams/']"):
        href = a.get("href")
        if href.count("/") == 2:
            team_links.append(BASE_URL + href)

    team_links = list(set(team_links))

    rows = []
    goalies = []

    progress = st.progress(0)
    status = st.empty()

    for i, team_url in enumerate(team_links):
        team_name = team_url.split("/")[-1].replace("-", " ").title()
        url = team_url + "/line-combinations/"

        status.text(f"Scraping {team_name}")
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")

        # ---------- Forwards & Defense ----------
        for unit in soup.select(".line-combination"):
            unit_label = unit.select_one(".line-combination__name")
            if not unit_label:
                continue

            unit_name = unit_label.text.strip()
            players = unit.select(".player-name")

            for p in players:
                rows.append({
                    "team": team_name,
                    "unit_type": "FWD" if unit_name.startswith("Line") else "DEF",
                    "unit": unit_name,
                    "player": p.text.strip()
                })

        # ---------- Goalies ----------
        for g in soup.select(".goalie"):
            name = g.select_one(".player-name")
            status_tag = g.select_one(".goalie-status")

            if name:
                goalies.append({
                    "team": team_name,
                    "goalie": name.text.strip(),
                    "status": status_tag.text.strip() if status_tag else "Unknown"
                })

        progress.progress((i + 1) / len(team_links))
        time.sleep(1)

    lines_df = pd.DataFrame(rows)
    goalies_df = pd.DataFrame(goalies)

    return lines_df, goalies_df


# --------------------------------------------------
# UI
# --------------------------------------------------
if st.button("▶️ Run DailyFaceoff Scraper"):
    with st.spinner("Scraping DailyFaceoff…"):
        lines_df, goalies_df = run_scraper()

    st.success("✅ Scraping complete!")

    st.subheader("📄 Line Combinations Preview")
    st.dataframe(lines_df.head(25), use_container_width=True)

    st.subheader("🥅 Goalies Preview")
    st.dataframe(goalies_df.head(25), use_container_width=True)

    st.download_button(
        "⬇️ Download Line Combinations CSV",
        lines_df.to_csv(index=False),
        file_name="dailyfaceoff_lines.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇️ Download Goalies CSV",
        goalies_df.to_csv(index=False),
        file_name="dailyfaceoff_goalies.csv",
        mime="text/csv"
    )
