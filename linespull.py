import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_URL = "https://www.dailyfaceoff.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(
    page_title="DailyFaceoff Line Scraper",
    layout="wide"
)

st.title("🏒 DailyFaceoff Line & Goalie Scraper")
st.caption("Scrapes all NHL teams • Line combinations • Goalie status")

# --------------------------------------------------
# Scraper Function
# --------------------------------------------------
def run_scraper():
    teams_page = requests.get(f"{BASE_URL}/teams", headers=HEADERS, timeout=20)
    soup = BeautifulSoup(teams_page.text, "html.parser")

    team_links = []
    for a in soup.select("a[href^='/teams/']"):
        href = a.get("href")
        if href and href.count("/") == 2:
            team_links.append(BASE_URL + href)

    team_links = list(set(team_links))

    rows = []
    goalies = []

    progress = st.progress(0)
    status = st.empty()

    for i, team_url in enumerate(team_links):
        team_name = team_url.split("/")[-1].replace("-", " ").title()
        page_url = team_url + "/line-combinations/"

        status.write(f"Scraping **{team_name}**")

        r = requests.get(page_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")

        # --------------------------------------------------
        # FORWARDS & DEFENSE
        # --------------------------------------------------
        for table in soup.select("div[class*='line-combinations__table']"):
            unit_title = table.select_one("div[class*='line-combinations__title']")
            if not unit_title:
                continue

            unit_name = unit_title.text.strip()

            players = table.select("a[href*='/players/']")
            for p in players:
                rows.append({
                    "team": team_name,
                    "unit_type": "FWD" if "Line" in unit_name else "DEF",
                    "unit": unit_name,
                    "player": p.text.strip()
                })

        # --------------------------------------------------
        # GOALIES
        # --------------------------------------------------
        for g in soup.select("div[class*='goalie']"):
            name = g.select_one("a[href*='/players/']")
            status_tag = g.select_one("span")

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

    # --------------------------------------------------
    # FAIL SAFES
    # --------------------------------------------------
    st.write("📊 Line rows scraped:", len(lines_df))
    st.write("🥅 Goalie rows scraped:", len(goalies_df))

    if lines_df.empty:
        st.error("❌ No line data scraped. DailyFaceoff structure may have changed.")
        st.stop()

    st.success("✅ Scraping complete!")

    # --------------------------------------------------
    # PREVIEWS
    # --------------------------------------------------
    st.subheader("📄 Line Combinations (Preview)")
    st.dataframe(lines_df.head(50), use_container_width=True)

    st.subheader("🥅 Goalies (Preview)")
    st.dataframe(goalies_df.head(50), use_container_width=True)

    # --------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------
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

