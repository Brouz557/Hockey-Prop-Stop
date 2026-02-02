import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://leftwinglock.com/teams"
HEADERS = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(page_title="LeftWingLock Line Scraper", layout="wide")
st.title("🏒 LeftWingLock Line & Defense Parser (HTML)")

st.markdown("""
Scrapes projected forward lines & defensive pairings from LeftWingLock.
Does NOT include goalies.
""")

def get_team_list():
    r = requests.get(BASE_URL, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    teams = []
    for a in soup.select("table tr td a"):
        href = a.get("href")
        if href and "/teams/" in href:
            teams.append(href.strip("/"))
    return sorted(list(set(teams)))

def scrape_team_page(slug):
    url = f"{BASE_URL}/{slug}/"
    r = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    lines = []
    defense = []

    # ---- Forward Lines ----
    for line_section in soup.select(".lines table"):
        cells = [c.get_text(strip=True) for c in line_section.select("td")]
        # Expected 3 players per forward line
        if len(cells) >= 3:
            lines.append({
                "team": slug,
                "line_type": "FWD",
                "line_combo": " ".join(cells[:3]),
                "player1": cells[0],
                "player2": cells[1],
                "player3": cells[2]
            })

    # ---- Defense Pairings ----
    for d_section in soup.select(".dcor table"):
        cells = [c.get_text(strip=True) for c in d_section.select("td")]
        if len(cells) >= 2:
            defense.append({
                "team": slug,
                "pairing": f"{cells[0]} / {cells[1]}",
                "d1": cells[0],
                "d2": cells[1]
            })

    return lines, defense

if st.button("📡 Fetch All Teams"):
    slugs = get_team_list()
    all_lines, all_def = [], []

    progress = st.progress(0)
    status = st.empty()

    for i, slug in enumerate(slugs):
        status.write(f"Fetching {slug}")
        lines, def_pairings = scrape_team_page(slug)
        all_lines.extend(lines)
        all_def.extend(def_pairings)
        progress.progress((i + 1) / len(slugs))
        time.sleep(0.5)

    lines_df = pd.DataFrame(all_lines)
    def_df = pd.DataFrame(all_def)

    st.write("⛸ Forward Lines rows:", len(lines_df))
    st.write("🛡 Defense Pairings rows:", len(def_df))

    st.dataframe(lines_df.head(40))
    st.dataframe(def_df.head(40))

    st.download_button(
        "⬇️ Download Forward Lines CSV",
        lines_df.to_csv(index=False),
        file_name="leftwinglock_lines.csv"
    )

    st.download_button(
        "⬇️ Download Defense Pairings CSV",
        def_df.to_csv(index=False),
        file_name="leftwinglock_defense.csv"
    )
