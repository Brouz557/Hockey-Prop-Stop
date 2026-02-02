import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

BASE_URL = "https://www.lineups.com/nhl/line-combinations"
HEADERS = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(page_title="NHL Line Scraper (Lineups.com)", layout="wide")
st.title("🏒 NHL Line Combinations Scraper")
st.caption("Source: Lineups.com • Forward lines 1–4 • Defensive pairings")

# --------------------------------------------------
# TEAM SLUGS (STATIC = STABLE)
# --------------------------------------------------
TEAM_SLUGS = {
    "Anaheim Ducks": "anaheim-ducks",
    "Arizona Coyotes": "arizona-coyotes",
    "Boston Bruins": "boston-bruins",
    "Buffalo Sabres": "buffalo-sabres",
    "Calgary Flames": "calgary-flames",
    "Carolina Hurricanes": "carolina-hurricanes",
    "Chicago Blackhawks": "chicago-blackhawks",
    "Colorado Avalanche": "colorado-avalanche",
    "Columbus Blue Jackets": "columbus-blue-jackets",
    "Dallas Stars": "dallas-stars",
    "Detroit Red Wings": "detroit-red-wings",
    "Edmonton Oilers": "edmonton-oilers",
    "Florida Panthers": "florida-panthers",
    "Los Angeles Kings": "los-angeles-kings",
    "Minnesota Wild": "minnesota-wild",
    "Montreal Canadiens": "montreal-canadiens",
    "Nashville Predators": "nashville-predators",
    "New Jersey Devils": "new-jersey-devils",
    "New York Islanders": "new-york-islanders",
    "New York Rangers": "new-york-rangers",
    "Ottawa Senators": "ottawa-senators",
    "Philadelphia Flyers": "philadelphia-flyers",
    "Pittsburgh Penguins": "pittsburgh-penguins",
    "San Jose Sharks": "san-jose-sharks",
    "Seattle Kraken": "seattle-kraken",
    "St. Louis Blues": "st-louis-blues",
    "Tampa Bay Lightning": "tampa-bay-lightning",
    "Toronto Maple Leafs": "toronto-maple-leafs",
    "Vancouver Canucks": "vancouver-canucks",
    "Vegas Golden Knights": "vegas-golden-knights",
    "Washington Capitals": "washington-capitals",
    "Winnipeg Jets": "winnipeg-jets"
}

# --------------------------------------------------
# SCRAPER
# --------------------------------------------------
def scrape_team(team, slug):
    url = f"{BASE_URL}/{slug}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []

    for cell in soup.select("td"):
        label = cell.get_text(strip=True)

        # Match positions like 1LW, 2C, 3RW, 1LD, 2RD
        match = re.match(r"([1-4])([A-Z]{1,2})", label)
        if not match:
            continue

        line_num = match.group(1)
        position = match.group(2)

        player_tag = cell.find_next("a")
        if not player_tag:
            continue

        player = player_tag.get_text(strip=True)

        unit_type = "DEF" if position in ["LD", "RD"] else "FWD"

        rows.append({
            "team": team,
            "unit_type": unit_type,
            "line": int(line_num),
            "position": position,
            "player": player
        })

    return rows

# --------------------------------------------------
# UI
# --------------------------------------------------
if st.button("▶️ Run Line Scraper"):
    all_rows = []

    progress = st.progress(0)
    status = st.empty()

    teams = list(TEAM_SLUGS.items())

    for i, (team, slug) in enumerate(teams):
        status.write(f"Scraping **{team}**")
        all_rows.extend(scrape_team(team, slug))
        progress.progress((i + 1) / len(teams))
        time.sleep(0.3)

    df = pd.DataFrame(all_rows)

    st.write("📊 Rows scraped:", len(df))

    if df.empty:
        st.error("❌ No data scraped. Site structure may have changed.")
        st.stop()

    st.success("✅ Scraping complete!")

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.download_button(
        "⬇️ Download NHL Line CSV",
        df.to_csv(index=False),
        file_name="nhl_lines_lineups.csv"
    )
