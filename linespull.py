import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.dailyfaceoff.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(page_title="DailyFaceoff Line Scraper", layout="wide")
st.title("🏒 DailyFaceoff Line & Goalie Scraper")
st.caption("Forward lines 1–4 • Defensive pairings • Goalie status")

# --------------------------------------------------
# STEP 1: GET TEAM SLUGS (HTML – this works)
# --------------------------------------------------
def get_team_slugs():
    r = requests.get(f"{BASE_URL}/teams", headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    slugs = set()
    for a in soup.select("a[href^='/teams/']"):
        href = a.get("href")
        if href and href.count("/") == 2:
            slugs.add(href.replace("/teams/", "").strip("/"))

    return sorted(slugs)

# --------------------------------------------------
# STEP 2: FETCH LINE JSON PER TEAM
# --------------------------------------------------
def fetch_team_lines(slug):
    url = f"{BASE_URL}/api/teams/{slug}/line-combinations"
    r = requests.get(url, headers=HEADERS, timeout=20)

    if r.status_code != 200:
        return None

    try:
        return r.json()
    except Exception:
        return None

# --------------------------------------------------
# STEP 3: PARSE JSON
# --------------------------------------------------
def parse_team_data(data):
    lines = []
    goalies = []

    team_name = data["team"]["name"]

    for unit in data.get("lines", []):
        unit_type = unit.get("type", "").upper()      # FWD / DEF / PP / PK
        unit_name = unit.get("position", "")          # Line 1 / Pairing 1 / PP1

        for p in unit.get("players", []):
            lines.append({
                "team": team_name,
                "unit_type": unit_type,
                "unit": unit_name,
                "player": p.get("fullName", "")
            })

    for g in data.get("goalies", []):
        goalies.append({
            "team": team_name,
            "goalie": g.get("player", {}).get("fullName", ""),
            "status": g.get("status", "")
        })

    return lines, goalies

# --------------------------------------------------
# UI
# --------------------------------------------------
if st.button("▶️ Run DailyFaceoff Scraper"):
    slugs = get_team_slugs()

    all_lines = []
    all_goalies = []

    progress = st.progress(0)
    status = st.empty()

    for i, slug in enumerate(slugs):
        status.write(f"Fetching **{slug}**")
        data = fetch_team_lines(slug)

        if data:
            lines, goalies = parse_team_data(data)
            all_lines.extend(lines)
            all_goalies.extend(goalies)

        progress.progress((i + 1) / len(slugs))
        time.sleep(0.5)

    lines_df = pd.DataFrame(all_lines)
    goalies_df = pd.DataFrame(all_goalies)

    st.write("📊 Line rows:", len(lines_df))
    st.write("🥅 Goalie rows:", len(goalies_df))

    if lines_df.empty:
        st.error("❌ No line data returned — API structure may have changed.")
        st.stop()

    st.success("✅ Scraping complete!")

    st.subheader("Line Combinations (Preview)")
    st.dataframe(lines_df.head(50), use_container_width=True)

    st.subheader("Goalies (Preview)")
    st.dataframe(goalies_df.head(50), use_container_width=True)

    st.download_button(
        "⬇️ Download Line CSV",
        lines_df.to_csv(index=False),
        file_name="dailyfaceoff_lines.csv"
    )

    st.download_button(
        "⬇️ Download Goalie CSV",
        goalies_df.to_csv(index=False),
        file_name="dailyfaceoff_goalies.csv"
    )
