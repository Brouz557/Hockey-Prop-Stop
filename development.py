# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Team Abbreviation Normalization (ESPN -> Data)
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ":"NJD","LA":"LAK","SJ":"SJS","TB":"TBL","ARI":"ARI","ANA":"ANA",
    "BOS":"BOS","BUF":"BUF","CAR":"CAR","CBJ":"CBJ","CGY":"CGY","CHI":"CHI",
    "COL":"COL","DAL":"DAL","DET":"DET","EDM":"EDM","FLA":"FLA","MIN":"MIN",
    "MTL":"MTL","NSH":"NSH","NYI":"NYI","NYR":"NYR","OTT":"OTT","PHI":"PHI",
    "PIT":"PIT","SEA":"SEA","STL":"STL","TOR":"TOR","VAN":"VAN","VGK":"VGK",
    "WSH":"WSH","WPG":"WPG"
}

st.set_page_config(page_title="Puck Shotz Hockey Analytics", layout="wide", page_icon="🏒")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups with inline logos, filters, injuries, and projections.</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load Data (auto-load like source of truth)
# ---------------------------------------------------------------
def safe_read(name):
    return pd.read_excel(name) if os.path.exists(name) else pd.DataFrame()

skaters_df  = safe_read("Skaters.xlsx")
shots_df    = safe_read("SHOT DATA.xlsx")
goalies_df  = safe_read("GOALTENDERS.xlsx")
lines_df    = safe_read("LINE DATA.xlsx")
teams_df    = safe_read("TEAMS.xlsx")
injuries_df = safe_read("injuries.xlsx")

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

if not injuries_df.empty:
    injuries_df.columns = injuries_df.columns.str.lower().str.strip()
    injuries_df["player"] = injuries_df["player"].astype(str).str.lower().str.strip()

team_col   = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"

shots_df = shots_df.rename(
    columns={next(c for c in shots_df.columns if "player" in c or "name" in c): "player"}
)
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c and "id" in c)

# ---------------------------------------------------------------
# ESPN Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_games():
    url="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    data=requests.get(url,timeout=10).json()
    games=[]
    for e in data.get("events",[]):
        c=e["competitions"][0]["competitors"]
        away = TEAM_ABBREV_MAP.get(c[0]["team"]["abbreviation"], c[0]["team"]["abbreviation"])
        home = TEAM_ABBREV_MAP.get(c[1]["team"]["abbreviation"], c[1]["team"]["abbreviation"])
        games.append({
            "away":away,
            "home":home,
            "away_logo":c[0]["team"]["logo"],
            "home_logo":c[1]["team"]["logo"]
        })
    return games

games = get_games()
if not games:
    st.stop()

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
col_run, col_line = st.columns([3,1])
with col_run:
    run_model = st.button("🚀 Run Model (All Games)")
with col_line:
    line_test = st.number_input("Line to Test",0.0,10.0,3.5,0.5)
    st.session_state.setdefault("line_test_val", line_test)
    if line_test != st.session_state.line_test_val:
        st.session_state.line_test_val = line_test

# ---------------------------------------------------------------
# Build Model (unchanged logic)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b):
    results=[]
    skaters = skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster  = skaters[[player_col,team_col]].drop_duplicates(subset=[player_col])
    grouped = {n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    for r in roster.itertuples(index=False):
        player, team = r
        dfp = grouped.get(player.lower(), pd.DataFrame())
        if dfp.empty or "sog" not in dfp.columns:
            continue

        sog_vals = dfp.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals) < 3:
            continue

        last3,last5,last10 = sog_vals[-3:], sog_vals[-5:], sog_vals[-10:]
        l3,l5,l10 = np.mean(last3), np.mean(last5), np.mean(last10)
        baseline = 0.55*l10 + 0.3*l5 + 0.15*l3
        trend = (l5-l10)/l10 if l10>0 else 0

        form_flag = "🟢 Above Baseline" if trend>0.05 else "🔴 Below Baseline" if trend<-0.05 else "⚪ Neutral"
        lam = round(baseline,2)

        # Injury popup
        injury_html=""
        hit = injuries_df[
            injuries_df["player"].str.endswith(player.split()[-1].lower())
        ] if not injuries_df.empty else pd.DataFrame()

        if not hit.empty:
            note = str(hit.iloc[0].get("injury note",""))
            itype = str(hit.iloc[0].get("injury type",""))
            date  = str(hit.iloc[0].get("date of injury",""))
            tooltip = "\n".join([x for x in [itype,note,date] if x])
            injury_html = f"<span style='cursor:pointer' onclick='alert({json.dumps(html.escape(tooltip))})'>🚑</span>"

        results.append({
            "Player":player,
            "Team":team,
            "Injury":injury_html,
            "Trend Score":round(trend,3),
            "Final Projection":lam,
            "Season Avg":round(np.mean(sog_vals),2),
            "Form Indicator":form_flag,
            "L3 Shots":", ".join(map(str,last3)),
            "L5 Shots":", ".join(map(str,last5)),
            "L10 Shots":", ".join(map(str,last10))
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model:
    tables=[]
    for g in games:
        df = build_model(g["away"], g["home"])
        if not df.empty:
            df["Matchup"] = f"{g['away']}@{g['home']}"
            tables.append(df)
    if tables:
        st.session_state.results = pd.concat(tables, ignore_index=True)
        st.session_state.matchups = games

# ---------------------------------------------------------------
# Display Buttons + Table (RESTORED)
# ---------------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state.results.copy()
    games = st.session_state.matchups

    # Matchup buttons
    cols = st.columns(3)
    for i,g in enumerate(games):
        match_id = f"{g['away']}@{g['home']}"
        selected = st.session_state.get("selected_match") == match_id
        bg = "#2F7DEB" if selected else "#1C5FAF"
        with cols[i%3]:
            with st.form(f"m_{i}"):
                st.markdown(f"""
                <div style="background:{bg};padding:10px;border-radius:8px;color:white;
                            display:flex;gap:6px;justify-content:center;align-items:center">
                    <img src="{g['away_logo']}" height="22">
                    {g['away']} @ {g['home']}
                    <img src="{g['home_logo']}" height="22">
                </div>
                """, unsafe_allow_html=True)
                if st.form_submit_button("View", use_container_width=True):
                    st.session_state.selected_match = None if selected else match_id
                    st.session_state.selected_teams = None if selected else {g["away"],g["home"]}
                    st.rerun()

    if st.session_state.get("selected_teams"):
        df = df[df["Team"].isin(st.session_state.selected_teams)]

    test_line = st.session_state.line_test_val
    df["Prob ≥ Line (%)"] = df["Final Projection"].apply(
        lambda x: round((1-poisson.cdf(test_line-1,mu=max(x,0.01)))*100,1)
    )

    cols = [
        "Player","Team","Injury","Final Projection","Prob ≥ Line (%)",
        "Season Avg","Form Indicator","L3 Shots","L5 Shots","L10 Shots"
    ]

    html_table = df[cols].sort_values("Final Projection",ascending=False)\
        .to_html(index=False,escape=False)

    components.html(f"""
    <style>
    table {{width:100%;border-collapse:collapse;color:#D6D6D6}}
    th {{background:#0A3A67;color:white;padding:6px;position:sticky;top:0}}
    td {{background:#0F2743;padding:4px;text-align:center}}
    tr:nth-child(even) td {{background:#142F52}}
    </style>
    <div style='height:900px;overflow:auto'>{html_table}</div>
    """,height=950)
