# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Logo Matchup Buttons + Live Line Test (Fixed)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, datetime, requests, html, json, contextlib, io
from scipy.stats import poisson
import streamlit.components.v1 as components

st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("🧪 TEST MODE — Logo matchup buttons + live line test enabled.")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
        <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
    </div>
    <h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
    <p style='text-align:center;color:#D6D6D6;'>Clickable logo matchups with Form, Trend, Season Avg, and live line testing.</p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])
injuries_file = st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Data Load Helpers
# ---------------------------------------------------------------
def load_file(file):
    if not file: return pd.DataFrame()
    try:
        return pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
    except Exception:
        return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None:
        return load_file(file_uploader)
    return safe_read(default_path)

@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
    def find_file(name):
        for p in base_paths:
            full = os.path.join(p, name)
            if os.path.exists(full): return full
        return None
    skaters = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
    shots   = load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
    goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
    lines   = load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
    teams   = load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")
    injuries = load_file(injuries_file) if injuries_file else pd.DataFrame()
    return skaters, shots, goalies, lines, teams, injuries

# ---------------------------------------------------------------
# Load + Normalize
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data.")
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns = df.columns.str.lower().str.strip()

if "player" not in shots_df.columns:
    candidate = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
    if candidate: shots_df = shots_df.rename(columns={candidate:"player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
col_run, col_line = st.columns([3,1])
with col_run:
    run_model = st.button("🚀 Run Model (All Games Today)")
with col_line:
    line_test = st.number_input("Line to Test", min_value=0.0, max_value=10.0, value=3.5, step=0.5)

# ---------------------------------------------------------------
# Fetch ESPN Games
# ---------------------------------------------------------------
def fetch_espn_games():
    today = datetime.datetime.now().strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={today}"
    try:
        data = requests.get(url, timeout=10).json()
        games = []
        for event in data.get("events", []):
            c = event.get("competitions", [{}])[0]
            teams = c.get("competitors", [])
            if len(teams)==2:
                home, away = teams[0]["team"], teams[1]["team"]
                games.append({
                    "home": home["abbreviation"], "away": away["abbreviation"],
                    "home_full": home["displayName"], "away_full": away["displayName"],
                    "home_logo": home["logo"], "away_logo": away["logo"]
                })
        return games
    except Exception as e:
        st.error(f"Failed to fetch games: {e}")
        return []

# ---------------------------------------------------------------
# Build Model (simplified core)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df):
    results=[]
    team_col = next((c for c in skaters_df.columns if "team" in c), None)
    player_col="name" if "name" in skaters_df.columns else "player"
    game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)
    skaters = skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster=skaters[[player_col,team_col]].rename(columns={player_col:"player",team_col:"team"}).drop_duplicates("player")
    grouped={n.lower():g.sort_values(game_col) for n,g in shots_df.groupby(shots_df["player"].str.lower())}
    for row in roster.itertuples(index=False):
        player,team=row.player,row.team
        dfp=grouped.get(player.lower(),pd.DataFrame())
        if dfp.empty: continue
        sog_vals=dfp.groupby(game_col)["sog"].sum().tolist()
        if not sog_vals: continue
        season_avg=np.mean(sog_vals)
        last5=sog_vals[-5:] if len(sog_vals)>=5 else sog_vals
        last10=sog_vals[-10:] if len(sog_vals)>=10 else sog_vals
        l5,l10=np.mean(last5),np.mean(last10)
        lam=(0.55*l10+0.45*l5)
        trend=(l5-l10)/l10 if l10>0 else 0
        trend_symbol="▲" if trend>0.05 else ("▼" if trend<-0.05 else "–")
        results.append({
            "Player":player,"Team":team,
            "Season Avg":round(season_avg,2),
            "Trend":trend_symbol,
            "Form Indicator":"⚪ Neutral Form",
            "Final Projection":round(lam,2),
            "Line Adj":1.0
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run + Logo Buttons
# ---------------------------------------------------------------
if run_model:
    games = fetch_espn_games()
    if not games:
        st.warning("⚠️ No NHL games found for today.")
    else:
        st.subheader("🏒 Today's Matchups")
        if "selected_game" not in st.session_state:
            st.session_state.selected_game=None

        combined=pd.DataFrame()
        for g in games:
            df=build_model(g["away"],g["home"],skaters_df,shots_df,goalies_df,lines_df,teams_df)
            df["Matchup"]=f"{g['away']} @ {g['home']}"
            combined=pd.concat([combined,df],ignore_index=True)
        st.session_state.results_base=combined
        st.session_state.matchups=games
        st.success("✅ Games processed!")

# ---------------------------------------------------------------
# Logo Matchup Tiles + Filter
# ---------------------------------------------------------------
if "results_base" in st.session_state and "matchups" in st.session_state:
    games=st.session_state.matchups
    base_df=st.session_state.results_base.copy()

    cols=st.columns(min(len(games),4))
    for i,g in enumerate(games):
        with cols[i%len(cols)]:
            active=(st.session_state.selected_game==f"{g['away']} @ {g['home']}")
            button_label=f"{g['away']} @ {g['home']}"
            html_button=f"""
                <div style='border-radius:10px; padding:8px; margin:4px; text-align:center;
                            background-color:{'#1E5A99' if active else '#0F2743'};
                            box-shadow:{'0 0 12px #1E90FF' if active else 'none'};
                            cursor:pointer;'>
                    <img src='{g['away_logo']}' width='35' style='vertical-align:middle;'> 
                    <b style='color:white;'>{g['away_full']}</b>
                    <span style='color:#D6D6D6;'> @ </span>
                    <b style='color:white;'>{g['home_full']}</b>
                    <img src='{g['home_logo']}' width='35' style='vertical-align:middle;'>
                </div>
            """
            if st.button(button_label, key=f"btn_{i}", help="Click to filter matchup"):
                st.session_state.selected_game=None if active else button_label
                st.rerun()
            components.html(html_button,height=80)

    df=base_df
    if st.session_state.selected_game:
        away,home=st.session_state.selected_game.split(" @ ")
        df=df[df["Team"].isin([away,home])]

    # Line recalculation
    lam_vals=df["Final Projection"].astype(float)
    probs=1-poisson.cdf(line_test-1,mu=lam_vals.clip(lower=0.01))
    df[f"Prob ≥ {line_test} (%)"]=(probs*100).round(1)
    odds=np.where(probs>=0.5,-100*(probs/(1-probs)),100*((1-probs)/probs))
    odds=np.clip(odds,-5000,5000)
    df[f"Playable Odds ({line_test})"]=[f"{'+' if o>0 else ''}{int(o)}" for o in odds]

    vis_cols=["Player","Team","Season Avg","Trend","Form Indicator",
              "Final Projection",f"Prob ≥ {line_test} (%)",f"Playable Odds ({line_test})","Line Adj"]
    vis=df[[c for c in vis_cols if c in df.columns]]
    html_table=vis.to_html(index=False,escape=False)

    components.html(f"""
        <style>
        table {{
            width:100%;border-collapse:collapse;font-family:'Source Sans Pro',sans-serif;color:#D6D6D6;
        }}
        th {{
            background-color:#0A3A67;color:#FFFFFF;padding:6px;text-align:center;position:sticky;top:0;
        }}
        td:first-child,th:first-child {{
            position:sticky;left:0;background-color:#1E5A99;color:#FFFFFF;font-weight:bold;
        }}
        td {{
            background-color:#0F2743;color:#D6D6D6;padding:4px;text-align:center;
        }}
        tr:nth-child(even) td {{background-color:#142F52;}}
        </style>
        <div style='overflow-x:auto;height:620px;'>{html_table}</div>
    """,height=650,scrolling=True)
else:
    st.info("👆 Click **Run Model** to load today's games first.")
