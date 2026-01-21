# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Full Interactive Matchup Version (Fixed)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, datetime, pytz, subprocess, html, json, requests
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("🧪 TEST MODE — You’re editing and running the sandbox version. Changes here will NOT affect your main app.")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
        <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
    </div>
    <h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
    <p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups — fully integrated player projections.</p>
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
# Helper Functions
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

# ---------------------------------------------------------------
# Cached Data Load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
    def find_file(name):
        for p in base_paths:
            full = os.path.join(p, name)
            if os.path.exists(full): return full
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        skaters = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
        shots   = load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
        goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
        lines   = load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
        teams   = load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")

        injuries = pd.DataFrame()
        for p in ["injuries.xlsx","Injuries.xlsx","./injuries.xlsx","data/injuries.xlsx","/mount/src/hockey-prop-stop/injuries.xlsx"]:
            if os.path.exists(p):
                injuries = load_file(open(p,"rb")); break
        if injuries.empty:
            injuries = load_file(injuries_file)
        if not injuries.empty:
            injuries.columns = injuries.columns.str.lower().str.strip()
            if "player" in injuries.columns:
                injuries["player"] = injuries["player"].astype(str).str.strip().str.lower()
    return skaters, shots, goalies, lines, teams, injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()
st.success("✅ Data loaded successfully.")

# Normalize Columns
for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns = df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
shots_df = shots_df.rename(columns={player_col_shots: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)

# ---------------------------------------------------------------
# ESPN Matchups Fetch
# ---------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_todays_matchups():
    url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        games = []
        for event in data.get("events", []):
            c = event["competitions"][0]["competitors"]
            home = c[0] if c[0]["homeAway"]=="home" else c[1]
            away = c[1] if c[0]["homeAway"]=="home" else c[0]
            games.append({
                "home": home["team"]["abbreviation"],
                "away": away["team"]["abbreviation"],
                "home_logo": home["team"]["logo"],
                "away_logo": away["team"]["logo"]
            })
        return games
    except Exception as e:
        st.error(f"Could not fetch games: {e}")
        return []

matchups = get_todays_matchups()

# ---------------------------------------------------------------
# Run + Line Input
# ---------------------------------------------------------------
col_run, col_line = st.columns([3,1])
with col_run:
    run_model = st.button("🚀 Run Model (All Games Today)")
with col_line:
    line_test = st.number_input("Line to Test", min_value=0.0, max_value=10.0, value=3.5, step=0.5)

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results=[]
    skaters=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    if skaters.empty:
        return pd.DataFrame()
    roster=skaters[[player_col,team_col]].rename(columns={player_col:"player",team_col:"team"}).drop_duplicates("player")
    grouped={n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}
    for row in roster.itertuples(index=False):
        player,team=row.player,row.team
        df_p=grouped.get(player.lower(),pd.DataFrame())
        if df_p.empty: continue
        sog_values=df_p["sog"].tolist()
        if not sog_values: continue
        last3=sog_values[-3:] if len(sog_values)>=3 else sog_values
        last5=sog_values[-5:] if len(sog_values)>=5 else sog_values
        last10=sog_values[-10:] if len(sog_values)>=10 else sog_values
        l3,l5,l10=np.mean(last3),np.mean(last5),np.mean(last10)
        lam=(0.55*l10)+(0.30*l5)+(0.15*l3)
        hit_l3=np.mean(np.array(last3)>=lam)
        hit_l5=np.mean(np.array(last5)>=lam)
        hit_l10=np.mean(np.array(last10)>=lam)
        empirical_prob=np.nanmean([0.55*hit_l10+0.30*hit_l5+0.15*hit_l3])
        poisson_prob=1-poisson.cdf(np.floor(lam)-1,mu=max(lam,0.01))
        final_prob=0.6*poisson_prob+0.4*empirical_prob
        trend=(l5-l10)/l10 if l10>0 else 0
        form_flag="⚪ Neutral Form"
        if trend>0.05: form_flag="🟢 Above-Baseline Form"
        elif trend<-0.05: form_flag="🔴 Below-Baseline Form"
        results.append({
            "Player":player,"Team":team,
            "Trend Score":round(trend,3),
            "Final Projection":round(lam,2),
            "Prob ≥ Projection (%) L5":round(final_prob*100,1),
            "Season Avg":round(np.mean(sog_values),2),
            "Line Adj":1.0,
            "Form Indicator":form_flag
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model:
    all_results=[]
    for m in matchups:
        team_a,team_b=m["away"],m["home"]
        df=build_model(team_a,team_b,skaters_df,shots_df,goalies_df,lines_df,teams_df,injuries_df)
        if not df.empty:
            all_results.append(df)
    if all_results:
        combined=pd.concat(all_results,ignore_index=True)
        combined=combined.sort_values(["Team","Final Projection","Line Adj"],ascending=[True,False,False]).reset_index(drop=True)
        st.session_state.results_base=combined
        st.session_state.selected_game=None
        st.success("✅ Model built successfully!")
    else:
        st.warning("⚠️ No valid player data found for today's teams.")

# ---------------------------------------------------------------
# Matchup Tiles
# ---------------------------------------------------------------
if matchups:
    st.markdown("### 🏒 Today's Matchups")
    cols=st.columns(3)
    for i,m in enumerate(matchups):
        with cols[i%3]:
            clicked=st.button(
                f"{m['away']} @ {m['home']}",
                key=f"matchup_{i}",
                help="Click to filter table for this game"
            )
            if clicked:
                if st.session_state.get("selected_game")==f"{m['away']}@{m['home']}":
                    st.session_state.selected_game=None
                else:
                    st.session_state.selected_game=f"{m['away']}@{m['home']}"

# ---------------------------------------------------------------
# Display Table
# ---------------------------------------------------------------
if "results_base" in st.session_state:
    df=st.session_state.results_base.copy()
    if "Trend Score" not in df.columns:
        df["Trend Score"]=0.0
    if st.session_state.get("selected_game"):
        away,home=st.session_state.selected_game.split("@")
        df=df[df["Team"].isin([away.strip(),home.strip()])]
    df["Trend"]=df["Trend Score"].apply(lambda v:"▲" if v>0.05 else ("▼" if v<-0.05 else "–"))
    df[f"Prob ≥ {line_test} (%)"]=(1-poisson.cdf(line_test-1,mu=df["Final Projection"].clip(lower=0.01)))*100
    df[f"Prob ≥ {line_test} (%)"]=df[f"Prob ≥ {line_test} (%)"].round(1)
    cols=["Player","Team","Trend","Final Projection",f"Prob ≥ {line_test} (%)",
          "Season Avg","Line Adj","Form Indicator"]
    vis=df[[c for c in cols if c in df.columns]]
    html_table=vis.to_html(index=False,escape=False)
    components.html(f"""
        <style>
        table {{
            width:100%;border-collapse:collapse;font-family:'Source Sans Pro',sans-serif;color:#D6D6D6;
        }}
        th {{
            background-color:#0A3A67;color:#FFFFFF;padding:6px;text-align:center;position:sticky;top:0;
            border-bottom:2px solid #1E5A99;
        }}
        td {{
            background-color:#0F2743;color:#D6D6D6;padding:4px;text-align:center;
        }}
        tr:nth-child(even) td {{background-color:#142F52;}}
        </style>
        <div style='overflow-x:auto;height:650px;'>{html_table}</div>
        """,height=680,scrolling=True)
