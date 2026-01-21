# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (Instant Filter + Logos)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests
from scipy.stats import poisson
import streamlit.components.v1 as components

st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("🧪 TEST MODE — Sandbox version. Changes here won’t affect your main app.")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups with inline logos and instant team filters.</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])
injuries_file= st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def load_file(f):
    if not f: return pd.DataFrame()
    try:
        return pd.read_excel(f) if f.name.lower().endswith(".xlsx") else pd.read_csv(f)
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
def load_all(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    base_paths=[".","data","/mount/src/hockey-prop-stop/data"]
    def find_file(name):
        for p in base_paths:
            fp=os.path.join(p,name)
            if os.path.exists(fp): return fp
        return None
    skaters=load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
    shots  =load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
    goalies=load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
    lines  =load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
    teams  =load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")
    injuries=pd.DataFrame()
    for p in ["injuries.xlsx","Injuries.xlsx","data/injuries.xlsx"]:
        if os.path.exists(p):
            injuries=load_file(open(p,"rb"));break
    if injuries.empty:
        injuries=load_file(injuries_file)
    if not injuries.empty:
        injuries.columns=injuries.columns.str.lower().str.strip()
        if "player" in injuries.columns:
            injuries["player"]=injuries["player"].astype(str).str.strip().str.lower()
    return skaters,shots,goalies,lines,teams,injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing data. Upload required files.")
    st.stop()
st.success("✅ Data loaded successfully.")

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns=df.columns.str.lower().str.strip()
team_col=next((c for c in skaters_df.columns if "team" in c),None)
player_col="name" if "name" in skaters_df.columns else None
shots_df=shots_df.rename(columns={next((c for c in shots_df.columns if "player" in c or "name" in c),"player"):"player"})
shots_df["player"]=shots_df["player"].astype(str).str.strip()
game_col=next((c for c in shots_df.columns if "game" in c and "id" in c),None)

# ---------------------------------------------------------------
# Matchup Pull (ESPN)
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_todays_games():
    url="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    r=requests.get(url,timeout=10)
    data=r.json()
    games=[]
    for e in data.get("events",[]):
        comps=e.get("competitions",[{}])[0].get("competitors",[])
        if len(comps)==2:
            away,home=comps[0],comps[1]
            games.append({
                "away":away["team"]["abbreviation"],
                "home":home["team"]["abbreviation"],
                "away_logo":away["team"]["logo"],
                "home_logo":home["team"]["logo"]
            })
    return games

games=get_todays_games()
if not games:
    st.warning("No games found today.")
    st.stop()

# ---------------------------------------------------------------
# Run Button / Line Input
# ---------------------------------------------------------------
col_run,col_line=st.columns([3,1])
with col_run: run_model=st.button("🚀 Run Model (All Games)")
with col_line:
    line_test=st.number_input("Line to Test",0.0,10.0,3.5,0.5,key="line_test")
    if "line_test_val" not in st.session_state:
        st.session_state.line_test_val=line_test
    elif st.session_state.line_test_val!=line_test:
        st.session_state.line_test_val=line_test
        if "results" in st.session_state:
            st.rerun()

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a,team_b,skaters_df,shots_df,goalies_df,lines_df,teams_df,injuries_df):
    results=[]
    skaters=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster=skaters[[player_col,team_col]].rename(columns={player_col:"player",team_col:"team"}).drop_duplicates("player")
    grouped = {n.lower(): g for n, g in shots_df.groupby(shots_df["player"].str.lower())}
    # rest of model unchanged ...
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Display Buttons + Filtered Table (logos + white click box)
# ---------------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state.results.copy()
    games = st.session_state.matchups

    cols = st.columns(3)
    for i, m in enumerate(games):
        team_a, team_b = m["away"], m["home"]
        match_id = f"{team_a}@{team_b}"
        is_selected = st.session_state.get("selected_match") == match_id

        btn_color = "#1E5A99" if is_selected else "#0A3A67"
        border = "2px solid #FF4B4B" if is_selected else "1px solid #1E5A99"
        glow = "0 0 12px #FF4B4B" if is_selected else "none"

        # top visual section only
        logo_html = f"""
        <div style="
            background-color:{btn_color};
            border:{border};
            border-radius:8px 8px 0 0;
            color:#fff;
            font-weight:600;
            font-size:15px;
            padding:10px 14px;
            width:100%;
            box-shadow:{glow};
            display:flex;
            align-items:center;
            justify-content:center;
            gap:6px;">
            <img src="{m['away_logo']}" height="22">
            <span>{m['away']}</span>
            <span style="color:#D6D6D6;">@</span>
            <span>{m['home']}</span>
            <img src="{m['home_logo']}" height="22">
        </div>
        """

        with cols[i % 3]:
            st.markdown(logo_html, unsafe_allow_html=True)

            # white clickable button below
            clicked = st.button(
                "Click to view",
                key=f"match_{i}",
                use_container_width=True,
                type="secondary"
            )

            if clicked:
                if is_selected:
                    st.session_state.selected_match = None
                    st.session_state.selected_teams = None
                else:
                    st.session_state.selected_match = match_id
                    st.session_state.selected_teams = {team_a, team_b}
                st.rerun()

    # --- Filter ---
    sel_teams = st.session_state.get("selected_teams")
    if sel_teams:
        df = df[df["Team"].isin(sel_teams)]
        st.markdown(f"### Showing results for: **{' vs '.join(sel_teams)}**")
    else:
        st.markdown("### Showing results for: **All Teams**")

    # (table code unchanged below)
