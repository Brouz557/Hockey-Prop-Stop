# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Fixed Overlay + Inline Buttons
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, requests, html, json
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
<p style='text-align:center;color:#D6D6D6;'>Automatic matchup analytics with clickable logo buttons + live line updates.</p>
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
# ESPN Matchup Pull
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
    st.warning("No NHL games found today.")
    st.stop()

# ---------------------------------------------------------------
# Run Button / Line Input
# ---------------------------------------------------------------
col_run,col_line=st.columns([3,1])
with col_run:
    run_model=st.button("🚀 Run Model (All Games)", use_container_width=True)
with col_line:
    line_test=st.number_input("Line to Test",0.0,10.0,3.5,0.5,key="line_test")

# ---------------------------------------------------------------
# Build Model (Simplified for display)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a,team_b,skaters_df,shots_df,goalies_df,lines_df,teams_df,injuries_df):
    results=[]
    skaters=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster=skaters[[player_col,team_col]].rename(columns={player_col:"player",team_col:"team"}).drop_duplicates("player")
    grouped={n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}
    for row in roster.itertuples(index=False):
        player,team=row.player,row.team
        df_p=grouped.get(player.lower(),pd.DataFrame())
        if df_p.empty: continue
        sog_vals=df_p.groupby(game_col)["sog"].sum().tolist()
        if not sog_vals: continue
        last3, last5, last10 = sog_vals[-3:], sog_vals[-5:], sog_vals[-10:]
        l3,l5,l10=np.mean(last3),np.mean(last5),np.mean(last10)
        baseline=(0.55*l10)+(0.3*l5)+(0.15*l3)
        lam=baseline
        poisson_prob=float(np.clip(1-poisson.cdf(np.floor(lam)-1,mu=max(lam,0.01)),0.0001,0.9999))
        results.append({
            "Player":player,"Team":team,
            "Trend Score":round((l5-l10)/l10 if l10>0 else 0,3),
            "Final Projection":round(lam,2),
            "Season Avg":round(np.mean(sog_vals),2),
            "Line Adj":round(np.random.uniform(0.9,1.1),2),
            "Prob Raw":poisson_prob
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model:
    all_tables=[]
    for m in games:
        team_a,team_b=m["away"],m["home"]
        df=build_model(team_a,team_b,skaters_df,shots_df,goalies_df,lines_df,teams_df,injuries_df)
        if not df.empty:
            df["Matchup"]=f"{team_a}@{team_b}"
            all_tables.append(df)
    if all_tables:
        combined=pd.concat(all_tables,ignore_index=True)
        st.session_state.results=combined
        st.session_state.matchups=games
        st.success("✅ Model built for all games.")

# ---------------------------------------------------------------
# Display Matchups + Table
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results.copy()
    games=st.session_state.matchups

    st.markdown("<h3 style='color:#1E5A99;'>Today's Matchups</h3>",unsafe_allow_html=True)
    cols=st.columns(3)
    for i,m in enumerate(games):
        match_id=f"{m['away']}@{m['home']}"
        selected=st.session_state.get("selected_match")==match_id
        bg="#1E5A99" if selected else "#0A3A67"
        border="3px solid #FF4B4B" if selected else "1px solid #1E5A99"
        html_button=f"""
        <button style='display:flex;align-items:center;justify-content:center;
            background:{bg};border:{border};border-radius:10px;
            padding:10px;margin:6px;width:100%;color:white;font-weight:600;
            font-size:15px;cursor:pointer;'
            onclick="fetch('/_stcore/streamlit/re-run?match={match_id}')">
            <img src='{m['away_logo']}' height='20'>
            <span style='margin:0 6px;'>{m['away']}</span>
            <span style='color:#D6D6D6;'>@</span>
            <span style='margin:0 6px;'>{m['home']}</span>
            <img src='{m['home_logo']}' height='20'>
        </button>
        """
        with cols[i%3]:
            if st.button(match_id,key=f"match_{i}",use_container_width=True):
                st.session_state.selected_match=None if selected else match_id
            st.markdown(html_button,unsafe_allow_html=True)

    sel=st.session_state.get("selected_match")
    if sel: df=df[df["Matchup"]==sel]

    lam_vals=df["Final Projection"].astype(float)
    probs=1-poisson.cdf(line_test-1,mu=lam_vals.clip(lower=0.01))
    df[f"Prob ≥ {line_test} (%)"]=(probs*100).round(1)
    odds=np.where(probs>=0.5,-100*(probs/(1-probs)),100*((1-probs)/probs))
    df[f"Playable Odds ({line_test})"]=[f"{'+' if o>0 else ''}{int(o)}" for o in odds]

    df["Trend"]=df["Trend Score"].apply(lambda v:"▲" if v>0.05 else("▼" if v<-0.05 else "–"))
    df=df.sort_values(["Team","Final Projection","Line Adj"],ascending=[True,False,False])

    html_table=df[["Player","Team","Trend","Final Projection",f"Prob ≥ {line_test} (%)",
                   f"Playable Odds ({line_test})","Season Avg","Line Adj"]].to_html(index=False,escape=False)

    components.html(f"""
    <style>
    table {{
        width:100%;border-collapse:collapse;font-family:'Source Sans Pro',sans-serif;color:#D6D6D6;
    }}
    th {{
        background-color:#0A3A67;color:#FFFFFF;padding:6px;text-align:center;position:sticky;top:0;
        border-bottom:2px solid #1E5A99;
    }}
    td:first-child,th:first-child {{
        position:sticky;left:0;background-color:#1E5A99;color:#FFFFFF;font-weight:bold;
    }}
    td {{
        background-color:#0F2743;color:#D6D6D6;padding:4px;text-align:center;
    }}
    tr:nth-child(even) td {{background-color:#142F52;}}
    </style>
    <div style='overflow-x:auto;height:650px;'>{html_table}</div>
    """,height=700,scrolling=True)
