# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (Search + Filters)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("🧪 TEST MODE — Sandbox version")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>Instant filters • Matchups • Injuries • xG</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar Uploads
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("Shot Data", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("Goalies", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("Lines", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("Teams", type=["xlsx","csv"])
injuries_file= st.sidebar.file_uploader("Injuries", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def load_file(f):
    if not f: return pd.DataFrame()
    try:
        return pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)
    except:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_all():
    skaters = load_file(skaters_file)
    shots   = load_file(shots_file)
    goalies = load_file(goalies_file)
    lines   = load_file(lines_file)
    teams   = load_file(teams_file)
    injuries= load_file(injuries_file)
    for df in [skaters,shots,goalies,lines,teams,injuries]:
        if not df.empty:
            df.columns = df.columns.str.lower().str.strip()
    return skaters,shots,goalies,lines,teams,injuries

skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all()
if skaters_df.empty or shots_df.empty:
    st.stop()

team_col   = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"
game_col   = next(c for c in shots_df.columns if "game" in c and "id" in c)

shots_df["player"] = shots_df["player"].astype(str)

# ---------------------------------------------------------------
# ESPN Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_games():
    url="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    r=requests.get(url,timeout=10).json()
    games=[]
    for e in r.get("events",[]):
        c=e["competitions"][0]["competitors"]
        games.append({
            "away":c[0]["team"]["abbreviation"],
            "home":c[1]["team"]["abbreviation"],
            "away_logo":c[0]["team"]["logo"],
            "home_logo":c[1]["team"]["logo"]
        })
    return games

games=get_games()

# ---------------------------------------------------------------
# Run Controls
# ---------------------------------------------------------------
run_model = st.button("🚀 Run Model")
line_test = st.sidebar.slider("Line to Test",0.0,10.0,3.5,0.5)

# ---------------------------------------------------------------
# Poisson Helper
# ---------------------------------------------------------------
def poisson_geq(k, mu):
    return float(np.clip(1 - poisson.cdf(k - 1, mu), 0.0001, 0.9999))

# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b):
    results=[]
    sk=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster=sk[[player_col,team_col]].drop_duplicates()

    grouped={n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    goalie_factor=goalies_df.groupby("team")["shots against"].mean()
    goalie_factor/=goalie_factor.mean()

    for _,r in roster.iterrows():
        player,team=r[player_col],r[team_col]
        df_p=grouped.get(player.lower())
        if df_p is None: continue

        sog=df_p.groupby(game_col)["sog"].sum().tolist()
        if len(sog)<3: continue

        l3,l5,l10=np.mean(sog[-3:]),np.mean(sog[-5:]),np.mean(sog[-10:])
        lam=0.55*l10+0.3*l5+0.15*l3
        lam*=goalie_factor.get(team,1.0)

        prob=poisson_geq(int(np.ceil(line_test)),lam)

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":round(lam,2),
            "Prob ≥ Line (%)":round(prob*100,1),
            "Season Avg":round(np.mean(sog),2),
            "Form Indicator":"🟢 Above Baseline" if l5>l10 else "🔴 Below Baseline"
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if run_model:
    dfs=[]
    for g in games:
        df=build_model(g["away"],g["home"])
        if not df.empty:
            dfs.append(df)
    st.session_state.results=pd.concat(dfs,ignore_index=True)

# ---------------------------------------------------------------
# SEARCH & FILTERS
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results.copy()

    st.sidebar.markdown("## 🔎 Search & Filters")
    name=st.sidebar.text_input("Player name").lower()
    teams=st.sidebar.multiselect("Teams",sorted(df["Team"].unique()))
    min_proj=st.sidebar.slider("Min Projection",0.0,float(df["Final Projection"].max()),0.0,0.5)
    min_prob=st.sidebar.slider("Min Prob ≥ Line",0.0,100.0,0.0,5.0)
    form=st.sidebar.multiselect("Form",["🟢 Above Baseline","🔴 Below Baseline"])
    top_n=st.sidebar.selectbox("Top N",[None,10,20,30,50])

    if name:
        df=df[df["Player"].str.lower().str.contains(name)]
    if teams:
        df=df[df["Team"].isin(teams)]
    df=df[df["Final Projection"]>=min_proj]
    df=df[df["Prob ≥ Line (%)"]>=min_prob]
    if form:
        df=df[df["Form Indicator"].isin(form)]
    df=df.sort_values("Final Projection",ascending=False)
    if top_n:
        df=df.head(top_n)

    st.markdown(f"### Showing {len(df)} Players")

    html=df.to_html(index=False,escape=False)
    components.html(f"""
    <style>
    table {{width:100%;color:#D6D6D6;border-collapse:collapse}}
    th {{background:#0A3A67;position:sticky;top:0}}
    td {{background:#0F2743;padding:4px;text-align:center}}
    </style>
    <div style='height:650px;overflow:auto'>{html}</div>
    """,height=700)
