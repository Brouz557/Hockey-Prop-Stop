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
    "NJ": "NJD","LA": "LAK","SJ": "SJS","TB": "TBL",
    "ARI":"ARI","ANA":"ANA","BOS":"BOS","BUF":"BUF",
    "CAR":"CAR","CBJ":"CBJ","CGY":"CGY","CHI":"CHI",
    "COL":"COL","DAL":"DAL","DET":"DET","EDM":"EDM",
    "FLA":"FLA","MIN":"MIN","MTL":"MTL","NSH":"NSH",
    "NYI":"NYI","NYR":"NYR","OTT":"OTT","PHI":"PHI",
    "PIT":"PIT","SEA":"SEA","STL":"STL","TOR":"TOR",
    "VAN":"VAN","VGK":"VGK","WSH":"WSH","WPG":"WPG"
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
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load Data (same behavior as source of truth)
# ---------------------------------------------------------------
def safe_read(path):
    return pd.read_excel(path) if os.path.exists(path) else pd.DataFrame()

skaters_df = safe_read("Skaters.xlsx")
shots_df   = safe_read("SHOT DATA.xlsx")
goalies_df = safe_read("GOALTENDERS.xlsx")
lines_df   = safe_read("LINE DATA.xlsx")
teams_df   = safe_read("TEAMS.xlsx")
injuries_df= safe_read("injuries.xlsx")

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    df.columns = df.columns.str.lower().str.strip()

team_col   = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"

shots_df = shots_df.rename(
    columns={next(c for c in shots_df.columns if "player" in c or "name" in c):"player"}
)
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c and "id" in c)

# ---------------------------------------------------------------
# Matchups (ESPN)
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_games():
    url="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    r=requests.get(url,timeout=10).json()
    games=[]
    for e in r.get("events",[]):
        c=e["competitions"][0]["competitors"]
        games.append({
            "away": TEAM_ABBREV_MAP.get(c[0]["team"]["abbreviation"]),
            "home": TEAM_ABBREV_MAP.get(c[1]["team"]["abbreviation"]),
            "away_logo": c[0]["team"]["logo"],
            "home_logo": c[1]["team"]["logo"]
        })
    return games

games=get_games()

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
col_run,col_line=st.columns([3,1])
with col_run:
    run_model=st.button("🚀 Run Model (All Games)")
with col_line:
    line_test=st.number_input("Line to Test",0.0,10.0,3.5,0.5)
    st.session_state.setdefault("line_test_val",line_test)
    if line_test!=st.session_state.line_test_val:
        st.session_state.line_test_val=line_test

# ---------------------------------------------------------------
# MODEL (CORSI ADDED, STRUCTURE PRESERVED)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a,team_b):

    results=[]
    skaters=skaters_df[skaters_df[team_col].isin([team_a,team_b])]

    # IMPORTANT: one row per player ONLY
    roster=skaters[[player_col,team_col]].drop_duplicates(subset=[player_col])

    league_player_corsi=skaters_df["on ice corsi"].mean()
    league_team_cp=teams_df["corsi%"].mean()

    for r in roster.itertuples(index=False):
        player,team=r
        dfp=shots_df[shots_df["player"].str.lower()==player.lower()]
        if dfp.empty: continue

        sog_vals=dfp.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals)<3: continue

        l3,l5,l10=np.mean(sog_vals[-3:]),np.mean(sog_vals[-5:]),np.mean(sog_vals[-10:])
        baseline=0.55*l10+0.3*l5+0.15*l3
        trend=(l5-l10)/l10 if l10>0 else 0

        # ---- Corsi factors (SAFE) ----
        pc=skaters_df.loc[skaters_df[player_col]==player,"on ice corsi"].mean()
        player_corsi_factor=np.clip(pc/league_player_corsi,0.85,1.20)

        opp=team_b if team==team_a else team_a
        team_cp=teams_df.loc[teams_df["team"]==team,"corsi%"].mean()
        opp_cp=teams_df.loc[teams_df["team"]==opp,"corsi%"].mean()
        pace_factor=np.clip(((team_cp+opp_cp)/2)/league_team_cp,0.92,1.08)

        lam=baseline*(1+0.15*(player_corsi_factor-1))*pace_factor
        lam=np.clip(lam,baseline*0.6,baseline*1.4)

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":round(lam,2),
            "Trend Score":round(trend,3),
            "Season Avg":round(np.mean(sog_vals),2),
            "Line Adj":round(player_corsi_factor,2)
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model:
    tables=[]
    for g in games:
        df=build_model(g["away"],g["home"])
        if not df.empty:
            df["Matchup"]=f"{g['away']}@{g['home']}"
            tables.append(df)
    if tables:
        st.session_state.results=pd.concat(tables,ignore_index=True)

# ---------------------------------------------------------------
# Display (HTML TABLE – COLORS RESTORED)
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results.copy()
    test_line=st.session_state.line_test_val

    df["Prob ≥ Line (%)"]=df["Final Projection"].apply(
        lambda x:round((1-poisson.cdf(test_line-1,mu=max(x,0.01)))*100,1)
    )

    cols=["Player","Team","Final Projection","Prob ≥ Line (%)","Season Avg","Line Adj","Trend Score"]
    html_table=df[cols].sort_values("Final Projection",ascending=False)\
        .to_html(index=False,escape=False)

    components.html(f"""
    <style>
    table {{
        width:100%;
        border-collapse:collapse;
        color:#D6D6D6;
        font-family:Source Sans Pro;
    }}
    th {{
        background:#0A3A67;
        color:white;
        padding:6px;
        position:sticky;
        top:0;
    }}
    td {{
        background:#0F2743;
        padding:4px;
        text-align:center;
    }}
    tr:nth-child(even) td {{background:#142F52;}}
    </style>
    <div style='height:650px;overflow:auto'>{html_table}</div>
    """,height=700)
