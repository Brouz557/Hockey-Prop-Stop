# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (Instant Filter + Logos + Injuries + xG)
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

st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("Production Version")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups with inline logos, instant filters, injuries, and expected goals.</p>
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

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty:
        df.columns=df.columns.str.lower().str.strip()

team_col=next((c for c in skaters_df.columns if "team" in c),None)
player_col="name"
shots_df=shots_df.rename(columns={
    next((c for c in shots_df.columns if "player" in c or "name" in c),"player"):"player"
})
shots_df["player"]=shots_df["player"].astype(str).str.strip()
game_col=next((c for c in shots_df.columns if "game" in c and "id" in c),None)

# ---------------------------------------------------------------
# 🔥 POSITION MATCHUP SECTION (ADDED)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_position_matchups(shots_df, skaters_df, game_col, team_col):
    if shots_df.empty or skaters_df.empty or not game_col:
        return {}
    if "position" not in skaters_df.columns:
        return {}

    def norm(x):
        return str(x).lower().replace(".","").replace(",","").strip()

    shots=shots_df.copy()
    skaters=skaters_df.copy()

    shots["p"]=shots["player"].apply(norm)
    skaters["p"]=skaters["name"].apply(norm)

    merged=shots.merge(
        skaters[["p","position",team_col]],
        on="p",how="left"
    )

    if "sog" not in merged.columns:
        return {}

    merged=merged.dropna(subset=["position","sog",game_col,team_col])

    per_game=merged.groupby([team_col,"position",game_col])["sog"].sum().reset_index()
    avg=per_game.groupby([team_col,"position"])["sog"].mean().reset_index()

    league_avg=avg.groupby("position")["sog"].mean().to_dict()
    avg["pos_factor"]=avg.apply(
        lambda r:r["sog"]/league_avg.get(r["position"],r["sog"]),axis=1
    )
    avg["pos_factor"]=avg["pos_factor"].clip(0.85,1.20)

    return {(r[team_col],r["position"]):r["pos_factor"] for _,r in avg.iterrows()}

pos_matchup_adj = build_position_matchups(
    shots_df, skaters_df, game_col, team_col
)

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
                "away":TEAM_ABBREV_MAP.get(away["team"]["abbreviation"]),
                "home":TEAM_ABBREV_MAP.get(home["team"]["abbreviation"]),
                "away_logo":away["team"]["logo"],
                "home_logo":home["team"]["logo"]
            })
    return games

games=get_todays_games()
if not games:
    st.warning("No games found today.")
    st.stop()

run_model=st.button("🚀 Run Model (All Games)")

# ---------------------------------------------------------------
# Build Model (ORIGINAL + MATCHUP)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results=[]
    skaters=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster=skaters[[player_col,team_col]].rename(
        columns={player_col:"player",team_col:"team"}
    ).drop_duplicates("player")

    grouped={n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    goalie_adj={}
    if {"team","shots against","games"}.issubset(goalies_df.columns):
        g=goalies_df.copy()
        g["shots against"]=pd.to_numeric(g["shots against"],errors="coerce").fillna(0)
        g["games"]=pd.to_numeric(g["games"],errors="coerce").fillna(1)
        g["shots_per_game"]=g["shots against"]/g["games"]
        g["goalie_factor"]=(g["shots_per_game"]/g["shots_per_game"].mean()).clip(0.7,1.3)
        goalie_adj=g.groupby("team")["goalie_factor"].mean().to_dict()

    for row in roster.itertuples(index=False):
        player,team=row.player,row.team
        df_p=grouped.get(player.lower(),pd.DataFrame())
        if df_p.empty or "sog" not in df_p.columns: continue

        sog_vals=df_p.groupby(game_col)["sog"].sum().tolist()
        if not sog_vals: continue

        l3,l5,l10=np.mean(sog_vals[-3:]),np.mean(sog_vals[-5:]),np.mean(sog_vals[-10:])
        baseline=0.55*l10+0.3*l5+0.15*l3

        opp=team_b if team==team_a else team_a
        goalie_factor=goalie_adj.get(opp,1.0)

        player_pos=skaters_df.loc[
            skaters_df[player_col].eq(player),"position"
        ].iloc[0] if "position" in skaters_df.columns else None

        pos_factor=pos_matchup_adj.get((opp,player_pos),1.0)

        lam=baseline*(1+(goalie_factor-1)*0.2)*pos_factor

        results.append({
            "Player":player,
            "Team":team,
            "Pos Matchup":round(pos_factor,2),
            "Final Projection":round(lam,2),
            "Season Avg":round(np.mean(sog_vals),2)
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model + Display
# ---------------------------------------------------------------
if run_model:
    tables=[]
    for m in games:
        df=build_model(
            m["away"],m["home"],
            skaters_df,shots_df,goalies_df,
            lines_df,teams_df,injuries_df
        )
        if not df.empty:
            df["Matchup"]=f'{m["away"]}@{m["home"]}'
            tables.append(df)

    if tables:
        st.dataframe(pd.concat(tables,ignore_index=True),use_container_width=True)
    else:
        st.warning("⚠️ No valid data generated.")
