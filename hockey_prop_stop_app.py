# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Full Enhanced + Persistent Version
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, io, contextlib, datetime
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with goalie, line, and trend visualization
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx", "csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx", "csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx", "csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx", "csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx", "csv"])

# ---------------------------------------------------------------
# File Helpers
# ---------------------------------------------------------------
def load_file(file):
    if not file: return pd.DataFrame()
    try:
        if file.name.lower().endswith(".csv"): return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception: return pd.DataFrame()

def safe_read(path):
    try:
        if not path or not os.path.exists(path): return pd.DataFrame()
        if path.lower().endswith(".csv"): return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception: return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None: return load_file(file_uploader)
    return safe_read(default_path)

# ---------------------------------------------------------------
# Cache + Load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
    def find_file(name):
        for p in base_paths:
            fp = os.path.join(p, name)
            if os.path.exists(fp): return fp
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        sk = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
        sh = load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
        go = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
        li = load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
        te = load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")
    return sk, sh, go, li, te

# ---------------------------------------------------------------
# Timestamp Helpers
# ---------------------------------------------------------------
def get_excel_lastupdated_tag(path):
    try:
        df = pd.read_excel(path, header=None, nrows=10)
        for _, r in df.iterrows():
            for c in r:
                if isinstance(c, str) and c.lower().startswith("lastupdated:"):
                    return c.split(":",1)[1].strip()
    except Exception: pass
    return None

def get_file_last_modified(path):
    try:
        t = os.path.getmtime(path)
        dt = datetime.datetime.fromtimestamp(t, ZoneInfo("America/Chicago"))
        return dt.strftime("%B %d, %Y — %I:%M %p CT")
    except Exception: return None

# ---------------------------------------------------------------
# Load Once + Persist
# ---------------------------------------------------------------
if "data_loaded" not in st.session_state or st.sidebar.button("🔄 Refresh Data"):
    sk, sh, go, li, te = load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file)
    st.session_state.update({
        "data_loaded": True,
        "skaters_df": sk, "shots_df": sh,
        "goalies_df": go, "lines_df": li, "teams_df": te
    })

skaters_df = st.session_state.skaters_df
shots_df   = st.session_state.shots_df
goalies_df = st.session_state.goalies_df
lines_df   = st.session_state.lines_df

# Timestamp display
paths = ["Skaters.xlsx","./data/Skaters.xlsx","/mount/src/hockey-prop-stop/data/Skaters.xlsx"]
path = next((p for p in paths if os.path.exists(p)), None)
tag = get_excel_lastupdated_tag(path) if path else None
stamp = tag or get_file_last_modified(path) or "(timestamp unavailable)"
st.success(f"✅ Data loaded successfully (Last updated: {stamp})")

# ---------------------------------------------------------------
# Data Prep
# ---------------------------------------------------------------
skaters_df.columns = skaters_df.columns.str.lower().str.strip()
shots_df.columns   = shots_df.columns.str.lower().str.strip()
if not goalies_df.empty: goalies_df.columns = goalies_df.columns.str.lower().str.strip()
if not lines_df.empty:   lines_df.columns   = lines_df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name"
sog_col = next((c for c in shots_df.columns if "sog" in c), None)
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)

# ---------------------------------------------------------------
# Team Selection
# ---------------------------------------------------------------
teams = sorted(skaters_df[team_col].dropna().unique().tolist())
c1,c2 = st.columns(2)
with c1: team_a = st.selectbox("Select Team A", teams, key="teamA")
with c2: team_b = st.selectbox("Select Team B", [t for t in teams if t != team_a], key="teamB")

st.markdown("---")
run_model = st.button("🚀 Run Model")

# ---------------------------------------------------------------
# Keep previous model unless rerun
# ---------------------------------------------------------------
if "results_df" not in st.session_state: st.session_state.results_df = None

# ---------------------------------------------------------------
# Model Build
# ---------------------------------------------------------------
if run_model:
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

    goalie_adj, rebound_rate = {}, {}
    if not goalies_df.empty:
        g = goalies_df.copy()
        g = g[g["situation"].str.lower()=="all"]
        g["games"]=pd.to_numeric(g["games"],errors="coerce").fillna(0)
        g["unblocked attempts"]=pd.to_numeric(g["unblocked attempts"],errors="coerce").fillna(0)
        g["rebounds"]=pd.to_numeric(g["rebounds"],errors="coerce").fillna(0)
        g["shots_allowed_per_game"]=np.where(g["games"]>0,g["unblocked attempts"]/g["games"],np.nan)
        g["rebound_rate"]=np.where(g["unblocked attempts"]>0,g["rebounds"]/g["unblocked attempts"],0)
        tavg=g.groupby("team")["shots_allowed_per_game"].mean(); lavg=tavg.mean()
        goalie_adj=(lavg/tavg).to_dict(); rebound_rate=g.groupby("team")["rebound_rate"].mean().to_dict()

    line_adj={}
    if not lines_df.empty:
        l=lines_df.copy()
        l["games"]=pd.to_numeric(l["games"],errors="coerce").fillna(0)
        l["sog against"]=pd.to_numeric(l["sog against"],errors="coerce").fillna(0)
        l=l.groupby(["line pairings","team"],as_index=False).agg({"games":"sum","sog against":"sum"})
        l["sog_against_per_game"]=np.where(l["games"]>0,l["sog against"]/l["games"],np.nan)
        tavg=l.groupby("team")["sog_against_per_game"].mean(); lavg=tavg.mean()
        l["line_factor"]=(lavg/l["sog_against_per_game"]).clip(0.7,1.3)
        line_adj=l.copy()

    roster=(skaters_df[skaters_df[team_col].isin([team_a,team_b])][[player_col,team_col]]
        .rename(columns={player_col:"player",team_col:"team"}).drop_duplicates("player").reset_index(drop=True))

    shots_df=shots_df.rename(columns={player_col_shots:"player",game_col:"gameid",sog_col:"sog"})
    shots_df["player"]=shots_df["player"].astype(str).str.strip(); roster["player"]=roster["player"].astype(str).str.strip()
    grouped={n.lower():g.sort_values("gameid") for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    res=[]
    for r in roster.itertuples(index=False):
        p,t=r.player,r.team; d=grouped.get(p.lower(),pd.DataFrame())
        if d.empty: continue
        sogs=d.groupby("gameid")["sog"].sum().sort_index().tolist()
        l3,l5,l10=sogs[-3:],sogs[-5:],sogs[-10:]
        l3m,l5m,l10m=np.mean(l3),np.mean(l5),np.mean(l10)
        seas=np.mean(sogs); trend=(l3m-l10m)/l10m if l10m else 0
        base=0.5*l3m+0.3*l5m+0.2*l10m
        opp=team_b if t==team_a else team_a
        gf=np.clip(goalie_adj.get(opp,1.0),0.7,1.3); rf=rebound_rate.get(opp,0.0)
        lf=1.0
        if not line_adj.empty:
            last=str(p).split()[-1].lower()
            m=line_adj[line_adj["line pairings"].str.contains(last,case=False,na=False)]
            if not m.empty: lf=np.average(m["line_factor"],weights=m["games"])
            lf=np.clip(lf,0.7,1.3)
        adj=base*(0.7+0.3*gf)*(0.7+0.3*lf)*(1+rf*0.1)
        adj=round(max(0,adj),2)
        l10f=("<br>".join([", ".join(map(str,l10[:5])),", ".join(map(str,l10[5:]))])
               if len(l10)>5 else ", ".join(map(str,l10)))
        res.append({
            "Player":p,"Team":t,"Season Avg":round(seas,2),
            "L3 Shots":", ".join(map(str,l3)),"L5 Shots":", ".join(map(str,l5)),
            "L10 Shots":l10f,"Trend Score":round(trend,3),
            "Base Projection":round(base,2),"Goalie Adj":round(gf,2),
            "Line Adj":round(lf,2),"Final Projection":adj
        })

    df=pd.DataFrame(res)
    a,m=df["Final Projection"].mean(),df["Final Projection"].std()
    df["Matchup Rating"]=df["Final Projection"].apply(lambda v:"Strong" if v>=a+m else "Moderate" if v>=a else "Weak")

    def trend_color(v):
        if pd.isna(v): return "<div style='background:#E0E0E0;'>–</div>"
        v=max(min(v,0.5),-0.5); norm=(v+0.5)
        if norm<0.5: r,g,b=255,int(255*(norm*2)),0
        else: r,g,b=int(255*(1-(norm-0.5)*2)),255,0
        c=f"rgb({r},{g},{b})"; sym="▲" if v>0.05 else "▼" if v<-0.05 else "–"
        return f"<div style='background:{c};color:#000;font-weight:600;border-radius:6px;padding:3px 6px;'>{sym}</div>"

    df["Trend"]=df["Trend Score"].apply(trend_color)

    logos={"Toronto Maple Leafs":"TOR","Vancouver Canucks":"VAN","Edmonton Oilers":"EDM",
           "Calgary Flames":"CGY","Montreal Canadiens":"MTL","Ottawa Senators":"OTT",
           "Boston Bruins":"BOS","New York Rangers":"NYR","New York Islanders":"NYI",
           "Philadelphia Flyers":"PHI","Pittsburgh Penguins":"PIT","Chicago Blackhawks":"CHI",
           "Colorado Avalanche":"COL","Dallas Stars":"DAL","Vegas Golden Knights":"VGK",
           "Los Angeles Kings":"LAK","San Jose Sharks":"SJS","Seattle Kraken":"SEA",
           "Detroit Red Wings":"DET","Tampa Bay Lightning":"TBL","Florida Panthers":"FLA",
           "Nashville Predators":"NSH","Washington Capitals":"WSH","Buffalo Sabres":"BUF",
           "St. Louis Blues":"STL","Winnipeg Jets":"WPG","Minnesota Wild":"MIN",
           "Anaheim Ducks":"ANA","Arizona Coyotes":"ARI","Columbus Blue Jackets":"CBJ",
           "New Jersey Devils":"NJD"}
    def logo(team):
        a=logos.get(team); 
        return f"<img src='https://assets.nhle.com/logos/nhl/svg/{a}_light.svg' width='26' style='vertical-align:middle;margin-right:6px;'> {team}" if a else team
    df["Team"]=df["Team"].apply(logo)

    st.session_state.results_df=df

# ---------------------------------------------------------------
# Display
# ---------------------------------------------------------------
if st.session_state.results_df is not None:
    df=st.session_state.results_df.sort_values("Final Projection",ascending=False)
    display_cols=["Player","Team","Trend","Final Projection","Season Avg","Matchup Rating",
                  "L3 Shots","L5 Shots","L10 Shots","Base Projection","Goalie Adj","Line Adj"]
    dfv=df[[c for c in display_cols if c in df.columns]]
    st.markdown(f"### 📊 {team_a} vs {team_b} — Player Projections (Adjusted)")
    st.markdown(dfv.to_html(index=False,escape=False),unsafe_allow_html=True)
