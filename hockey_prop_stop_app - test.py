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
    "NJ":  "NJD", "LA":  "LAK", "SJ":  "SJS", "TB":  "TBL", "ARI": "ARI",
    "ANA": "ANA", "BOS": "BOS", "BUF": "BUF", "CAR": "CAR", "CBJ": "CBJ",
    "CGY": "CGY", "CHI": "CHI", "COL": "COL", "DAL": "DAL", "DET": "DET",
    "EDM": "EDM", "FLA": "FLA", "MIN": "MIN", "MTL": "MTL", "NSH": "NSH",
    "NYI": "NYI", "NYR": "NYR", "OTT": "OTT", "PHI": "PHI", "PIT": "PIT",
    "SEA": "SEA", "STL": "STL", "TOR": "TOR", "VAN": "VAN", "VGK": "VGK",
    "WSH": "WSH", "WPG": "WPG"
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
<p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups with inline logos, instant filters, injuries, and consistency ratings.</p>
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
    except Exception: return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception: return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None: return load_file(file_uploader)
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
        if os.path.exists(p): injuries=load_file(open(p,"rb")); break
    if injuries.empty: injuries=load_file(injuries_file)
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
    try:
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
    except: return []

games=get_todays_games()
if not games:
    st.warning("No games found today.")
    st.stop()

for g in games:
    g["away"] = TEAM_ABBREV_MAP.get(g["away"], g["away"])
    g["home"] = TEAM_ABBREV_MAP.get(g["home"], g["home"])

# ---------------------------------------------------------------
# Run Button / Line Input
# ---------------------------------------------------------------
col_run,col_line=st.columns([3,1])
with col_run: run_model=st.button("🚀 Run Model (All Games)")
with col_line:
    line_test=st.number_input("Line to Test",0.0,10.0,3.5,0.5,key="line_test")
    st.session_state.line_test_val=line_test

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results=[]
    skaters=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster=skaters[[player_col,team_col]].rename(columns={player_col:"player",team_col:"team"}).drop_duplicates("player")
    grouped={n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    # Adjustments
    line_adj={}
    if not lines_df.empty and "line pairings" in lines_df.columns:
        l=lines_df.copy()
        l["games"]=pd.to_numeric(l["games"],errors="coerce").fillna(0)
        l["sog against"]=pd.to_numeric(l["sog against"],errors="coerce").fillna(0)
        l=l.groupby(["line pairings","team"],as_index=False).agg({"games":"sum","sog against":"sum"})
        l["sog_against_per_game"]=np.where(l["games"]>0,l["sog against"]/l["games"],np.nan)
        team_avg=l.groupby("team")["sog_against_per_game"].mean()
        league_avg=team_avg.mean()
        l["line_factor"]=(league_avg/l["sog_against_per_game"]).clip(0.7,1.3)
        line_adj=l.copy()

    goalie_adj={}
    if not goalies_df.empty and {"team","shots against","games"}.issubset(goalies_df.columns):
        g=goalies_df.copy()
        g["shots against"]=pd.to_numeric(g["shots against"],errors="coerce").fillna(0)
        g["games"]=pd.to_numeric(g["games"],errors="coerce").fillna(1)
        g["shots_per_game"]=g["shots against"]/g["games"]
        league_avg_sa=g["shots_per_game"].mean()
        g["goalie_factor"]=(g["shots_per_game"]/league_avg_sa).clip(0.7,1.3)
        goalie_adj=g.groupby("team")["goalie_factor"].mean().to_dict()

    for row in roster.itertuples(index=False):
        player,team=row.player,row.team
        df_p=grouped.get(player.lower(),pd.DataFrame())
        if df_p.empty or "sog" not in df_p.columns: continue

        sog_vals=df_p.groupby(game_col)["sog"].sum().tolist()
        if not sog_vals: continue

        last3, last5, last10 = sog_vals[-3:], sog_vals[-5:], sog_vals[-10:]
        l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)
        
        # --- L10 Consistency Logic ---
        if len(last10) >= 3:
            l10_avg = np.mean(last10)
            l10_std = np.std(last10)
            cv = l10_std / l10_avg if l10_avg > 0 else 1.0
            consistency_score = round(np.clip(10 - (cv * 10), 1, 10), 1)
            stars = "⭐" * int(consistency_score // 2) if consistency_score > 2 else "🌑"
            consist_display = f"{stars} ({consistency_score})"
        else:
            consist_display = "N/A"

        baseline=(0.55*l10)+(0.3*l5)+(0.15*l3)
        trend=(l5-l10)/l10 if l10>0 else 0
        form_flag="🟢 Above Baseline" if trend>0.05 else "🔴 Below Baseline" if trend<-0.05 else "⚪ Neutral"

        line_factor_internal=1.0
        if isinstance(line_adj,pd.DataFrame) and not line_adj.empty:
            last_name=str(player).split()[-1].lower()
            m=line_adj[line_adj["line pairings"].str.contains(last_name,case=False,na=False)]
            if not m.empty:
                line_factor_internal=np.average(m["line_factor"],weights=m["games"])

        opp_team=team_b if team==team_a else team_a
        goalie_factor=goalie_adj.get(opp_team,1.0)
        
        # --- Matchup Heat Rating ---
        heat_multiplier = (1+(goalie_factor-1.0)*0.2)*line_factor_internal
        heat_score = round(np.clip(heat_multiplier * 5, 1, 10), 1)
        heat_flame = "🔥" * int(heat_score // 2) if heat_score > 4 else "🧊"
        matchup_heat = f"{heat_flame} ({heat_score})"

        lam=baseline*heat_multiplier
        poisson_prob=float(np.clip(1-poisson.cdf(np.floor(lam)-1,mu=max(lam,0.01)),0.0001,0.9999))
        
        injury_html=""
        if not injuries_df.empty and {"player","team"}.issubset(injuries_df.columns):
            match=injuries_df[injuries_df["team"].str.lower().str.strip().eq(team.lower()) & 
                              injuries_df["player"].str.lower().str.endswith(player.lower().split()[-1])]
            if not match.empty:
                tooltip=str(match.iloc[0].get("injury note","")).strip()
                injury_html=f"<span style='cursor:pointer;' onclick='alert({json.dumps(html.escape(tooltip))})' title='Injury info'>🚑</span>"

        if "goal" in df_p.columns:
            agg=df_p.groupby(game_col).agg({"sog":"sum","goal":"sum"}).reset_index()
            shooting_pct=agg["goal"].mean()/agg["sog"].mean() if agg["sog"].mean()>0 else 0
            exp_goals=shooting_pct*lam*line_factor_internal
        else: exp_goals,shooting_pct=np.nan,np.nan

        results.append({
            "Player":player,
            "Team":team,
            "Injury":injury_html,
            "Consistency (L10)":consist_display,
            "Matchup Heat":matchup_heat,
            "Trend Score":round(trend,3),
            "Final Projection":round(lam,2),
            "Season Avg":round(np.mean(sog_vals),2),
            "Line Adj":round(line_factor_internal,2),
            "Exp Goals (xG)":round(exp_goals,3) if not np.isnan(exp_goals) else "",
            "Form Indicator":form_flag,
            "L10 Shots":", ".join(map(str,last10))
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Logic & Display
# ---------------------------------------------------------------
if run_model:
    all_tables=[]
    for m in games:
        df=build_model(m["away"],m["home"],skaters_df,shots_df,goalies_df,lines_df,teams_df,injuries_df)
        if not df.empty:
            df["Matchup"]=f"{m['away']}@{m['home']}"
            all_tables.append(df)
    if all_tables:
        st.session_state.results=pd.concat(all_tables,ignore_index=True)
        st.session_state.matchups=games
        st.rerun()

if "results" in st.session_state:
    df=st.session_state.results.copy()
    
    # Matchup Filtering Buttons
    cols = st.columns(3)
    for i, m in enumerate(st.session_state.matchups):
        match_id = f"{m['away']}@{m['home']}"
        with cols[i % 3]:
            if st.button(f"{m['away']} @ {m['home']}", key=f"btn_{i}", use_container_width=True):
                st.session_state.selected_match = None if st.session_state.get("selected_match")==match_id else match_id
                st.session_state.selected_teams = None if not st.session_state.selected_match else {m['away'], m['home']}
                st.rerun()

    if st.session_state.get("selected_teams"):
        df=df[df["Team"].isin(st.session_state.selected_teams)]

    # Formatting
    def color_trend(v):
        if v>0.05: return "<span style='color:#00FF00;font-weight:bold;'>▲</span>"
        elif v<-0.05: return "<span style='color:#FF4B4B;font-weight:bold;'>▼</span>"
        return "<span style='color:#D6D6D6;'>–</span>"

    df["Trend"]=df["Trend Score"].apply(color_trend)
    
    test_line=st.session_state.get("line_test_val", 3.5)
    df["Prob ≥ Line (%)"]=df["Final Projection"].apply(lambda lam:round((1-poisson.cdf(test_line-1,mu=max(lam,0.01)))*100,1))
    
    def get_odds(p):
        p=np.clip(p,0.1,99.9)
        val = -100*(p/100)/(1-p/100) if p>=50 else 100*(1-p/100)/(p/100)
        return f"{'+' if val>0 else ''}{int(round(val))}"
    
    df["Playable Odds"]=df["Prob ≥ Line (%)"].apply(get_odds)

    # Column Ordering & Display
    cols_to_show = ["Player", "Team", "Injury", "Consistency (L10)", "Matchup Heat", "Trend", "Final Projection", 
                    "Prob ≥ Line (%)", "Playable Odds", "Season Avg", "Exp Goals (xG)", "Form Indicator", "L10 Shots"]
    
    html_table=df[cols_to_show].to_html(index=False,escape=False)
    
    components.html(f"""
    <style>
    table {{ width:100%; border-collapse:collapse; font-family:'Source Sans Pro',sans-serif; color:#D6D6D6; }}
    th {{ background-color:#0A3A67; color:#FFFFFF; padding:6px; text-align:center; position:sticky; top:0; border-bottom:2px solid #1E5A99; }}
    td:first-child, th:first-child {{ position:sticky; left:0; background-color:#1E5A99; color:#FFFFFF; font-weight:bold; }}
    td {{ background-color:#0F2743; padding:4px; text-align:center; border-bottom:1px solid #142F52; }}
    tr:nth-child(even) td {{ background-color:#142F52; }}
    td:nth-child(4) {{ color:#FFD700; font-weight:bold; }} /* Consistency Col */
    td:nth-child(5) {{ color:#FF4500; font-weight:bold; }} /* Matchup Heat Col */
    td:nth-child(11) {{ color:#7FFF00; font-weight:bold; }} /* xG Col */
    </style>
    <div style='overflow-x:auto;height:650px;'>{html_table}</div>
    """, height=700, scrolling=True)

    st.download_button("💾 Download CSV", df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")
