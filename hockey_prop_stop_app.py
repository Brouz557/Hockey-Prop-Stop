# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — L5 Probability Update (Safe injury popups, unchanged layout)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, datetime, pytz, subprocess, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown("""
<h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
<p style='text-align:center;color:#BFC0C0;'>
Team-vs-Team matchup analytics with blended regression and L5-based probabilities
</p>
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
injuries_file = st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def load_file(f):
    if not f: return pd.DataFrame()
    try:
        return pd.read_excel(f) if f.name.lower().endswith(".xlsx") else pd.read_csv(f)
    except Exception:
        return pd.DataFrame()

def safe_read(p):
    if not os.path.exists(p): return pd.DataFrame()
    try:
        return pd.read_excel(p) if p.lower().endswith(".xlsx") else pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def load_data(uploader, default_path):
    return load_file(uploader) if uploader is not None else safe_read(default_path)

# ---------------------------------------------------------------
# Cached data load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    base = [".","data","/mount/src/hockey-prop-stop/data"]
    def find(name):
        for b in base:
            full = os.path.join(b,name)
            if os.path.exists(full): return full
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        skaters = load_data(skaters_file, find("Skaters.xlsx") or "Skaters.xlsx")
        shots   = load_data(shots_file,   find("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
        goalies = load_data(goalies_file, find("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
        lines   = load_data(lines_file,   find("LINE DATA.xlsx") or "LINE DATA.xlsx")
        teams   = load_data(teams_file,   find("TEAMS.xlsx") or "TEAMS.xlsx")

        inj = pd.DataFrame()
        for p in ["injuries.xlsx","Injuries.xlsx","./injuries.xlsx","data/injuries.xlsx",
                  "/mount/src/hockey-prop-stop/injuries.xlsx"]:
            if os.path.exists(p):
                inj = load_file(open(p,"rb")); break
        if inj.empty: inj = load_file(injuries_file)
        if not inj.empty:
            inj.columns = inj.columns.str.lower().str.strip()
            if "player" in inj.columns:
                inj["player"] = inj["player"].astype(str).str.strip().str.lower()
    return skaters, shots, goalies, lines, teams, inj

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file)

if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()
st.success("✅ Data loaded successfully.")

# ---------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------
def get_git_time():
    tz = pytz.timezone("America/Chicago")
    for f in ["data/SHOT DATA.xlsx","/mount/src/hockey-prop-stop/data/SHOT DATA.xlsx","SHOT DATA.xlsx"]:
        if os.path.exists(f):
            try:
                s = subprocess.check_output(
                    ["git","log","-1","--format=%cd","--date=iso",f],
                    stderr=subprocess.DEVNULL).decode().strip()
                if s:
                    t = datetime.datetime.fromisoformat(s.replace("Z","+00:00"))
                    return t.astimezone(tz).strftime("%Y-%m-%d %I:%M %p CST")
            except: pass
    return None
st.markdown(f"🕒 **Data last updated:** {get_git_time() or 'Unknown'}")

# ---------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------
for d in [skaters_df,shots_df,goalies_df,lines_df,teams_df]:
    if not d.empty: d.columns = d.columns.str.lower().str.strip()
team_col = next((c for c in skaters_df.columns if "team" in c),None)
player_col = "name" if "name" in skaters_df.columns else None
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c),None)
shots_df = shots_df.rename(columns={player_col_shots:"player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c),None)

# ---------------------------------------------------------------
# Team selectors
# ---------------------------------------------------------------
teams = sorted(skaters_df[team_col].dropna().unique().tolist())
c1,c2 = st.columns(2)
with c1: team_a = st.selectbox("Select Team A",teams)
with c2: team_b = st.selectbox("Select Team B",[t for t in teams if t!=team_a])
st.markdown("---")

# ---------------------------------------------------------------
# Build model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, inj_df):
    res = []
    skaters = skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster = skaters[[player_col,team_col]].rename(columns={player_col:"player",team_col:"team"}).drop_duplicates("player")
    grouped = {n.lower():g.sort_values(game_col) for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    for row in roster.itertuples(index=False):
        player,team = row.player,row.team
        dfp = grouped.get(player.lower(),pd.DataFrame())
        if dfp.empty: continue
        sogs = dfp.groupby(game_col)[["sog","goal"]].sum().reset_index().sort_values(game_col)
        vals = sogs["sog"].tolist()
        if not vals: continue

        last3,last5,last10 = vals[-3:],vals[-5:],vals[-10:]
        l3,l5,l10 = np.mean(last3),np.mean(last5),np.mean(last10)
        trend = (l5-l10)/l10 if l10>0 else 0
        line,prob = round(l5,2),1-poisson.cdf(np.floor(l5)-1,mu=l5)
        p = min(max(prob,0.001),0.999)
        odds = -100*(p/(1-p)) if p>=0.5 else 100*((1-p)/p)
        implied = f"{'+' if odds>0 else ''}{int(odds)}"

        form_flag="⚪ Neutral Form"
        # Injury column (robust escaping)
        inj_html=""
        if not inj_df.empty and {"player","team"}.issubset(inj_df.columns):
            last_name = player.lower().split()[-1]
            t_lower = team.lower()
            match = inj_df[inj_df["team"].str.lower().str.contains(t_lower,na=False)
                           & inj_df["player"].str.lower().str.contains(last_name,na=False)]
            if not match.empty:
                note  = str(match.iloc[0].get("injury note","")).strip()
                typ   = str(match.iloc[0].get("injury type","")).strip()
                date  = str(match.iloc[0].get("date of injury","")).strip()
                tooltip_raw = "\n".join([p for p in [typ,note,date] if p]) or "Injury info unavailable"
                # escape safely for HTML + JS
                safe = html.escape(tooltip_raw)
                inj_html = f"<span style='cursor:pointer;' onclick='alert({json.dumps(safe)})' title='Tap or click for injury info'>🚑</span>"

        res.append({
            "Player":player,"Team":team,"Injury":inj_html,
            "Final Projection":line,
            "Prob ≥ Projection (%) L5":round(p*100,1),
            "Playable Odds":implied,
            "Trend Score":round(trend,3),
        })
    return pd.DataFrame(res)

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if st.button("🚀 Run Model"):
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")
    df = build_model(team_a,team_b,skaters_df,shots_df,goalies_df,lines_df,teams_df,injuries_df)
    if "Injury" not in df.columns: df["Injury"]=""
    st.session_state.results_raw = df.sort_values("Final Projection",ascending=False).reset_index(drop=True)
    st.success("✅ Model built successfully!")

# ---------------------------------------------------------------
# Display
# ---------------------------------------------------------------
if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
    df = st.session_state.results_raw.copy()
    def trend_color(v):
        if pd.isna(v): return "–"
        v=max(min(v,0.5),-0.5); n=v+0.5
        if n<0.5: r,g,b=255,int(255*(n*2)),0
        else: r,g,b=int(255*(1-(n-0.5)*2)),255,0
        t="▲" if v>0.05 else ("▼" if v<-0.05 else "–")
        txt="#000" if abs(v)<0.2 else "#fff"
        return f"<div style='background:rgb({r},{g},{b});color:{txt};border-radius:6px;padding:4px 8px;text-align:center;font-weight:600'>{t}</div>"
    df["Trend"]=df["Trend Score"].apply(trend_color)

    vis=df[["Player","Team","Injury","Trend","Final Projection","Prob ≥ Projection (%) L5","Playable Odds"]]
    html_table=vis.to_html(index=False,escape=False)
    components.html(f"""
    <style>
    div.scrollable-table{{overflow:auto;height:600px}}
    table{{width:100%;border-collapse:collapse;font-family:'Source Sans Pro',sans-serif;color:#f0f0f0}}
    th{{background:#00B140;color:white;position:sticky;top:0;padding:6px;text-align:center}}
    td:first-child,th:first-child{{position:sticky;left:0;background:#00B140;color:white;font-weight:bold}}
    td{{background:#1e1e1e;color:#f0f0f0;padding:4px;text-align:center}}
    tr:nth-child(even) td{{background:#2a2a2a}}
    </style>
    <div class='scrollable-table'>{html_table}</div>
    """,height=620,scrolling=True)

    # Save + Download
    st.markdown("---")
    st.subheader("💾 Save or Download Projections")
    sel_date = st.date_input("Select game date:",datetime.date.today())
    if st.button("💾 Save Projections for Selected Date"):
        df2=df.copy()
        df2["Date_Game"]=sel_date.strftime("%Y-%m-%d")
        df2["Matchup"]=f"{team_a} vs {team_b}"
        os.makedirs("projections",exist_ok=True)
        fn=f"{team_a}_vs_{team_b}_{sel_date.strftime('%Y-%m-%d')}.csv"
        path=os.path.join("projections",fn)
        df2.to_csv(path,index=False)
        st.success(f"✅ Saved projections to **{path}**")
        st.download_button("📥 Download Projections CSV",df2.to_csv(index=False).encode("utf-8"),fn,"text/csv")
