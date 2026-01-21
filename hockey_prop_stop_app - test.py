# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (Full ESPN Matchups)
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
st.warning("🧪 TEST MODE — Sandbox version. Changes here will NOT affect production.")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
    <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>Live matchup analytics — full model with form, line, and trend.</p>
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
    if file_uploader is not None: return load_file(file_uploader)
    return safe_read(default_path)

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
        if injuries.empty: injuries = load_file(injuries_file)
        if not injuries.empty:
            injuries.columns = injuries.columns.str.lower().str.strip()
            if "player" in injuries.columns:
                injuries["player"] = injuries["player"].astype(str).str.strip().str.lower()
    return skaters, shots, goalies, lines, teams, injuries

skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()
st.success("✅ Data loaded successfully.")

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns = df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
shots_df = shots_df.rename(columns={player_col_shots: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)

# ---------------------------------------------------------------
# ESPN Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_todays_matchups():
    url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    data = requests.get(url).json()
    games = []
    for g in data.get("events", []):
        if "competitions" in g:
            comp = g["competitions"][0]
            if "competitors" in comp and len(comp["competitors"]) == 2:
                away = comp["competitors"][0]
                home = comp["competitors"][1]
                games.append({
                    "away": away["team"]["abbreviation"],
                    "home": home["team"]["abbreviation"],
                    "away_logo": away["team"]["logo"],
                    "home_logo": home["team"]["logo"]
                })
    return games

games = get_todays_matchups()
if not games:
    st.warning("No NHL games found for today.")
else:
    st.subheader("🏒 Today's Matchups")
    cols = st.columns(min(4, len(games)))
    for i, m in enumerate(games):
        col = cols[i % len(cols)]
        html_block = f"""
        <div style='text-align:center; padding:8px; border-radius:10px; background-color:#0F2743;
                    transition:0.3s; border:2px solid transparent;'
             onmouseover="this.style.borderColor='#1E5A99';"
             onmouseout="this.style.borderColor='transparent';">
            <div style='display:flex;justify-content:space-around;align-items:center;'>
                <div><img src='{m["away_logo"]}' width='45'><div style='color:#D6D6D6;font-size:13px;'>{m["away"]}</div></div>
                <div style='color:#D6D6D6;'>@</div>
                <div><img src='{m["home_logo"]}' width='45'><div style='color:#D6D6D6;font-size:13px;'>{m["home"]}</div></div>
            </div>
        </div>
        """
        if st.button(f"{m['away']}@{m['home']}", key=f"matchup_{i}"):
            st.session_state.selected_matchup = (m["away"], m["home"])
        col.markdown(html_block, unsafe_allow_html=True)

# ---------------------------------------------------------------
# Line Input + Run
# ---------------------------------------------------------------
col_run, col_line = st.columns([3,1])
with col_run:
    run_model = st.button("🚀 Run Model for All Games")
with col_line:
    line_test = st.number_input("Line to Test", min_value=0.0, max_value=10.0, value=3.5, step=0.5)

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results = []
    skaters = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
    roster = skaters[[player_col, team_col]].rename(columns={player_col:"player", team_col:"team"}).drop_duplicates("player")
    grouped = {n.lower():g.sort_values(game_col) for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    line_adj = {}
    if not lines_df.empty and "line pairings" in lines_df.columns:
        l = lines_df.copy()
        l["games"] = pd.to_numeric(l["games"], errors="coerce").fillna(0)
        l["sog against"] = pd.to_numeric(l["sog against"], errors="coerce").fillna(0)
        l = l.groupby(["line pairings","team"], as_index=False).agg({"games":"sum","sog against":"sum"})
        l["sog_against_per_game"] = np.where(l["games"]>0, l["sog against"]/l["games"], np.nan)
        team_avg = l.groupby("team")["sog_against_per_game"].mean()
        league_avg = team_avg.mean()
        l["line_factor"] = (league_avg / l["sog_against_per_game"]).clip(0.7,1.3)
        line_adj = l.copy()

    goalie_adj = {}
    if not goalies_df.empty and {"team","shots against","games"}.issubset(goalies_df.columns):
        g = goalies_df.copy()
        g["shots against"] = pd.to_numeric(g["shots against"], errors="coerce").fillna(0)
        g["games"] = pd.to_numeric(g["games"], errors="coerce").fillna(1)
        g["shots_per_game"] = g["shots against"] / g["games"]
        league_avg_sa = g["shots_per_game"].mean()
        g["goalie_factor"] = (g["shots_per_game"] / league_avg_sa).clip(0.7,1.3)
        goalie_adj = g.groupby("team")["goalie_factor"].mean().to_dict()

    for row in roster.itertuples(index=False):
        player, team = row.player, row.team
        df_p = grouped.get(player.lower(), pd.DataFrame())
        if df_p.empty: continue
        sog_values = df_p.groupby(game_col)["sog"].sum().tolist()
        if not sog_values: continue
        last3 = sog_values[-3:] if len(sog_values)>=3 else sog_values
        last5 = sog_values[-5:] if len(sog_values)>=5 else sog_values
        last10 = sog_values[-10:] if len(sog_values)>=10 else sog_values
        l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)
        baseline = (0.55*l10 + 0.30*l5 + 0.15*l3)
        line_factor_internal = 1.0
        if isinstance(line_adj, pd.DataFrame) and not line_adj.empty:
            last_name = str(player).split()[-1].lower()
            m = line_adj[line_adj["line pairings"].str.contains(last_name, case=False, na=False)]
            if not m.empty: line_factor_internal = np.average(m["line_factor"], weights=m["games"])
        opp_team = team_b if team == team_a else team_a
        goalie_factor = goalie_adj.get(opp_team, 1.0)
        goalie_term = (goalie_factor - 1.0) * 0.2
        lam_base = baseline * (1 + goalie_term)
        if line_factor_internal >= 1: scale = 1 + 7.0 * (line_factor_internal - 1.0)**1.5
        else: scale = max(0.05, line_factor_internal**3.5)
        lam = lam_base * scale
        trend = (l5 - l10)/l10 if l10>0 else 0
        poisson_prob = 1 - poisson.cdf(np.floor(line_test)-1, mu=max(lam,0.01))
        odds = -100*(poisson_prob/(1-poisson_prob)) if poisson_prob>=0.5 else 100*((1-poisson_prob)/poisson_prob)
        injury_html = ""
        if not injuries_df.empty and {"player","team"}.issubset(injuries_df.columns):
            player_lower=player.lower().strip(); last_name=player_lower.split()[-1]; team_lower=team.lower().strip()
            match=injuries_df[injuries_df["team"].str.lower().str.strip().eq(team_lower)&injuries_df["player"].str.lower().str.endswith(last_name)]
            if not match.empty:
                note=str(match.iloc[0].get("injury note","")).strip()
                injury_html=f"<span style='cursor:pointer;' title='{note}'>🚑</span>"
        form_flag="⚪ Neutral Form"
        results.append({
            "Player":player,"Team":team,"Injury":injury_html,"Trend Score":round(trend,3),
            "Final Projection":round(lam,2),"Prob ≥ Line (%)":round(poisson_prob*100,1),
            "Playable Odds":f"{'+' if odds>0 else ''}{int(odds)}","Season Avg":round(np.mean(sog_values),2),
            "Line Adj":round(line_factor_internal,2),"Form Indicator":form_flag,
            "L3 Shots":", ".join(map(str,last3)),"L5 Shots":", ".join(map(str,last5)),"L10 Shots":", ".join(map(str,last10))
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model:
    combined = []
    for m in games:
        df = build_model(m["away"], m["home"], skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df)
        combined.append(df)
    combined = pd.concat(combined, ignore_index=True)
    combined = combined.sort_values(["team","final projection","line adj"], ascending=[True,False,False])
    st.session_state.results_base = combined
    st.success("✅ Model built successfully!")

# ---------------------------------------------------------------
# Display Table
# ---------------------------------------------------------------
if "results_base" in st.session_state:
    df = st.session_state.results_base.copy()
    def trend_color(v):
        if pd.isna(v): return "–"
        if v>0.05: color,txt,sym="#00B140","#fff","▲"
        elif v<-0.05: color,txt,sym="#E63946","#fff","▼"
        else: color,txt,sym="#6C7A89","#fff","–"
        return f"<div style='background:{color};color:{txt};font-weight:600;border-radius:6px;padding:4px 8px;text-align:center;'>{sym}</div>"
    df["Trend"] = df["Trend Score"].apply(trend_color)
    cols = ["Player","Team","Injury","Trend","Final Projection","Prob ≥ Line (%)","Playable Odds",
            "Season Avg","Line Adj","Form Indicator","L3 Shots","L5 Shots","L10 Shots"]
    vis = df[[c for c in cols if c in df.columns]]
    html_table = vis.to_html(index=False, escape=False)
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
    """, height=670, scrolling=True)
