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
    "NJ": "NJD", "LA": "LAK", "SJ": "SJS", "TB": "TBL", "ARI": "ARI",
    "ANA": "ANA", "BOS": "BOS", "BUF": "BUF", "CAR": "CAR", "CBJ": "CBJ",
    "CGY": "CGY", "CHI": "CHI", "COL": "COL", "DAL": "DAL", "DET": "DET",
    "EDM": "EDM", "FLA": "FLA", "MIN": "MIN", "MTL": "MTL", "NSH": "NSH",
    "NYI": "NYI", "NYR": "NYR", "OTT": "OTT", "PHI": "PHI", "PIT": "PIT",
    "SEA": "SEA", "STL": "STL", "TOR": "TOR", "VAN": "VAN", "VGK": "VGK",
    "WSH": "WSH", "WPG": "WPG"
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
<p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups with injuries, expected goals, and consistency ratings.</p>
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
    base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
    def find_file(name):
        for p in base_paths:
            fp = os.path.join(p, name)
            if os.path.exists(fp): return fp
        return None
        
    skaters = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
    shots   = load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
    goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
    lines   = load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
    teams   = load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")

    injuries = pd.DataFrame()
    for p in ["injuries.xlsx", "Injuries.xlsx", "data/injuries.xlsx"]:
        if os.path.exists(p): injuries = load_file(open(p, "rb")); break
    if injuries.empty: injuries = load_file(injuries_file)
    if not injuries.empty:
        injuries.columns = injuries.columns.str.lower().str.strip()
        if "player" in injuries.columns:
            injuries["player"] = injuries["player"].astype(str).str.strip().str.lower()
    return skaters, shots, goalies, lines, teams, injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file)

if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing data. Upload required files.")
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns = df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), "team")
player_col = "name" if "name" in skaters_df.columns else "player"
shots_df = shots_df.rename(columns={next((c for c in shots_df.columns if "player" in c or "name" in c), "player"): "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), "game_id")

# ---------------------------------------------------------------
# Matchup Pull (ESPN)
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_todays_games():
    url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        games = []
        for e in data.get("events", []):
            comps = e.get("competitions", [{}])[0].get("competitors", [])
            if len(comps) == 2:
                away, home = comps[0], comps[1]
                games.append({
                    "away": away["team"]["abbreviation"],
                    "home": home["team"]["abbreviation"],
                    "away_logo": away["team"]["logo"],
                    "home_logo": home["team"]["logo"]
                })
        return games
    except: return []

games = get_todays_games()
for g in games:
    g["away"] = TEAM_ABBREV_MAP.get(g["away"], g["away"])
    g["home"] = TEAM_ABBREV_MAP.get(g["home"], g["home"])

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results = []
    skaters = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
    roster = skaters[[player_col, team_col]].rename(columns={player_col: "player", team_col: "team"}).drop_duplicates("player")
    grouped = {n.lower(): g for n, g in shots_df.groupby(shots_df["player"].str.lower())}

    line_adj = {}
    if not lines_df.empty and "line pairings" in lines_df.columns:
        l = lines_df.copy()
        l["games"] = pd.to_numeric(l["games"], errors="coerce").fillna(0)
        l["sog against"] = pd.to_numeric(l["sog against"], errors="coerce").fillna(0)
        l = l.groupby(["line pairings", "team"], as_index=False).agg({"games": "sum", "sog against": "sum"})
        l["sog_against_per_game"] = np.where(l["games"] > 0, l["sog against"] / l["games"], np.nan)
        team_avg = l.groupby("team")["sog_against_per_game"].mean()
        league_avg = team_avg.mean()
        l["line_factor"] = (league_avg / l["sog_against_per_game"]).clip(0.7, 1.3)
        line_adj = l.copy()

    goalie_adj = {}
    if not goalies_df.empty and {"team", "shots against", "games"}.issubset(goalies_df.columns):
        g = goalies_df.copy()
        g["shots against"] = pd.to_numeric(g["shots against"], errors="coerce").fillna(0)
        g["games"] = pd.to_numeric(g["games"], errors="coerce").fillna(1)
        g["shots_per_game"] = g["shots against"] / g["games"]
        league_avg_sa = g["shots_per_game"].mean()
        g["goalie_factor"] = (g["shots_per_game"] / league_avg_sa).clip(0.7, 1.3)
        goalie_adj = g.groupby("team")["goalie_factor"].mean().to_dict()

    for row in roster.itertuples(index=False):
        player, team = row.player, row.team
        df_p = grouped.get(player.lower(), pd.DataFrame())
        if df_p.empty or "sog" not in df_p.columns: continue

        sog_vals = df_p.groupby(game_col)["sog"].sum().tolist()
        if not sog_vals: continue
        last3, last5, last10 = sog_vals[-3:], sog_vals[-5:], sog_vals[-10:]
        l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)
        
        # Consistency L10
        l10_avg = np.mean(last10)
        l10_std = np.std(last10)
        cv = l10_std / l10_avg if l10_avg > 0 else 1.0
        consistency_score = round(np.clip(10 - (cv * 10), 1, 10), 1)
        stars = "⭐" * int(consistency_score // 2) if consistency_score > 2 else "🌑"

        baseline = (0.55 * l10) + (0.3 * l5) + (0.15 * l3)
        trend = (l5 - l10) / l10 if l10 > 0 else 0
        form_flag = "🟢 Above Baseline" if trend > 0.05 else "🔴 Below Baseline" if trend < -0.05 else "⚪ Neutral"

        line_factor_internal = 1.0
        if isinstance(line_adj, pd.DataFrame) and not line_adj.empty:
            last_name = str(player).split()[-1].lower()
            m = line_adj[line_adj["line pairings"].str.contains(last_name, case=False, na=False)]
            if not m.empty: line_factor_internal = np.average(m["line_factor"], weights=m["games"])

        opp_team = team_b if team == team_a else team_a
        goalie_factor = goalie_adj.get(opp_team, 1.0)
        heat_multiplier = (1 + (goalie_factor - 1.0) * 0.2) * line_factor_internal
        heat_score = round(np.clip(heat_multiplier * 5, 1, 10), 1)
        heat_flame = "🔥" * int(heat_score // 2) if heat_score > 4 else "🧊"

        lam = baseline * heat_multiplier
        
        injury_html = ""
        if not injuries_df.empty and {"player", "team"}.issubset(injuries_df.columns):
            match = injuries_df[injuries_df["team"].str.lower().str.strip().eq(team.lower()) & 
                              injuries_df["player"].str.lower().str.endswith(player.lower().split()[-1])]
            if not match.empty:
                tooltip = str(match.iloc[0].get("injury note", "")).strip()
                injury_html = f"<span style='cursor:pointer;' onclick='alert({json.dumps(html.escape(tooltip))})'>🚑</span>"

        if "goal" in df_p.columns:
            agg = df_p.groupby(game_col).agg({"sog": "sum", "goal": "sum"}).reset_index()
            shooting_pct = agg["goal"].mean() / agg["sog"].mean() if agg["sog"].mean() > 0 else 0
            exp_goals = shooting_pct * lam
        else: exp_goals, shooting_pct = np.nan, np.nan

        results.append({
            "Player": player, "Team": team, "Injury": injury_html, 
            "Consistency (L10)": f"{stars} ({consistency_score})",
            "Matchup Heat": f"{heat_flame} ({heat_score})", "Final Projection": round(lam, 2),
            "Season Avg": round(np.mean(sog_vals), 2), "Exp Goals (xG)": round(exp_goals, 3) if not np.isnan(exp_goals) else "",
            "Form Indicator": form_flag, "L10 Shots": ", ".join(map(str, last10)), "Trend Score": trend
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Logic & Display
# ---------------------------------------------------------------
col_run, col_line = st.columns([3, 1])
with col_run: run_model = st.button("🚀 Run Model (All Games)")
with col_line: line_test = st.number_input("Line to Test", 0.0, 10.0, 3.5, 0.5)

if run_model:
    all_tables = []
    for m in games:
        df = build_model(m["away"], m["home"], skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df)
        if not df.empty:
            df["Matchup"] = f"{m['away']}@{m['home']}"
            all_tables.append(df)
    if all_tables:
        st.session_state.results = pd.concat(all_tables, ignore_index=True)
        st.session_state.matchups = games

if "results" in st.session_state:
    df = st.session_state.results.copy()
    
    # Matching Buttons
    cols = st.columns(3)
    for i, m in enumerate(st.session_state.matchups):
        match_id = f"{m['away']}@{m['home']}"
        with cols[i % 3]:
            if st.button(f"{m['away']} @ {m['home']}", key=f"btn_{i}"):
                st.session_state.selected_teams = {m['away'], m['home']}

    if st.session_state.get("selected_teams"):
        df = df[df["Team"].isin(st.session_state.selected_teams)]

    df["Trend"] = df["Trend Score"].apply(lambda v: "▲" if v > 0.05 else ("▼" if v < -0.05 else "–"))
    df["Prob ≥ Line (%)"] = df["Final Projection"].apply(lambda lam: round((1-poisson.cdf(line_test-1, mu=max(lam, 0.01)))*100, 1))

    cols_to_show = ["Player", "Team", "Injury", "Consistency (L10)", "Matchup Heat", "Trend", "Final Projection", "Prob ≥ Line (%)", "Season Avg", "Exp Goals (xG)", "Form Indicator", "L10 Shots"]
    html_table = df[cols_to_show].to_html(index=False, escape=False)
    
    components.html(f"""
    <style>
    table {{ width:100%; border-collapse:collapse; color:#D6D6D6; font-family:sans-serif; }}
    th {{ background-color:#0A3A67; color:white; padding:8px; }}
    td {{ background-color:#0F2743; padding:6px; text-align:center; border-bottom:1px solid #142F52; }}
    td:first-child {{ position:sticky; left:0; background:#1E5A99; font-weight:bold; }}
    </style>
    <div style='overflow-x:auto;'>{html_table}</div>
    """, height=600, scrolling=True)

    st.download_button("💾 Download CSV", df.to_csv(index=False), "results.csv")
