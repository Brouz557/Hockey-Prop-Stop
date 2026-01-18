# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Defensive Rating Integration (Opponent fix)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, datetime, pytz, subprocess
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with blended regression, L5-based probabilities, and defensive ratings
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def load_file(file):
    if not file: return pd.DataFrame()
    try:
        return pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
    except Exception: return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception: return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None: return load_file(file_uploader)
    return safe_read(default_path)

# ---------------------------------------------------------------
# Cached Data Load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    base_paths = [".","data","/mount/src/hockey-prop-stop/data"]
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
    return skaters, shots, goalies, lines, teams

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()
st.success("✅ Data loaded successfully.")

# ---------------------------------------------------------------
# 🕒 Data Last Updated — True Git Commit Timestamp for SHOT DATA
# ---------------------------------------------------------------
def get_shots_file_git_time():
    tz_cst = pytz.timezone("America/Chicago")
    file_candidates = [
        "data/SHOT DATA.xlsx",
        "/mount/src/hockey-prop-stop/data/SHOT DATA.xlsx",
        "SHOT DATA.xlsx"
    ]
    for f in file_candidates:
        if os.path.exists(f):
            try:
                git_time_str = subprocess.check_output(
                    ["git", "log", "-1", "--format=%cd", "--date=iso", f],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                if git_time_str:
                    git_time = datetime.datetime.fromisoformat(git_time_str.replace("Z","+00:00"))
                    return git_time.astimezone(tz_cst).strftime("%Y-%m-%d %I:%M %p CST")
            except Exception:
                continue
    return None

last_update = get_shots_file_git_time()
if last_update:
    st.markdown(f"🕒 **Data last updated:** {last_update}")
else:
    st.markdown("🕒 **Data last updated:** Unknown")

# ---------------------------------------------------------------
# Normalize Columns and Detect Key Fields
# ---------------------------------------------------------------
for df in [skaters_df, shots_df, goalies_df, lines_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None

# Detect key shot data columns dynamically
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
if player_col_shots:
    shots_df = shots_df.rename(columns={player_col_shots: "player"})
else:
    st.error("❌ Could not find a player column in SHOT DATA file.")
    st.stop()

# --- Normalize and detect Opponent column ---
opp_col = next((c for c in shots_df.columns if "opp" in c.lower()), None)
if not opp_col:
    st.error("❌ Could not find an Opponent column in SHOT DATA file.")
    st.stop()
else:
    shots_df.rename(columns={opp_col: "opponent"}, inplace=True)

# --- Detect and standardize SOG column ---
sog_col = next((c for c in shots_df.columns if "sog" in c), None)
if not sog_col:
    st.error("❌ Could not find a shots-on-goal (SOG) column in SHOT DATA file.")
    st.stop()
shots_df.rename(columns={sog_col: "sog"}, inplace=True)

shots_df["player"] = shots_df["player"].astype(str).str.strip()
shots_df["opponent"] = shots_df["opponent"].astype(str).str.strip()

# ---------------------------------------------------------------
# Team Selection
# ---------------------------------------------------------------
teams = sorted(skaters_df[team_col].dropna().unique().tolist())
col1, col2 = st.columns(2)
with col1: team_a = st.selectbox("Select Team A", teams)
with col2: team_b = st.selectbox("Select Team B", [t for t in teams if t != team_a])
st.markdown("---")

# ---------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df):
    results = []
    roster = skaters_df[skaters_df[team_col].isin([team_a, team_b])][[player_col, team_col, "position"]]
    roster = roster.rename(columns={player_col:"player", team_col:"team"}).drop_duplicates("player")
    grouped = {n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    for row in roster.itertuples(index=False):
        player, team = row.player, row.team
        df_p = grouped.get(player.lower(), pd.DataFrame())
        if df_p.empty: continue
        sog_values = df_p["sog"].tolist()
        if not sog_values: continue

        last5 = sog_values[-5:] if len(sog_values)>=5 else sog_values
        l5 = np.mean(last5)
        season_avg = np.mean(sog_values)
        trend = (l5 - season_avg)/season_avg if season_avg>0 else 0

        lam = l5  # λ = L5 average
        line = round(lam, 2)
        prob = 1 - poisson.cdf(np.floor(line) - 1, mu=lam)
        p = min(max(prob, 0.001), 0.999)
        odds = -100*(p/(1-p)) if p>=0.5 else 100*((1-p)/p)
        implied_odds = f"{'+' if odds>0 else ''}{int(odds)}"

        results.append({
            "Player":player,
            "Team":team,
            "Position":row.position,
            "Season Avg":round(season_avg,2),
            "L5 Avg":round(l5,2),
            "Final Projection":round(line,2),
            "Prob ≥ Projection (%) L5":round(p*100,1),
            "Playable Odds":implied_odds,
            "Trend Score":round(trend,3)
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if st.button("🚀 Run Model"):
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")
    df = build_model(team_a, team_b, skaters_df, shots_df)
    df = df.sort_values("Final Projection", ascending=False).reset_index(drop=True)
    st.session_state.results_raw = df.copy()
    st.success("✅ Model built successfully!")

# ---------------------------------------------------------------
# Display Main Table
# ---------------------------------------------------------------
if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
    df = st.session_state.results_raw.copy()
    def trend_color(v):
        if pd.isna(v): return "–"
        v = max(min(v,0.5),-0.5)
        n = v+0.5
        if n<0.5: r,g,b=255,int(255*(n*2)),0
        else: r,g,b=int(255*(1-(n-0.5)*2)),255,0
        color=f"rgb({r},{g},{b})"
        t="▲" if v>0.05 else ("▼" if v<-0.05 else "–")
        txt="#000" if abs(v)<0.2 else "#fff"
        return f"<div style='background:{color};color:{txt};font-weight:600;border-radius:6px;padding:4px 8px;text-align:center;'>{t}</div>"
    df["Trend"] = df["Trend Score"].apply(trend_color)

    cols = ["Player","Team","Position","Trend","Final Projection","Prob ≥ Projection (%) L5",
            "Playable Odds","Season Avg","L5 Avg"]
    vis = df[[c for c in cols if c in df.columns]]

    html_table = vis.to_html(index=False, escape=False)
    components.html(
        f"""
        <style>
        div.scrollable-table {{
            overflow-x: auto;
            overflow-y: auto;
            height: 600px;
            position: relative;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Source Sans Pro', sans-serif;
            color: #f0f0f0;
        }}
        th {{
            background-color: #00B140;
            color: white;
            padding: 6px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 3;
        }}
        td:first-child, th:first-child {{
            position: sticky;
            left: 0;
            z-index: 4;
            background-color: #00B140;
            color: white;
            font-weight: bold;
        }}
        td {{
            background-color: #1e1e1e;
            color: #f0f0f0;
            padding: 4px;
            text-align: center;
        }}
        tr:nth-child(even) td {{ background-color: #2a2a2a; }}
        </style>
        <div class='scrollable-table'>{html_table}</div>
        """,
        height=620,
        scrolling=True,
    )

# ---------------------------------------------------------------
# Defensive Rating Table (New Section)
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("### 🧱 Team Defensive Ratings (by Position)")

try:
    merged = shots_df.merge(
        skaters_df[[player_col, "position"]],
        left_on="player", right_on=player_col, how="left"
    )
    merged = merged.dropna(subset=["opponent","position","sog"])
    merged["sog"] = pd.to_numeric(merged["sog"], errors="coerce").fillna(0)

    team_pos = merged.groupby(["opponent","position"])["sog"].mean().reset_index()
    team_pos.rename(columns={"opponent":"Team","sog":"Avg SOG Allowed"}, inplace=True)

    league_avg = team_pos.groupby("position")["Avg SOG Allowed"].transform("mean")
    team_pos["Rel to Avg"] = team_pos["Avg SOG Allowed"]/league_avg

    team_pos["Percentile"] = team_pos.groupby("position")["Avg SOG Allowed"].rank(pct=True)
    team_pos["Def Rating (1=Best,5=Worst)"] = pd.cut(
        team_pos["Percentile"],
        bins=[0,0.2,0.4,0.6,0.8,1.0],
        labels=[1,2,3,4,5]
    ).astype(int)

    team_pos = team_pos.sort_values(["position","Def Rating (1=Best,5=Worst)"])
    st.dataframe(team_pos, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error computing defensive ratings: {e}")
