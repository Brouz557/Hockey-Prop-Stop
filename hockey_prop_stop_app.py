# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Full App with Timestamp + Everything
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import contextlib
import datetime
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
shots_file = st.sidebar.file_uploader("SHOT DATA", type=["xlsx", "csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx", "csv"])
lines_file = st.sidebar.file_uploader("LINE DATA", type=["xlsx", "csv"])
teams_file = st.sidebar.file_uploader("TEAMS", type=["xlsx", "csv"])

# ---------------------------------------------------------------
# File Helpers
# ---------------------------------------------------------------
def load_file(file):
    if not file:
        return pd.DataFrame()
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception:
        return pd.DataFrame()

def safe_read(path):
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception:
        return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None:
        return load_file(file_uploader)
    return safe_read(default_path)

# ---------------------------------------------------------------
# Cache + Load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
    def find_file(filename):
        for p in base_paths:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return full
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        skaters = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
        shots = load_data(shots_file, find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
        goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
        lines = load_data(lines_file, find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
        teams = load_data(teams_file, find_file("TEAMS.xlsx") or "TEAMS.xlsx")

    return skaters, shots, goalies, lines, teams

# ---------------------------------------------------------------
# Timestamp Helpers
# ---------------------------------------------------------------
def get_excel_lastupdated_tag(file_path):
    try:
        df = pd.read_excel(file_path, header=None, nrows=10)
        for _, row in df.iterrows():
            for cell in row:
                if isinstance(cell, str) and cell.strip().lower().startswith("lastupdated:"):
                    return cell.split(":", 1)[1].strip()
    except Exception:
        return None
    return None

def get_file_last_modified(path):
    try:
        ts = os.path.getmtime(path)
        dt = datetime.datetime.fromtimestamp(ts, ZoneInfo("America/Chicago"))
        return dt.strftime("%B %d, %Y — %I:%M %p CT")
    except Exception:
        return None

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file
)

# Determine update timestamp
possible_paths = [
    "Skaters.xlsx",
    "./data/Skaters.xlsx",
    "/mount/src/hockey-prop-stop/data/Skaters.xlsx",
]
skaters_path = next((p for p in possible_paths if os.path.exists(p)), None)

update_info = None
if skaters_path:
    excel_tag = get_excel_lastupdated_tag(skaters_path)
    if excel_tag:
        update_info = f"Last updated: {excel_tag}"
    else:
        file_time = get_file_last_modified(skaters_path)
        if file_time:
            update_info = f"Last updated: {file_time}"
if not update_info:
    update_info = "Last updated: (timestamp unavailable)"

st.success(f"✅ Data loaded successfully ({update_info}).")

# ---------------------------------------------------------------
# Stop if missing key data
# ---------------------------------------------------------------
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()

# ---------------------------------------------------------------
# Data Prep
# ---------------------------------------------------------------
skaters_df.columns = skaters_df.columns.str.lower().str.strip()
shots_df.columns = shots_df.columns.str.lower().str.strip()
if not goalies_df.empty:
    goalies_df.columns = goalies_df.columns.str.lower().str.strip()
if not lines_df.empty:
    lines_df.columns = lines_df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
sog_col = next((c for c in shots_df.columns if "sog" in c), None)
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)

# ---------------------------------------------------------------
# Team Selectors
# ---------------------------------------------------------------
all_teams = sorted(skaters_df[team_col].dropna().unique().tolist())
col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A", all_teams)
with col2:
    team_b = st.selectbox("Select Team B", [t for t in all_teams if t != team_a])

st.markdown("---")
run_model = st.button("🚀 Run Model")

# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
if run_model:
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

    # 🥅 Goalie Adjustments
    goalie_adj, rebound_rate = {}, {}
    if not goalies_df.empty:
        df_g = goalies_df.copy()
        df_g = df_g[df_g["situation"].str.lower() == "all"]
        df_g["games"] = pd.to_numeric(df_g["games"], errors="coerce").fillna(0)
        df_g["unblocked attempts"] = pd.to_numeric(df_g["unblocked attempts"], errors="coerce").fillna(0)
        df_g["rebounds"] = pd.to_numeric(df_g["rebounds"], errors="coerce").fillna(0)
        df_g["shots_allowed_per_game"] = np.where(df_g["games"] > 0, df_g["unblocked attempts"] / df_g["games"], np.nan)
        df_g["rebound_rate"] = np.where(df_g["unblocked attempts"] > 0, df_g["rebounds"] / df_g["unblocked attempts"], 0)
        team_avg = df_g.groupby("team")["shots_allowed_per_game"].mean()
        league_avg = team_avg.mean()
        goalie_adj = (league_avg / team_avg).to_dict()
        rebound_rate = df_g.groupby("team")["rebound_rate"].mean().to_dict()

    # 🧱 Line Adjustments
    line_adj = {}
    if not lines_df.empty:
        df_l = lines_df.copy()
        df_l["games"] = pd.to_numeric(df_l["games"], errors="coerce").fillna(0)
        df_l["sog against"] = pd.to_numeric(df_l["sog against"], errors="coerce").fillna(0)
        df_l = (
            df_l.groupby(["line pairings", "team"], as_index=False)
            .agg({"games": "sum", "sog against": "sum"})
        )
        df_l["sog_against_per_game"] = np.where(df_l["games"] > 0, df_l["sog against"] / df_l["games"], np.nan)
        team_avg = df_l.groupby("team")["sog_against_per_game"].mean()
        league_avg = team_avg.mean()
        df_l["line_factor"] = (league_avg / df_l["sog_against_per_game"]).clip(0.7, 1.3)
        line_adj = df_l.copy()

    # ---------------------------------------------------------------
    # Build roster and projections
    # ---------------------------------------------------------------
    roster = (
        skaters_df[skaters_df[team_col].isin([team_a, team_b])][[player_col, team_col]]
        .rename(columns={player_col: "player", team_col: "team"})
        .drop_duplicates(subset=["player"])
        .reset_index(drop=True)
    )

    shots_df = shots_df.rename(columns={player_col_shots: "player", game_col: "gameid", sog_col: "sog"})
    shots_df["player"] = shots_df["player"].astype(str).str.strip()
    roster["player"] = roster["player"].astype(str).str.strip()

    grouped_shots = {
        name.lower(): g.sort_values("gameid")
        for name, g in shots_df.groupby(shots_df["player"].str.lower())
    }

    results = []
    progress = st.progress(0)
    total = len(roster)

    for i, row in enumerate(roster.itertuples(index=False), start=1):
        player, team = row.player, row.team
        df_p = grouped_shots.get(str(player).lower(), pd.DataFrame())
        if df_p.empty:
            progress.progress(i / total)
            continue

        game_sogs = df_p.groupby("gameid")["sog"].sum().reset_index().sort_values("gameid")
        sog_values = game_sogs["sog"].tolist()

        last3, last5, last10 = list(reversed(sog_values[-3:])), list(reversed(sog_values[-5:])), list(reversed(sog_values[-10:]))
        l3, l5, l10 = np.mean(last3) if last3 else np.nan, np.mean(last5) if last5 else np.nan, np.mean(last10) if last10 else np.nan
        season_avg = np.mean(sog_values)
        trend = 0 if pd.isna(l10) or l10 == 0 else (l3 - l10) / l10
        base_proj = np.nansum([0.5 * l3, 0.3 * l5, 0.2 * l10])

        opp_team = team_b if team == team_a else team_a
        goalie_factor = np.clip(goalie_adj.get(opp_team, 1.0), 0.7, 1.3)
        rebound_factor = rebound_rate.get(opp_team, 0.0)
        line_factor = 1.0

        if not line_adj.empty:
            last_name = str(player).split()[-1].lower()
            matching = line_adj[line_adj["line pairings"].str.contains(last_name, case=False, na=False)]
            if not matching.empty:
                line_factor = np.average(matching["line_factor"], weights=matching["games"])
            line_factor = np.clip(line_factor, 0.7, 1.3)

        adj_proj = base_proj * (0.7 + 0.3 * goalie_factor) * (0.7 + 0.3 * line_factor)
        adj_proj *= (1 + rebound_factor * 0.1)
        adj_proj = max(0, round(adj_proj, 2))

        l10_shots_formatted = (
            "<br>".join([", ".join(map(str, last10[:5])), ", ".join(map(str, last10[5:]))])
            if len(last10) > 5
            else ", ".join(map(str, last10))
        )

        results.append({
            "Player": player,
            "Team": team,
            "Season Avg": round(season_avg, 2),
            "L3 Shots": ", ".join(map(str, last3)),
            "L5 Shots": ", ".join(map(str, last5)),
            "L10 Shots": l10_shots_formatted,
            "Trend Score": round(trend, 3),
            "Base Projection": round(base_proj, 2),
            "Goalie Adj": round(goalie_factor, 2),
            "Line Adj": round(line_factor, 2),
            "Final Projection": adj_proj,
        })
        progress.progress(i / total)
    progress.empty()

    # ---------------------------------------------------------------
    # Ratings + Trend color
    # ---------------------------------------------------------------
    result_df = pd.DataFrame(results)
    avg_proj, std_proj = result_df["Final Projection"].mean(), result_df["Final Projection"].std()

    def rate(val):
        if val >= avg_proj + std_proj:
            return "Strong"
        elif val >= avg_proj:
            return "Moderate"
        else:
            return "Weak"

    result_df["Matchup Rating"] = result_df["Final Projection"].apply(rate)

    def trend_color(val):
        if pd.isna(val):
            return "<div style='background:#E0E0E0;color:#000;border-radius:6px;'>–</div>"
        val = max(min(val, 0.5), -0.5)
        norm = (val + 0.5)
        if norm < 0.5:
            r, g, b = 255, int(255 * (norm * 2)), 0
        else:
            r, g, b = int(255 * (1 - (norm - 0.5) * 2)), 255, 0
        color = f"rgb({r},{g},{b})"
        text = "▲" if val > 0.05 else ("▼" if val < -0.05 else "–")
        text_color = "#000" if abs(val) < 0.2 else "#fff"
        return f"<div style='background:{color};color:{text_color};font-weight:600;border-radius:6px;padding:4px 8px;text-align:center;' title='Trend: {val:+.2f}'>{text}</div>"

    result_df["Trend"] = result_df["Trend Score"].apply(trend_color)

    # ---------------------------------------------------------------
    # TEAM LOGOS
    # ---------------------------------------------------------------
    team_logos = {
        "Toronto Maple Leafs": "TOR", "Vancouver Canucks": "VAN", "Edmonton Oilers": "EDM",
        "Calgary Flames": "CGY", "Montreal Canadiens": "MTL", "Ottawa Senators": "OTT",
        "Boston Bruins": "BOS", "New York Rangers": "NYR", "New York Islanders": "NYI",
        "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "Chicago Blackhawks": "CHI",
        "Colorado Avalanche": "COL", "Dallas Stars": "DAL", "Vegas Golden Knights": "VGK",
        "Los Angeles Kings": "LAK", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
        "Detroit Red Wings": "DET", "Tampa Bay Lightning": "TBL", "Florida Panthers": "FLA",
        "Nashville Predators": "NSH", "Washington Capitals": "WSH", "Buffalo Sabres": "BUF",
        "St. Louis Blues": "STL", "Winnipeg Jets": "WPG", "Minnesota Wild": "MIN",
        "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Columbus Blue Jackets": "CBJ",
        "New Jersey Devils": "NJD"
    }

    def get_logo_html(team_name):
        abbr = team_logos.get(team_name, None)
        if not abbr:
            return team_name
        url = f"https://assets.nhle.com/logos/nhl/svg/{abbr}_light.svg"
        return f"<img src='{url}' width='26' style='vertical-align:middle;margin-right:6px;'> {team_name}"

    result_df["Team"] = result_df["Team"].apply(get_logo_html)

    # ---------------------------------------------------------------
    # Display Table
    # ---------------------------------------------------------------
    display_cols = [
        "Player", "Team", "Trend", "Final Projection", "Season Avg", "Matchup Rating",
        "L3 Shots", "L5 Shots", "L10 Shots", "Base Projection", "Goalie Adj", "Line Adj"
    ]

    visible_df = result_df[[c for c in display_cols if c in result_df.columns]]
    visible_df = visible_df.sort_values("Final Projection", ascending=False)

    st.success(f"✅ Model built successfully for {team_a} vs {team_b}!")
    st.markdown(f"### 📊 {team_a} vs {team_b} — Player Projections (Adjusted)")
    html_table = visible_df.to_html(index=False, escape=False)
    st.markdown(f"<div style='overflow-x:auto'>{html_table}</div>", unsafe_allow_html=True)
