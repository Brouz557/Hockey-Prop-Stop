# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Usage-Adjusted Regression Detection
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io
from scipy.stats import poisson

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with regression-to-mean and usage insights
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .dataframe td {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: center !important;
        line-height: 1.3em !important;
    }
    .stDataFrame { overflow-x: auto; }
    </style>
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
# Helper functions
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
    if file_uploader is not None:
        return load_file(file_uploader)
    return safe_read(default_path)

# ---------------------------------------------------------------
# Cached load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    base_paths = [".","data","/mount/src/hockey-prop-stop/data"]
    def find_file(filename):
        for p in base_paths:
            full = os.path.join(p, filename)
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
# Normalize columns
# ---------------------------------------------------------------
skaters_df.columns = skaters_df.columns.str.lower().str.strip()
shots_df.columns   = shots_df.columns.str.lower().str.strip()
if not goalies_df.empty: goalies_df.columns = goalies_df.columns.str.lower().str.strip()
if not lines_df.empty:   lines_df.columns   = lines_df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
toi_col = next((c for c in skaters_df.columns if "toi" in c), None)
gp_col = next((c for c in skaters_df.columns if "gp" in c or "games" in c), None)

sog_col = next((c for c in shots_df.columns if "sog" in c), None)
goal_col = next((c for c in shots_df.columns if "goal" in c), None)
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)

# ---------------------------------------------------------------
# Team selection
# ---------------------------------------------------------------
all_teams = sorted(skaters_df[team_col].dropna().unique().tolist())
col1, col2 = st.columns(2)
with col1: team_a = st.selectbox("Select Team A", all_teams)
with col2: team_b = st.selectbox("Select Team B", [t for t in all_teams if t != team_a])
st.markdown("---")

if "results" not in st.session_state: st.session_state.results = None
run_model = st.button("🚀 Run Model")

# ---------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------
if run_model:
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

    goalie_adj, rebound_rate = {}, {}
    if not goalies_df.empty:
        g = goalies_df.copy()
        g = g[g["situation"].str.lower() == "all"]
        g["games"] = pd.to_numeric(g["games"], errors="coerce").fillna(0)
        g["unblocked attempts"] = pd.to_numeric(g["unblocked attempts"], errors="coerce").fillna(0)
        g["rebounds"] = pd.to_numeric(g["rebounds"], errors="coerce").fillna(0)
        g["shots_allowed_per_game"] = np.where(g["games"]>0, g["unblocked attempts"]/g["games"], np.nan)
        g["rebound_rate"] = np.where(g["unblocked attempts"]>0, g["rebounds"]/g["unblocked attempts"], 0)
        team_avg = g.groupby("team")["shots_allowed_per_game"].mean()
        league_avg = team_avg.mean()
        goalie_adj = (league_avg / team_avg).to_dict()
        rebound_rate = g.groupby("team")["rebound_rate"].mean().to_dict()

    # --- Roster setup ---
    roster = (
        skaters_df[skaters_df[team_col].isin([team_a, team_b])]
        [[player_col, team_col]]
        .rename(columns={player_col: "player", team_col: "team"})
        .drop_duplicates("player")
        .reset_index(drop=True)
    )
    shots_df = shots_df.rename(
        columns={player_col_shots: "player", game_col: "gameid", sog_col: "sog", goal_col: "goal"}
    )
    shots_df["player"] = shots_df["player"].astype(str).str.strip()
    roster["player"] = roster["player"].astype(str).str.strip()
    grouped = {n.lower(): g.sort_values("gameid") for n, g in shots_df.groupby(shots_df["player"].str.lower())}

    results, total = [], len(roster)
    progress = st.progress(0)

    for i, row in enumerate(roster.itertuples(index=False), 1):
        player, team = row.player, row.team
        df_p = grouped.get(str(player).lower(), pd.DataFrame())
        if df_p.empty:
            progress.progress(i / total)
            continue

        game_sogs = df_p.groupby("gameid")[["sog","goal"]].sum().reset_index().sort_values("gameid")
        sog_values = game_sogs["sog"].tolist()
        last3, last5, last10 = sog_values[-3:], sog_values[-5:], sog_values[-10:]
        l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)
        season_avg = np.mean(sog_values)
        trend = 0 if pd.isna(l10) or l10 == 0 else (l3 - l10) / l10
        base_proj = np.nansum([0.5*l3, 0.3*l5, 0.2*l10])

        opp = team_b if team == team_a else team_a
        goalie_factor = np.clip(goalie_adj.get(opp, 1.0), 0.7, 1.3)
        adj_proj = base_proj * (0.7 + 0.3 * goalie_factor)
        adj_proj = max(0, round(adj_proj, 2))

        # --- Poisson probability ---
        lambda_recent = np.mean(df_p.groupby("gameid")["sog"].sum().tail(10))
        lambda_season = np.mean(df_p.groupby("gameid")["sog"].sum())
        if np.isnan(lambda_recent): lambda_recent = 0
        if np.isnan(lambda_season): lambda_season = 0
        lambda_blend = 0.7*lambda_recent + 0.3*lambda_season
        prob_hit_proj = 1 - poisson.cdf(np.floor(adj_proj) - 1, mu=lambda_blend)
        prob_hit_proj_pct = round(prob_hit_proj * 100, 1) if not pd.isna(prob_hit_proj) else np.nan

        # --- Usage-Adjusted Regression Indicator ---
        regression_flag = "Unknown"
        if toi_col and gp_col and player.lower() in skaters_df[player_col].str.lower().values:
            row_skat = skaters_df[skaters_df[player_col].str.lower() == player.lower()]
            season_toi = pd.to_numeric(row_skat[toi_col], errors="coerce").mean()
            games_played = pd.to_numeric(row_skat[gp_col], errors="coerce").mean()
            if season_toi > 0 and games_played > 0:
                avg_toi = season_toi / games_played
                sog_per60 = (season_avg / avg_toi) * 60
                recent_per60 = (lambda_recent / avg_toi) * 60
                rate_change = (recent_per60 - sog_per60) / sog_per60 if sog_per60 > 0 else 0
                if rate_change > 0.15:
                    regression_flag = "Usage-Driven Breakout"
                elif rate_change < -0.15:
                    regression_flag = "Usage Drop"
                else:
                    regression_flag = "Stable"

        # --- American odds conversion ---
        p = min(max(prob_hit_proj, 0.001), 0.999)
        if p >= 0.5:
            odds_val = -100 * (p / (1 - p))
        else:
            odds_val = 100 * ((1 - p) / p)
        odds_val = round(odds_val)
        implied_odds = f"+{odds_val}" if odds_val > 0 else str(odds_val)

        results.append({
            "Player": player, "Team": team,
            "Season Avg": round(season_avg, 2),
            "Trend Score": round(trend, 3),
            "Final Projection": adj_proj,
            "Prob ≥ Projection (%)": prob_hit_proj_pct,
            "Playable Odds": implied_odds,
            "Regression Indicator": regression_flag
        })
        progress.progress(i / total)
    progress.empty()

    df = pd.DataFrame(results)

    # --- Trend color ---
    def trend_color(v):
        if pd.isna(v): return "–"
        v = max(min(v, 0.5), -0.5)
        n = v + 0.5
        if n < 0.5:
            r, g, b = 255, int(255*(n*2)), 0
        else:
            r, g, b = int(255*(1-(n-0.5)*2)), 255, 0
        color = f"rgb({r},{g},{b})"
        t = "▲" if v > 0.05 else ("▼" if v < -0.05 else "–")
        txt = "#000" if abs(v) < 0.2 else "#fff"
        return f"<div style='background:{color};color:{txt};font-weight:600;border-radius:6px;padding:4px 8px;text-align:center;'>{t}</div>"
    df["Trend"] = df["Trend Score"].apply(trend_color)

    cols = ["Player","Team","Trend","Final Projection","Prob ≥ Projection (%)",
            "Playable Odds","Season Avg","Regression Indicator"]
    vis = df[[c for c in cols if c in df.columns]].sort_values("Final Projection",ascending=False)
    st.session_state.results = vis

# ---------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------
if st.session_state.results is not None:
    st.markdown("### 📊 Player Projections + Regression Insight")
    html_table = st.session_state.results.to_html(index=False, escape=False)
    st.markdown(f"<div style='overflow-x:auto'>{html_table}</div>", unsafe_allow_html=True)
