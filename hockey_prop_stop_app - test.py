# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — L5 Probability Update (TEST MODE, Stronger Line Adj Penalty)
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
st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("🧪 TEST MODE — You’re editing and running the sandbox version. Changes here will NOT affect your main app.")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
        <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
    </div>
    <h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
    <p style='text-align:center;color:#D6D6D6;'>
        Weighted L10/L5/L3 projections with strong asymmetric Line Adj impact
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
injuries_file = st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helper Functions
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
# Cached Data Load
# ---------------------------------------------------------------
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
        if injuries.empty:
            injuries = load_file(injuries_file)
        if not injuries.empty:
            injuries.columns = injuries.columns.str.lower().str.strip()
            if "player" in injuries.columns:
                injuries["player"] = injuries["player"].astype(str).str.strip().str.lower()
    return skaters, shots, goalies, lines, teams, injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()
st.success("✅ Data loaded successfully.")

# ---------------------------------------------------------------
# Normalize Columns
# ---------------------------------------------------------------
for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns = df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
shots_df = shots_df.rename(columns={player_col_shots: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)

# ---------------------------------------------------------------
# Team Selection
# ---------------------------------------------------------------
teams = sorted(skaters_df[team_col].dropna().unique().tolist())
col1, col2 = st.columns(2)
with col1: team_a = st.selectbox("Select Team A", teams)
with col2: team_b = st.selectbox("Select Team B", [t for t in teams if t != team_a])
st.markdown("---")

# ---------------------------------------------------------------
# Build Model — Strong Asymmetric Line Adj Impact
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results = []
    skaters = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
    roster = skaters[[player_col, team_col]].rename(columns={player_col:"player", team_col:"team"}).drop_duplicates("player")
    grouped = {n.lower():g.sort_values(game_col) for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    # Line adjustment
    line_adj = {}
    if not lines_df.empty and "line pairings" in lines_df.columns:
        l = lines_df.copy()
        l["games"] = pd.to_numeric(l["games"], errors="coerce").fillna(0)
        l["sog against"] = pd.to_numeric(l["sog against"], errors="coerce").fillna(0)
        l = l.groupby(["line pairings","team"], as_index=False).agg({"games":"sum","sog against":"sum"})
        l["sog_against_per_game"] = np.where(l["games"]>0, l["sog against"]/l["games"], np.nan)
        team_avg = l.groupby("team")["sog_against_per_game"].mean()
        league_avg = team_avg.mean()
        l["line_factor"] = (l["sog_against_per_game"] / league_avg).clip(0.7,1.3)
        line_adj = l.copy()

    # Goalie adjustment
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

        l3 = np.mean(sog_values[-3:]) if len(sog_values)>=3 else np.mean(sog_values)
        l5 = np.mean(sog_values[-5:]) if len(sog_values)>=5 else np.mean(sog_values)
        l10 = np.mean(sog_values[-10:]) if len(sog_values)>=10 else np.mean(sog_values)
        baseline = (0.55*l10) + (0.30*l5) + (0.15*l3)

        # --- Line factor (steeper penalty, softer boost) ---
        line_factor_internal = 1.0
        if not isinstance(line_adj,dict):
            last_name = str(player).split()[-1].lower()
            m = line_adj[line_adj["line pairings"].str.contains(last_name,case=False,na=False)]
            if not m.empty:
                line_factor_internal = np.average(m["line_factor"],weights=m["games"])
        if line_factor_internal < 1:
            line_term = -3.0 * (1 - line_factor_internal) ** 2.0  # very strong penalty
        else:
            line_term = 0.6 * (line_factor_internal - 1.0) ** 1.2  # modest boost

        # --- Goalie factor ---
        opp_team = team_b if team == team_a else team_a
        goalie_factor = goalie_adj.get(opp_team, 1.0)
        goalie_term = (goalie_factor - 1.0) * 0.2

        # --- Form factor ---
        form_flag, form_term = "⚪ Neutral Form", 0.0
        try:
            season_toi = pd.to_numeric(skaters_df.loc[skaters_df[player_col].str.lower()==player.lower(), "icetime"], errors="coerce").mean()
            games_played = pd.to_numeric(skaters_df.loc[skaters_df[player_col].str.lower()==player.lower(), "games"], errors="coerce").mean()
            if season_toi > 0 and games_played >= 10:
                avg_toi = (season_toi / games_played) / 60.0
                sog_per60 = (np.mean(sog_values) / avg_toi) * 60
                recent_per60 = (baseline / avg_toi) * 60 if avg_toi>0 else 0
                usage_delta = (recent_per60 - sog_per60)/sog_per60 if sog_per60>0 else 0
                if usage_delta > 0.10:
                    form_flag, form_term = "🟢 Above-Baseline Form", 0.05
                elif usage_delta < -0.10:
                    form_flag, form_term = "🔴 Below-Baseline Form", -0.05
        except Exception: pass

        # --- Combined Projection ---
        lam = baseline * (1 + goalie_term + form_term) * (1 + line_term * 2.0)
        lam = np.clip(lam, 0.1, 7.0)

        # --- Probability & Odds ---
        prob = 1 - poisson.cdf(np.floor(lam) - 1, mu=lam)
        p = min(max(prob, 0.001), 0.999)
        odds = -100*(p/(1-p)) if p>=0.5 else 100*((1-p)/p)
        implied_odds = f"{'+' if odds>0 else ''}{int(odds)}"

        display_line_adj = 1 / line_factor_internal if line_factor_internal != 0 else 1.0

        results.append({
            "Player":player,"Team":team,"Injury":"",
            "Trend Score":round((l5 - l10)/l10 if l10>0 else 0,3),
            "Final Projection":round(lam,2),
            "Prob ≥ Projection (%) L5":round(p*100,1),
            "Playable Odds":implied_odds,
            "Season Avg":round(np.mean(sog_values),2),
            "Line Adj":round(display_line_adj,2),
            "Form Indicator":form_flag
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run + Display
# ---------------------------------------------------------------
if st.button("🚀 Run Model"):
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")
    df = build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df)
    if "Injury" not in df.columns: df["Injury"] = ""
    df = df.sort_values("Final Projection", ascending=False).reset_index(drop=True)
    st.session_state.results_raw = df.copy()
    st.success("✅ Model built successfully!")

if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
    df = st.session_state.results_raw.copy()

    def trend_color(v):
        if pd.isna(v): return "–"
        if v > 0.05: color, symbol = "#00B140","▲"
        elif v < -0.05: color, symbol = "#E63946","▼"
        else: color, symbol = "#6C7A89","–"
        return f"<div style='background:{color};color:#fff;font-weight:600;border-radius:6px;padding:4px 8px;text-align:center;'>{symbol}</div>"

    df["Trend"] = df["Trend Score"].apply(trend_color)

    cols = [
        "Player","Team","Injury","Trend","Final Projection",
        "Prob ≥ Projection (%) L5","Playable Odds",
        "Season Avg","Line Adj","Form Indicator"
    ]
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
        <div style='overflow-x:auto;height:620px;'>{html_table}</div>
        """,height=650,scrolling=True)
