# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Stable Version (Reverted)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io
import streamlit.components.v1 as components
from scipy.stats import poisson
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with improved goal projections
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
if st.sidebar.button("🔄 Force Reload Data Cache"):
    st.cache_data.clear()
    st.experimental_rerun()

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def load_file(file):
    if not file:
        return pd.DataFrame()
    try:
        return pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file):
    skaters = load_file(skaters_file)
    shots = load_file(shots_file)
    return skaters, shots

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df = load_all_data(skaters_file, shots_file)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload both skaters and shot data.")
    st.stop()

# Normalize columns (lowercase + strip)
for df in [skaters_df, shots_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

# Detect key columns
team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = next((c for c in skaters_df.columns if "name" in c), None)
sog_col = next((c for c in shots_df.columns if "sog" in c), None)
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)

if not all([team_col, player_col, sog_col, player_col_shots]):
    st.error("❌ Could not find required columns in uploaded files.")
    st.stop()

# ---------------------------------------------------------------
# Team Selection
# ---------------------------------------------------------------
all_teams = sorted(skaters_df[team_col].dropna().unique().tolist())
col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A", all_teams)
with col2:
    team_b = st.selectbox("Select Team B", [t for t in all_teams if t != team_a])
st.markdown("---")

# ---------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df):
    results = []
    pos_col = next((c for c in skaters_df.columns if "pos" in c), None)
    roster = skaters_df[skaters_df[team_col].isin([team_a, team_b])][[player_col, team_col, pos_col]]
    roster = roster.rename(columns={player_col: "player", team_col: "team", pos_col: "position"}).drop_duplicates("player")
    shots_df = shots_df.rename(columns={player_col_shots: "player", sog_col: "sog"})
    grouped = {n.lower(): g for n, g in shots_df.groupby(shots_df["player"].str.lower())}

    for row in roster.itertuples(index=False):
        player, team, position = row.player, row.team, row.position
        df_p = grouped.get(str(player).lower(), pd.DataFrame())
        if df_p.empty:
            continue
        sog_values = df_p["sog"].tolist()
        if not sog_values:
            continue

        # --- Core metrics ---
        last5 = sog_values[-5:] if len(sog_values) >= 5 else sog_values
        l5 = np.mean(last5)
        season_avg = np.mean(sog_values)
        trend = (l5 - season_avg) / season_avg if season_avg > 0 else 0

        # --- Probability (Poisson) ---
        lam = l5
        line = round(lam, 2)
        prob = 1 - poisson.cdf(np.floor(line) - 1, mu=lam)
        p = min(max(prob, 0.001), 0.999)

        # --- Convert to American odds ---
        odds = -100 * (p / (1 - p)) if p >= 0.5 else 100 * ((1 - p) / p)
        implied_odds = f"{'+' if odds > 0 else ''}{int(odds)}"

        results.append({
            "Player": player,
            "Team": team,
            "Position": position,
            "Season Avg": round(season_avg, 2),
            "L5 Avg": round(l5, 2),
            "Final Projection": round(line, 2),
            "Prob ≥ Projection (%) L5": round(p * 100, 1),
            "Playable Odds": implied_odds,
            "Trend Score": round(trend, 3)
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
# Display Results
# ---------------------------------------------------------------
if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
    df = st.session_state.results_raw.copy()

    # --- Trend Color Formatting ---
    def trend_color(v):
        if pd.isna(v):
            return "–"
        v = max(min(v, 0.5), -0.5)
        n = v + 0.5
        if n < 0.5:
            r, g, b = 255, int(255 * (n * 2)), 0
        else:
            r, g, b = int(255 * (1 - (n - 0.5) * 2)), 255, 0
        color = f"rgb({r},{g},{b})"
        t = "▲" if v > 0.05 else ("▼" if v < -0.05 else "–")
        txt = "#000" if abs(v) < 0.2 else "#fff"
        return f"<div style='background:{color};color:{txt};font-weight:600;border-radius:6px;padding:4px 8px;text-align:center;'>{t}</div>"

    df["Trend"] = df["Trend Score"].apply(trend_color)

    # --- Visible Columns ---
    cols = [
        "Player", "Team", "Position", "Trend",
        "Final Projection", "Prob ≥ Projection (%) L5",
        "Playable Odds", "Season Avg", "L5 Avg"
    ]
    vis = df[[c for c in cols if c in df.columns]]

    # --- Render Table with Sticky Headers ---
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
