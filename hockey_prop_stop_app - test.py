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
# Team Abbreviation Normalization
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ":"NJD","LA":"LAK","SJ":"SJS","TB":"TBL","ARI":"ARI","ANA":"ANA",
    "BOS":"BOS","BUF":"BUF","CAR":"CAR","CBJ":"CBJ","CGY":"CGY","CHI":"CHI",
    "COL":"COL","DAL":"DAL","DET":"DET","EDM":"EDM","FLA":"FLA","MIN":"MIN",
    "MTL":"MTL","NSH":"NSH","NYI":"NYI","NYR":"NYR","OTT":"OTT","PHI":"PHI",
    "PIT":"PIT","SEA":"SEA","STL":"STL","TOR":"TOR","VAN":"VAN","VGK":"VGK",
    "WSH":"WSH","WPG":"WPG"
}

st.set_page_config(page_title="Puck Shotz Hockey Analytics", layout="wide", page_icon="🏒")
st.warning("Production Version")

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])
injuries_file= st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def load_file(f):
    if not f: return pd.DataFrame()
    try:
        return pd.read_excel(f) if f.name.lower().endswith(".xlsx") else pd.read_csv(f)
    except:
        return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except:
        return pd.DataFrame()

def load_data(uploader, fallback):
    return load_file(uploader) if uploader else safe_read(fallback)

@st.cache_data(show_spinner=False)
def load_all():
    skaters = load_data(skaters_file, "Skaters.xlsx")
    shots   = load_data(shots_file,   "SHOT DATA.xlsx")
    goalies = load_data(goalies_file, "GOALTENDERS.xlsx")
    lines   = load_data(lines_file,   "LINE DATA.xlsx")
    teams   = load_data(teams_file,   "TEAMS.xlsx")
    injuries= load_file(injuries_file)
    return skaters, shots, goalies, lines, teams, injuries

skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all()

if skaters_df.empty or shots_df.empty:
    st.warning("Upload required files")
    st.stop()

# ---------------------------------------------------------------
# Normalize columns
# ---------------------------------------------------------------
for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name" if "name" in skaters_df.columns else skaters_df.columns[0]

# 🔒 SAFE rename (this fixes your KeyError permanently)
shots_df = shots_df.rename(
    columns={next((c for c in shots_df.columns if "player" in c or "name" in c), "player"): "player"}
)
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c and "id" in c)

# ---------------------------------------------------------------
# Line input
# ---------------------------------------------------------------
col_run, col_line = st.columns([3,1])
with col_run:
    run_model = st.button("🚀 Run Model")
with col_line:
    line_test = st.number_input("Line to Test",0.0,10.0,3.5,0.5)
    st.session_state.line_test_val = line_test

# ---------------------------------------------------------------
# BUILD MODEL (unchanged)
# ---------------------------------------------------------------
def build_model():
    results=[]
    roster = skaters_df[[player_col,team_col]].drop_duplicates()
    grouped = {n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    for _,r in roster.iterrows():
        player, team = r[player_col], r[team_col]
        dfp = grouped.get(str(player).lower(), pd.DataFrame())
        if dfp.empty or "sog" not in dfp: continue

        sog_vals = dfp.groupby(game_col)["sog"].sum().tolist()
        if not sog_vals: continue

        lam = np.mean(sog_vals[-5:])
        prob = (1 - poisson.cdf(line_test-1, lam)) * 100

        results.append({
            "Player":player,
            "Team":team,
            "Final Projection":round(lam,2),
            "Prob ≥ Line (%)":round(prob,1),
            "Line Adj":1.0,
            "Exp Goals (xG)":round(0.1*lam,2),
            "L10 Shots":", ".join(map(str,sog_vals[-10:]))
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# RUN MODEL
# ---------------------------------------------------------------
if run_model:
    st.session_state.results = build_model()
    st.success("Model ran")

# ---------------------------------------------------------------
# DISPLAY + 🟢🟡🔴 SIGNAL
# ---------------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state.results.copy()
    test_line = st.session_state.line_test_val

    def model_score_line(row):
        score = 0.0
        diff = row["Final Projection"] - test_line
        if diff >= 1.0: score += 1.5
        elif diff >= 0.4: score += 1.0
        if row["Prob ≥ Line (%)"] >= 60: score += 1.0
        elif row["Prob ≥ Line (%)"] >= 55: score += 0.5
        if row["Line Adj"] >= 1.05: score += 1.0
        try:
            if float(row["Exp Goals (xG)"]) >= 0.30: score += 1.0
        except:
            pass
        return round(score,2)

    def model_light(score):
        if score >= 4.0: return "<span style='color:#00FF00;font-weight:bold;'>🟢</span>"
        elif score >= 2.5: return "<span style='color:#FFD700;font-weight:bold;'>🟡</span>"
        else: return "<span style='color:#FF4B4B;font-weight:bold;'>🔴</span>"

    df["Model Score"] = df.apply(model_score_line, axis=1)
    df["Signal"] = df["Model Score"].apply(model_light)

    cols = ["Signal","Model Score","Player","Team","Final Projection","Prob ≥ Line (%)","L10 Shots"]
    html_table = df[cols].to_html(index=False, escape=False)
    components.html(html_table, height=600)
