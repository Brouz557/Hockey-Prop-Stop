# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Logo Overlay + Fix
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, datetime, pytz, subprocess, html, json, requests
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
    <p style='text-align:center;color:#B0B0B0;'>Automatic matchup analytics with logo-overlay buttons and live line testing.</p>
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
# Helper functions
# ---------------------------------------------------------------
def load_file(file):
    try:
        return pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_all(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    base_paths = [
        ".", "data", "/app/hockey-prop-stop/data", "/mount/src/hockey-prop-stop/data"
    ]
    def find_file(name):
        for p in base_paths:
            full = os.path.join(p, name)
            if os.path.exists(full):
                return full
        return None
    def try_load(name, file_uploader):
        path = find_file(name)
        if file_uploader is not None:
            return load_file(file_uploader)
        if path:
            return load_file(open(path,"rb"))
        return pd.DataFrame()

    with contextlib.redirect_stdout(io.StringIO()):
        skaters  = try_load("Skaters.xlsx", skaters_file)
        shots    = try_load("SHOT DATA.xlsx", shots_file)
        goalies  = try_load("GOALTENDERS.xlsx", goalies_file)
        lines    = try_load("LINE DATA.xlsx", lines_file)
        teams    = try_load("TEAMS.xlsx", teams_file)
        injuries = try_load("injuries.xlsx", injuries_file)

    return skaters, shots, goalies, lines, teams, injuries

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data files.")
    st.stop()
st.success("✅ Data loaded successfully.")

# ---------------------------------------------------------------
# Normalize columns
# ---------------------------------------------------------------
for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
shots_df = shots_df.rename(columns={player_col_shots: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)

# ---------------------------------------------------------------
# Fetch today's games (from ESPN)
# ---------------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_todays_games():
    try:
        today = datetime.date.today().strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={today}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        games = []
        for e in data.get("events", []):
            c = e["competitions"][0]["competitors"]
            home = c[0]["team"]
            away = c[1]["team"]
            games.append({
                "away": away["abbreviation"],
                "home": home["abbreviation"],
                "away_logo": away["logo"],
                "home_logo": home["logo"],
                "matchup": f"{away['abbreviation']}@{home['abbreviation']}"
            })
        return games
    except Exception as e:
        st.error(f"Could not fetch games: {e}")
        return []

matchups = fetch_todays_games()

# ---------------------------------------------------------------
# Display matchups (logo overlay buttons)
# ---------------------------------------------------------------
if matchups:
    st.subheader("🏒 Today's Matchups")
    col_count = 3
    cols = st.columns(col_count)
    for i, m in enumerate(matchups):
        col = cols[i % col_count]
        with col:
            html_btn = f"""
            <div style='display:flex;align-items:center;justify-content:center;
                        background-color:#0A3A67;border-radius:10px;padding:6px 12px;
                        margin:4px;cursor:pointer;color:white;font-weight:600;'>
                <img src='{m['away_logo']}' style='height:25px;margin-right:6px;'>
                {m['away']} @ {m['home']}
                <img src='{m['home_logo']}' style='height:25px;margin-left:6px;'>
            </div>
            """
            if st.button(m["matchup"]):
                st.session_state.selected_game = m["matchup"]
else:
    st.warning("⚠️ No NHL games found for today.")

# ---------------------------------------------------------------
# Line test input and model trigger
# ---------------------------------------------------------------
col_run, col_line = st.columns([3,1])
with col_run:
    run_model = st.button("🚀 Run Model (All Games Today)")
with col_line:
    line_test = st.number_input("Line to Test", min_value=0.0, max_value=10.0, value=3.5, step=0.5)

# ---------------------------------------------------------------
# Build model
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
        g["shots_per_game"] = g["shots against"]/g["games"]
        league_avg_sa = g["shots_per_game"].mean()
        g["goalie_factor"] = (g["shots_per_game"]/league_avg_sa).clip(0.7,1.3)
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
            if not m.empty:
                line_factor_internal = np.average(m["line_factor"], weights=m["games"])

        opp_team = team_b if team == team_a else team_a
        goalie_factor = goalie_adj.get(opp_team, 1.0)
        goalie_term = (goalie_factor - 1.0) * 0.2

        lam_base = baseline * (1 + goalie_term)
        if line_factor_internal >= 1:
            scale = 1 + 7.0 * (line_factor_internal - 1.0) ** 1.5
        else:
            scale = max(0.05, line_factor_internal ** 3.5)
        lam = lam_base * scale

        prob = 1 - poisson.cdf(np.floor(line_test) - 1, mu=max(lam,0.01))
        odds = -100*(prob/(1-prob)) if prob>=0.5 else 100*((1-prob)/prob)
        implied = f"{'+' if odds>0 else ''}{int(odds)}"

        trend = (l5 - l10)/l10 if l10>0 else 0
        form = "⚪ Neutral Form"
        if trend>0.1: form="🟢 Above-Baseline Form"
        elif trend<-0.1: form="🔴 Below-Baseline Form"

        results.append({
            "Player":player,"Team":team,"Trend Score":round(trend,3),
            "Final Projection":round(lam,2),
            f"Prob ≥ {line_test} (%)":round(prob*100,1),
            f"Playable Odds ({line_test})":implied,
            "Season Avg":round(np.mean(sog_values),2),
            "Line Adj":round(line_factor_internal,2),
            "Form Indicator":form
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run model
# ---------------------------------------------------------------
if run_model:
    all_results = []
    for m in matchups:
        df_match = build_model(m["away"], m["home"], skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df)
        df_match["Matchup"] = f"{m['away']}@{m['home']}"
        all_results.append(df_match)
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values(["Team","Final Projection","Line Adj"], ascending=[True,False,False]).reset_index(drop=True)
    st.session_state.results = combined
    st.success("✅ Model built successfully!")

# ---------------------------------------------------------------
# Display
# ---------------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state.results.copy()
    vis_cols = ["Matchup","Player","Team","Trend Score","Final Projection",
                f"Prob ≥ {line_test} (%)",f"Playable Odds ({line_test})",
                "Season Avg","Line Adj","Form Indicator"]
    vis = df[[c for c in vis_cols if c in df.columns]]
    st.dataframe(vis, use_container_width=True)
