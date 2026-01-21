# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Clickable Matchup Toggle + Live Line Test
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, datetime, requests, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("🧪 TEST MODE — Clickable Matchups + Live Line Testing enabled.")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
        <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
    </div>
    <h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
    <p style='text-align:center;color:#D6D6D6;'>Clickable matchups with Form, Trend, Season Avg, and live line testing.</p>
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
# Helpers
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

@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
    def find_file(name):
        for p in base_paths:
            full = os.path.join(p, name)
            if os.path.exists(full): return full
        return None

    skaters = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
    shots   = load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
    goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
    lines   = load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
    teams   = load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")

    injuries = load_file(injuries_file) if injuries_file else pd.DataFrame()
    if not injuries.empty:
        injuries.columns = injuries.columns.str.lower().str.strip()
        if "player" in injuries.columns:
            injuries["player"] = injuries["player"].astype(str).str.strip().str.lower()
    return skaters, shots, goalies, lines, teams, injuries

# ---------------------------------------------------------------
# Load + Normalize
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data.")
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns = df.columns.str.lower().str.strip()

if "player" not in shots_df.columns:
    candidate = next((c for c in shots_df.columns if "player" in c or "name" in c or "skater" in c), None)
    if candidate:
        shots_df = shots_df.rename(columns={candidate: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()

# ---------------------------------------------------------------
# Controls
# ---------------------------------------------------------------
col_run, col_line = st.columns([3,1])
with col_run:
    run_model = st.button("🚀 Run Model (All Games Today)")
with col_line:
    line_test = st.number_input("Line to Test", min_value=0.0, max_value=10.0, value=3.5, step=0.5)

# ---------------------------------------------------------------
# Fetch ESPN Games
# ---------------------------------------------------------------
def fetch_espn_games():
    today = datetime.datetime.now().strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={today}"
    try:
        data = requests.get(url, timeout=10).json()
        games = []
        for event in data.get("events", []):
            c = event.get("competitions", [{}])[0]
            teams = c.get("competitors", [])
            if len(teams) == 2:
                home, away = teams[0]["team"], teams[1]["team"]
                games.append({
                    "home": home["abbreviation"], "away": away["abbreviation"],
                    "home_full": home["displayName"], "away_full": away["displayName"],
                    "home_logo": home["logo"], "away_logo": away["logo"]
                })
        return games
    except Exception as e:
        st.error(f"Failed to fetch games: {e}")
        return []

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df):
    results = []
    team_col = next((c for c in skaters_df.columns if "team" in c), None)
    player_col = "name" if "name" in skaters_df.columns else "player"
    game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)

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
        l["line_factor"] = (league_avg / l["sog_against_per_game"]).clip(0.7,1.3)
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
        season_avg = np.mean(sog_values)

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
        lam_base = baseline * (1 + (goalie_factor - 1.0) * 0.2)
        if line_factor_internal >= 1:
            scale = 1 + 7.0 * (line_factor_internal - 1.0) ** 1.5
        else:
            scale = max(0.05, line_factor_internal ** 3.5)
        lam = lam_base * scale

        form_flag = "⚪ Neutral Form"
        try:
            season_toi = pd.to_numeric(skaters_df.loc[skaters_df[player_col].str.lower()==player.lower(),"icetime"], errors="coerce").mean()
            games_played = pd.to_numeric(skaters_df.loc[skaters_df[player_col].str.lower()==player.lower(),"games"], errors="coerce").mean()
            if season_toi>0 and games_played>=10:
                avg_toi = (season_toi/games_played)/60.0
                sog_per60 = (np.mean(sog_values)/avg_toi)*60
                blended_recent = 0.7*l5+0.3*l10
                recent_per60 = (blended_recent/avg_toi)*60 if avg_toi>0 else 0
                usage_delta = (recent_per60-sog_per60)/sog_per60 if sog_per60>0 else 0
                if usage_delta>0.10: form_flag="🟢 Above-Baseline Form"
                elif usage_delta<-0.10: form_flag="🔴 Below-Baseline Form"
        except Exception:
            pass

        trend = (l5 - l10)/l10 if l10>0 else 0
        trend_cell = "▲" if trend>0.05 else ("▼" if trend<-0.05 else "–")

        results.append({
            "Player": player,
            "Team": team,
            "Season Avg": round(season_avg,2),
            "Trend": trend_cell,
            "Form Indicator": form_flag,
            "Final Projection": round(lam,2),
            "Line Adj": round(line_factor_internal,2),
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model + Matchup Toggles
# ---------------------------------------------------------------
if run_model:
    games = fetch_espn_games()
    if not games:
        st.warning("⚠️ No NHL games found for today.")
    else:
        st.subheader("🏒 Today's Matchups")

        if "selected_game" not in st.session_state:
            st.session_state.selected_game = None

        combined = pd.DataFrame()
        for g in games:
            df = build_model(g["away"], g["home"], skaters_df, shots_df, goalies_df, lines_df, teams_df)
            df["Matchup"] = f"{g['away']} @ {g['home']}"
            combined = pd.concat([combined, df], ignore_index=True)

        st.session_state.results_base = combined.sort_values(["Team","Final Projection"], ascending=[True,False]).reset_index(drop=True)
        st.session_state.matchups = games
        st.success("✅ All games processed successfully!")

# ---------------------------------------------------------------
# Matchup Buttons + Filter Logic
# ---------------------------------------------------------------
if "results_base" in st.session_state:
    games = st.session_state.matchups
    base_df = st.session_state.results_base.copy()

    cols = st.columns(min(len(games), 4))
    for idx, g in enumerate(games):
        with cols[idx % len(cols)]:
            label = f"{g['away']} @ {g['home']}"
            active = st.session_state.selected_game == label
            if st.button(label, type="primary" if active else "secondary"):
                st.session_state.selected_game = None if active else label
                st.rerun()

    df = base_df
    if st.session_state.selected_game:
        away, home = st.session_state.selected_game.split(" @ ")
        df = df[df["Team"].isin([away, home])]

    lam_vals = df["Final Projection"].astype(float)
    probs = 1 - poisson.cdf(line_test - 1, mu=lam_vals.clip(lower=0.01))
    df[f"Prob ≥ {line_test} (%)"] = (probs*100).round(1)
    odds = np.where(probs>=0.5, -100*(probs/(1-probs)), 100*((1-probs)/probs))
    odds = np.clip(odds, -5000, 5000)
    df[f"Playable Odds ({line_test})"] = [f"{'+' if o>0 else ''}{int(o)}" for o in odds]

    vis_cols = [
        "Player","Team","Season Avg","Trend","Form Indicator",
        "Final Projection",f"Prob ≥ {line_test} (%)",f"Playable Odds ({line_test})","Line Adj"
    ]
    vis = df[[c for c in vis_cols if c in df.columns]]
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
        """, height=650, scrolling=True)
