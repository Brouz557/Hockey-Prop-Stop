# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — L5 Probability Update (TEST MODE)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, datetime, pytz, subprocess, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page Setup — must be first Streamlit call
# ---------------------------------------------------------------
st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")

# 🧪 TEST MODE WARNING — Safe sandbox banner
st.warning("🧪 TEST MODE — You’re editing and running the sandbox version. Changes here will NOT affect your main app.")

# ---------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
        <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
    </div>
    <h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
    <p style='text-align:center;color:#D6D6D6;'>
        Team-vs-Team matchup analytics with blended regression and L5-based probabilities
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
# 🕒 Data Last Updated
# ---------------------------------------------------------------
def get_shots_file_git_time():
    tz_cst = pytz.timezone("America/Chicago")
    for f in ["data/SHOT DATA.xlsx","/mount/src/hockey-prop-stop/data/SHOT DATA.xlsx","SHOT DATA.xlsx"]:
        if os.path.exists(f):
            try:
                git_time_str = subprocess.check_output(
                    ["git","log","-1","--format=%cd","--date=iso",f],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                if git_time_str:
                    git_time = datetime.datetime.fromisoformat(git_time_str.replace("Z","+00:00"))
                    return git_time.astimezone(tz_cst).strftime("%Y-%m-%d %I:%M %p CST")
            except Exception:
                continue
    return None

st.markdown(f"🕒 **Data last updated:** {get_shots_file_git_time() or 'Unknown'}")

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
# Build Model (with Corsi-based pace factor)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results = []
    skaters = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
    roster = skaters[[player_col, team_col]].rename(columns={player_col:"player", team_col:"team"}).drop_duplicates("player")
    grouped = {n.lower():g.sort_values(game_col) for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    # --- Line Adjustment ---
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

    # --- Goalie Adjustment ---
    goalie_adj = {}
    if not goalies_df.empty and {"team","shots against","games"}.issubset(goalies_df.columns):
        g = goalies_df.copy()
        g["shots against"] = pd.to_numeric(g["shots against"], errors="coerce").fillna(0)
        g["games"] = pd.to_numeric(g["games"], errors="coerce").fillna(1)
        g["shots_per_game"] = g["shots against"] / g["games"]
        league_avg_sa = g["shots_per_game"].mean()
        g["goalie_factor"] = (g["shots_per_game"] / league_avg_sa).clip(0.7,1.3)
        goalie_adj = g.groupby("team")["goalie_factor"].mean().to_dict()

    # --- Player Loop ---
    for row in roster.itertuples(index=False):
        player, team = row.player, row.team
        df_p = grouped.get(player.lower(), pd.DataFrame())
        if df_p.empty:
            continue

        sog_values = df_p.groupby(game_col)["sog"].sum().tolist()
        if not sog_values:
            continue

        last3 = sog_values[-3:] if len(sog_values)>=3 else sog_values
        last5 = sog_values[-5:] if len(sog_values)>=5 else sog_values
        last10 = sog_values[-10:] if len(sog_values)>=10 else sog_values
        l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)

        # --- Corsi-based Pace Factor ---
        pace_factor = 1.0
        try:
            if "on ice corsi" in skaters_df.columns:
                player_corsi = skaters_df.loc[
                    skaters_df[player_col].str.lower() == player.lower(), "on ice corsi"
                ]
                league_avg_corsi = skaters_df["on ice corsi"].mean()
                if not player_corsi.empty and league_avg_corsi > 0:
                    pace_factor = player_corsi.iloc[0] / league_avg_corsi
            elif not teams_df.empty and "corsi%" in teams_df.columns and "team" in teams_df.columns:
                team_corsi = teams_df.loc[
                    teams_df["team"].str.lower() == team.lower(), "corsi%"
                ]
                league_avg_corsi = teams_df["corsi%"].mean()
                if not team_corsi.empty and league_avg_corsi > 0:
                    pace_factor = team_corsi.iloc[0] / league_avg_corsi
        except Exception:
            pace_factor = 1.0

        # --- Base Weighted Shots ---
        baseline = ((0.55 * l10) + (0.30 * l5) + (0.15 * l3)) * pace_factor

        # --- Line and Goalie Adjustments ---
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

        # --- Probabilities ---
        hit_l3  = np.mean(np.array(last3)  >= lam)
        hit_l5  = np.mean(np.array(last5)  >= lam)
        hit_l10 = np.mean(np.array(last10) >= lam)
        empirical_prob = np.nanmean([0.55*hit_l10 + 0.30*hit_l5 + 0.15*hit_l3])
        poisson_prob = 1 - poisson.cdf(np.floor(lam) - 1, mu=max(lam, 0.01))
        final_prob = 0.6 * poisson_prob + 0.4 * empirical_prob

        # --- Form Indicator ---
        form_flag = "⚪ Neutral Form"
        try:
            season_toi = pd.to_numeric(
                skaters_df.loc[skaters_df[player_col].str.lower()==player.lower(),"icetime"], errors="coerce"
            ).mean()
            games_played = pd.to_numeric(
                skaters_df.loc[skaters_df[player_col].str.lower()==player.lower(),"games"], errors="coerce"
            ).mean()
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

        # --- Injury HTML ---
        injury_html = ""
        if not injuries_df.empty and {"player","team"}.issubset(injuries_df.columns):
            player_lower = player.lower().strip()
            last_name = player_lower.split()[-1]
            team_lower = team.lower().strip()
            match = injuries_df[
                injuries_df["team"].str.lower().str.strip().eq(team_lower)
                & injuries_df["player"].str.lower().str.endswith(last_name)
            ]
            if not match.empty:
                note = str(match.iloc[0].get("injury note","")).strip()
                injury_type = str(match.iloc[0].get("injury type","")).strip()
                date_injury = str(match.iloc[0].get("date of injury","")).strip()
                tooltip = "\n".join([p for p in [injury_type,note,date_injury] if p]) or "Injury info unavailable"
                safe = html.escape(tooltip)
                injury_html = f"<span style='cursor:pointer;' onclick='alert({json.dumps(safe)})' title='Tap or click for injury info'>🚑</span>"

        # --- Append Row ---
        results.append({
            "Player": player,
            "Team": team,
            "Injury": injury_html,
            "Trend Score": round(trend,3),
            "Final Projection": round(lam,2),
            "Prob ≥ Projection (%) L5": round(final_prob*100,1),
            "Season Avg": round(np.mean(sog_values),2),
            "Line Adj": round(line_factor_internal,2),
            "Form Indicator": form_flag,
            "L3 Shots": ", ".join(map(str,last3)),
            "L5 Shots": ", ".join(map(str,last5)),
            "L10 Shots": ", ".join(map(str,last10))
        })
    return pd.DataFrame(results)
