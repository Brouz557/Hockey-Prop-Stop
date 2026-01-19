# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — L5 Probability Update (Injury Modal Fixed)
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
            if os.path.exists(full):
                return full
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        skaters = load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
        shots   = load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
        goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
        lines   = load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
        teams   = load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")

        injuries_path_candidates = [
            "injuries.xlsx", "Injuries.xlsx", "./injuries.xlsx", "data/injuries.xlsx",
            "/mount/src/hockey-prop-stop/injuries.xlsx"
        ]
        injuries = pd.DataFrame()
        for path in injuries_path_candidates:
            if os.path.exists(path):
                injuries = load_file(open(path, "rb"))
                break
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
    file_candidates = ["data/SHOT DATA.xlsx", "/mount/src/hockey-prop-stop/data/SHOT DATA.xlsx", "SHOT DATA.xlsx"]
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
st.markdown(f"🕒 **Data last updated:** {last_update or 'Unknown'}")

# ---------------------------------------------------------------
# Normalize Columns
# ---------------------------------------------------------------
for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

team_col = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
shots_df = shots_df.rename(columns={player_col_shots: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()

sog_col  = next((c for c in shots_df.columns if "sog" in c), None)
goal_col = next((c for c in shots_df.columns if "goal" in c), None)
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
# Build Model
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

    for row in roster.itertuples(index=False):
        player, team = row.player, row.team
        df_p = grouped.get(player.lower(), pd.DataFrame())
        if df_p.empty: continue

        game_sogs = df_p.groupby(game_col)[["sog","goal"]].sum().reset_index().sort_values(game_col)
        sog_values = game_sogs["sog"].tolist()
        if not sog_values: continue

        last3 = sog_values[-3:] if len(sog_values)>=3 else sog_values
        last5 = sog_values[-5:] if len(sog_values)>=5 else sog_values
        last10 = sog_values[-10:] if len(sog_values)>=10 else sog_values

        l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)
        season_avg = np.mean(sog_values)
        trend = (l5 - l10)/l10 if l10>0 else 0

        line_factor = 1.0
        if not isinstance(line_adj,dict):
            last_name = str(player).split()[-1].lower()
            m = line_adj[line_adj["line pairings"].str.contains(last_name,case=False,na=False)]
            if not m.empty:
                line_factor = np.average(m["line_factor"],weights=m["games"])
            line_factor = np.clip(line_factor,0.7,1.3)

        lam = l5
        line = round(lam, 2)
        prob = 1 - poisson.cdf(np.floor(line) - 1, mu=lam)
        p = min(max(prob, 0.001), 0.999)
        odds = -100*(p/(1-p)) if p>=0.5 else 100*((1-p)/p)
        implied_odds = f"{'+' if odds>0 else ''}{int(odds)}"

        # --- Form Indicator ---
        form_flag = "⚪ Neutral Form"
        try:
            season_toi = pd.to_numeric(
                skaters_df.loc[skaters_df[player_col].str.lower() == player.lower(), "icetime"],
                errors="coerce"
            ).mean()
            games_played = pd.to_numeric(
                skaters_df.loc[skaters_df[player_col].str.lower() == player.lower(), "games"],
                errors="coerce"
            ).mean()
            if season_toi > 0 and games_played >= 10:
                avg_toi = (season_toi / games_played) / 60.0
                sog_per60 = (season_avg / avg_toi) * 60
                blended_recent = 0.7 * l5 + 0.3 * l10
                recent_per60 = (blended_recent / avg_toi) * 60 if avg_toi>0 else 0
                usage_delta = (recent_per60 - sog_per60)/sog_per60 if sog_per60>0 else 0
                if usage_delta > 0.10: form_flag = "🟢 Above-Baseline Form"
                elif usage_delta < -0.10: form_flag = "🔴 Below-Baseline Form"
        except Exception:
            pass

        # --- Injury Modal ---
        injury_html = ""
        if not injuries_df.empty and {"player","team"}.issubset(injuries_df.columns):
            player_lower = player.lower().strip()
            last_name = player_lower.split()[-1]
            team_lower = team.lower().strip()
            match = injuries_df[
                injuries_df["player"].str.lower().str.endswith(last_name)
                & injuries_df["team"].str.lower().str.strip().eq(team_lower)
            ]
            if not match.empty:
                note = str(match.iloc[0].get("injury note","")).strip()
                injury_type = str(match.iloc[0].get("injury type","")).strip()
                date_injury = str(match.iloc[0].get("date of injury","")).strip()
                tooltip = "\\n".join([p for p in [injury_type,note,date_injury] if p]) or "Injury info unavailable"

                # Unique modal ID per player
                modal_id = f"injuryModal_{player_lower.replace(' ', '_')}"

                injury_html = (
                    "<span style='cursor:pointer;' title='Tap or click for injury info' "
                    "onclick=\""
                    f"const msg = `{tooltip.replace('`','\\`').replace(chr(10),' ')}`;"
                    "const modal = document.createElement('div');"
                    "modal.innerHTML = `"
                    f"<div id='{modal_id}' style='position:fixed;top:0;left:0;width:100%;height:100%;"
                    "background:rgba(0,0,0,0.6);display:flex;align-items:center;"
                    "justify-content:center;z-index:9999;'>"
                    "<div style='background:#1e1e1e;padding:20px 25px;border-radius:10px;"
                    "widt
