# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — L5 Probability Update (Injuries Fixed)
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
# Cached Data Load (with fixed Injuries file logic)
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

        # --- FIXED: Injuries file lookup ---
        injuries_path_candidates = [
            "Injuries.xlsx",
            "injuries.xlsx",
            "./Injuries.xlsx",
            "./injuries.xlsx",
            "data/Injuries.xlsx",
            "data/injuries.xlsx",
            "/mount/src/hockey-prop-stop/Injuries.xlsx",
            "/mount/src/hockey-prop-stop/injuries.xlsx",
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
# 🕒 Data Last Updated — Git Commit Timestamp
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
# Normalize Columns
# ---------------------------------------------------------------
for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns = df.columns.str.lower().str.strip()

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
# Cached Model Build
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results = []
    skaters = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
    roster = skaters[[player_col, team_col]].rename(columns={player_col:"player", team_col:"team"}).drop_duplicates("player")
    grouped = {n.lower():g.sort_values(game_col) for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    # --- Line Adjustments ---
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
                else: form_flag = "⚪ Neutral Form"
        except Exception: pass

        # --- Injury Flag with Hover Tooltip (fuzzy match) ---
        injury_html = ""
        if not injuries_df.empty and "player" in injuries_df.columns:
            player_lower = player.lower().strip()
            last_name = player_lower.split()[-1]
            match = injuries_df[injuries_df["player"].str.contains(last_name, case=False, na=False)]
            if not match.empty:
                note = str(match.iloc[0].get("injury note", "")).strip()
                injury_type = str(match.iloc[0].get("injury type", "")).strip()
                date_injury = str(match.iloc[0].get("date of injury", "")).strip()
                tooltip_parts = [p for p in [injury_type, note, date_injury] if p]
                tooltip = " — ".join(tooltip_parts) if tooltip_parts else "Injury info unavailable"
                injury_html = f"<span title='{tooltip}'>🚑</span>"

        results.append({
            "Player":player,"Team":team,"Injury":injury_html,
            "Season Avg":round(season_avg,2),
            "L3 Shots":", ".join(map(str,last3)),
            "L5 Shots":", ".join(map(str,last5)),
            "L10 Shots":", ".join(map(str,last10)),
            "Trend Score":round(trend,3),
            "Final Projection":round(line,2),
            "Prob ≥ Projection (%) L5":round(p*100,1),
            "Playable Odds":implied_odds,
            "Line Adj":round(line_factor,2),
            "Form Indicator":form_flag
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if st.button("🚀 Run Model"):
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")
    df = build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df)
    if "Injury" not in df.columns:
        df["Injury"] = ""
    df = df.sort_values("Final Projection", ascending=False).reset_index(drop=True)
    st.session_state.results_raw = df.copy()
    st.success("✅ Model built successfully!")

# ---------------------------------------------------------------
# Display Table + Save/Download
# ---------------------------------------------------------------
if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
    df = st.session_state.results_raw.copy()
    def trend_color(v):
        if pd.isna(v): return "–"
        v = max(min(v, 0.5), -0.5)
        n = v + 0.5
        if n < 0.5: r,g,b = 255,int(255*(n*2)),0
        else: r,g,b = int(255*(1-(n-0.5)*2)),255,0
        color=f"rgb({r},{g},{b})"
        t="▲" if v>0.05 else ("▼" if v<-0.05 else "–")
        txt="#000" if abs(v)<0.2 else "#fff"
        return f"<div style='background:{color};color:{txt};font-weight:600;border-radius:6px;padding:4px 8px;text-align:center;'>{t}</div>"
    df["Trend"] = df["Trend Score"].apply(trend_color)
    cols = [
        "Player","Team","Injury","Trend","Final Projection",
        "Prob ≥ Projection (%) L5","Playable Odds",
        "Season Avg","Line Adj","Form Indicator",
        "L3 Shots","L5 Shots","L10 Shots"
    ]
    vis = df[[c for c in cols if c in df.columns]]
    html_table = vis.to_html(index=False, escape=False)
    components.html(f"""
        <style>
        div.scrollable-table {{
            overflow-x: auto;
            overflow-y: auto;
            height: 600px;
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
        }}
        td:first-child, th:first-child {{
            position: sticky;
            left: 0;
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
        """, height=620, scrolling=True)
    st.markdown("---")
    st.subheader("💾 Save or Download Projections")
    selected_date = st.date_input("Select game date:", datetime.date.today())
    if st.button("💾 Save Projections for Selected Date"):
        df_to_save = df.copy()
        df_to_save["Date_Game"] = selected_date.strftime("%Y-%m-%d")
        df_to_save["Matchup"] = f"{team_a} vs {team_b}"
        save_dir = "projections"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{team_a}_vs_{team_b}_{selected_date.strftime('%Y-%m-%d')}.csv"
        save_path = os.path.join(save_dir, filename)
        df_to_save.to_csv(save_path, index=False)
        st.success(f"✅ Saved projections to **{save_path}**")
        csv = df_to_save.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Projections CSV", data=csv, file_name=filename, mime="text/csv")


