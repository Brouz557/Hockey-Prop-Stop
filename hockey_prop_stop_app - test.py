# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — FULLY FIXED VERSION
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Puck Shotz Hockey Analytics (Test)",
    layout="wide",
    page_icon="🏒"
)

st.warning("⚠️ Test / Debug Version")

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

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar Uploads
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files")
skaters_file   = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file     = st.sidebar.file_uploader("Shot Data", type=["xlsx","csv"])
goalies_file   = st.sidebar.file_uploader("Goalies (optional)", type=["xlsx","csv"])
lines_file     = st.sidebar.file_uploader("Lines (optional)", type=["xlsx","csv"])
teams_file     = st.sidebar.file_uploader("Teams (optional)", type=["xlsx","csv"])
injuries_file  = st.sidebar.file_uploader("Injuries (optional)", type=["xlsx","csv"])

run_model = st.sidebar.button("🚀 Run Model")

st.session_state.line_test_val = st.sidebar.number_input(
    "Test Line",
    min_value=0.5,
    max_value=10.5,
    step=0.5,
    value=2.5
)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def load_file(f):
    if f is None:
        return pd.DataFrame()
    try:
        if f.name.lower().endswith(".xlsx"):
            return pd.read_excel(f)
        return pd.read_csv(f)
    except Exception:
        return pd.DataFrame()

def normalize(df):
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()
    return df

def find_col(df, keywords):
    return next(
        (c for c in df.columns if any(k in c for k in keywords)),
        None
    )

# ---------------------------------------------------------------
# Load + Normalize Data
# ---------------------------------------------------------------
skaters_df  = normalize(load_file(skaters_file))
shots_df    = normalize(load_file(shots_file))
goalies_df  = normalize(load_file(goalies_file))
lines_df    = normalize(load_file(lines_file))
teams_df    = normalize(load_file(teams_file))
injuries_df = normalize(load_file(injuries_file))

# ---------------------------------------------------------------
# Auto-Detect Skaters Columns (CRITICAL FIX)
# ---------------------------------------------------------------
sk_name_col = find_col(skaters_df, ["name","player","skater"])
sk_team_col = find_col(skaters_df, ["team"])
sk_pos_col  = find_col(skaters_df, ["position","pos"])

if not sk_name_col or not sk_team_col:
    st.error(
        f"❌ Skaters file missing required columns.\n\n"
        f"Found columns:\n{list(skaters_df.columns)}"
    )
    st.stop()

skaters_df = skaters_df.rename(columns={
    sk_name_col: "name",
    sk_team_col: "team"
})

if sk_pos_col:
    skaters_df = skaters_df.rename(columns={sk_pos_col: "position"})
else:
    skaters_df["position"] = "F"   # safe default

# ---------------------------------------------------------------
# Auto-Detect Shot Data Columns
# ---------------------------------------------------------------
shot_player_col = find_col(shots_df, ["player","shooter","name"])
shot_team_col   = find_col(shots_df, ["team"])
shot_game_col   = find_col(shots_df, ["game","match"])
shot_sog_col    = find_col(shots_df, ["sog","shots on goal"])

if not all([shot_player_col, shot_team_col, shot_game_col, shot_sog_col]):
    st.error(
        f"❌ Shot data missing required columns.\n\n"
        f"Found columns:\n{list(shots_df.columns)}"
    )
    st.stop()

shots_df = shots_df.rename(columns={
    shot_player_col: "player",
    shot_team_col: "team",
    shot_game_col: "game_id",
    shot_sog_col: "sog"
})

# ---------------------------------------------------------------
# Normalize Team Abbreviations
# ---------------------------------------------------------------
skaters_df["team"] = skaters_df["team"].map(TEAM_ABBREV_MAP).fillna(skaters_df["team"])
shots_df["team"]   = shots_df["team"].map(TEAM_ABBREV_MAP).fillna(shots_df["team"])

# ---------------------------------------------------------------
# Position Matchup Adjustment
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_position_matchups(shots, skaters):

    def norm(x):
        return str(x).lower().replace(".","").replace(",","").strip()

    s = shots.copy()
    k = skaters.copy()

    s["player_norm"] = s["player"].apply(norm)
    k["player_norm"] = k["name"].apply(norm)

    s = s.merge(k[["player_norm","position"]], on="player_norm", how="left")
    s = s.dropna(subset=["position","sog","game_id"])

    per_game = s.groupby(["team","position","game_id"])["sog"].sum().reset_index()
    avg = per_game.groupby(["team","position"])["sog"].mean().reset_index()

    league_avg = avg.groupby("position")["sog"].mean().to_dict()
    avg["pos_factor"] = avg.apply(
        lambda r: r["sog"] / league_avg.get(r["position"], r["sog"]),
        axis=1
    )

    avg["pos_factor"] = avg["pos_factor"].clip(0.85,1.20)

    return {(r["team"], r["position"]): r["pos_factor"] for _,r in avg.iterrows()}

pos_matchup_adj = build_position_matchups(shots_df, skaters_df)

# ---------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------
def build_model(team_a, team_b):

    roster = skaters_df[skaters_df["team"].isin([team_a, team_b])]
    roster = roster[["name","team","position"]].drop_duplicates()

    grouped = {
        p.lower(): g
        for p, g in shots_df.groupby(shots_df["player"].str.lower())
    }

    rows = []

    for r in roster.itertuples(index=False):
        player, team, position = r.name, r.team, r.position
        df_p = grouped.get(player.lower())

        if df_p is None:
            continue

        sog_vals = df_p.groupby("game_id")["sog"].sum().tolist()
        if len(sog_vals) < 3:
            continue

        l3  = np.mean(sog_vals[-3:])
        l5  = np.mean(sog_vals[-5:])
        l10 = np.mean(sog_vals[-10:])

        baseline = 0.55*l10 + 0.30*l5 + 0.15*l3
        opp = team_b if team == team_a else team_a
        pos_factor = pos_matchup_adj.get((opp, position), 1.0)

        lam = baseline * pos_factor

        rows.append({
            "Player": player,
            "Team": team,
            "Position": position,
            "Final Projection": round(lam,2),
            "Pos Adj": round(pos_factor,2),
            "Season Avg": round(np.mean(sog_vals),2)
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------
# TEMP MATCHUPS (replace with ESPN later)
# ---------------------------------------------------------------
games = [
    {"away":"BOS","home":"NYR"},
    {"away":"TOR","home":"MTL"}
]

# ---------------------------------------------------------------
# Run Model
# ---------------------------------------------------------------
if run_model:
    tables = []

    for g in games:
        df = build_model(g["away"], g["home"])
        if not df.empty:
            df["Matchup"] = f"{g['away']}@{g['home']}"
            tables.append(df)

    if tables:
        st.session_state.results = pd.concat(tables, ignore_index=True)
        st.success("✅ Model Built Successfully")
    else:
        st.warning("⚠️ No valid results generated")

# ---------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state.results.copy()
    line = st.session_state.line_test_val

    df["Prob ≥ Line (%)"] = df["Final Projection"].apply(
        lambda lam: round((1 - poisson.cdf(line-1, mu=max(lam,0.01))) * 100, 1)
    )

    df = df.sort_values(["Team","Final Projection"], ascending=[True,False])

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "💾 Download Results (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "puck_shotz_results.csv",
        "text/csv"
    )
