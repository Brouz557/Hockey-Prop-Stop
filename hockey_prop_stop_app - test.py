# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — FINAL STABLE VERSION
# Repo-first • Case-insensitive filenames • Schema-safe
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.stats import poisson

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
# Sidebar
# ---------------------------------------------------------------
st.sidebar.header("📂 Optional Uploads (Repo is Source of Truth)")
skaters_file  = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file    = st.sidebar.file_uploader("Shot Data", type=["xlsx","csv"])

run_model = st.sidebar.button("🚀 Run Model")

st.session_state.line_test_val = st.sidebar.number_input(
    "Test Line",
    min_value=0.5,
    max_value=10.5,
    step=0.5,
    value=2.5
)

# ---------------------------------------------------------------
# Repo File Discovery (CASE-INSENSITIVE, RECURSIVE)
# ---------------------------------------------------------------
def find_repo_file(target_name: str):
    target_name = target_name.lower()
    search_roots = [
        ".",
        "./data",
        "/mount/src/hockey-prop-stop",
        "/mount/src/hockey-prop-stop/data"
    ]

    for base in search_roots:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower() == target_name:
                    return os.path.join(root, f)
    return None

def load_data(upload, repo_path):
    if upload is not None:
        try:
            return pd.read_excel(upload) if upload.name.lower().endswith(".xlsx") else pd.read_csv(upload)
        except Exception:
            return pd.DataFrame()

    if repo_path and os.path.exists(repo_path):
        try:
            return pd.read_excel(repo_path) if repo_path.lower().endswith(".xlsx") else pd.read_csv(repo_path)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()

def normalize(df):
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()
    return df

def find_col(df, keys):
    return next((c for c in df.columns if any(k in c for k in keys)), None)

# ---------------------------------------------------------------
# Load Data (Repo First, Upload Override)
# ---------------------------------------------------------------
skaters_df = normalize(load_data(
    skaters_file,
    find_repo_file("skaters.xlsx")
))

shots_df = normalize(load_data(
    shots_file,
    find_repo_file("shot data.xlsx")
))

# ---------------------------------------------------------------
# Guard
# ---------------------------------------------------------------
if skaters_df.empty or shots_df.empty:
    st.info("📂 Waiting for Skaters.xlsx and SHOT DATA.xlsx to load from repo…")
    st.stop()

# ---------------------------------------------------------------
# Auto-Detect Skaters Schema
# ---------------------------------------------------------------
sk_name = find_col(skaters_df, ["name","player","skater"])
sk_team = find_col(skaters_df, ["team"])
sk_pos  = find_col(skaters_df, ["position","pos"])

if not sk_name or not sk_team:
    st.error(f"❌ Skaters schema invalid. Found columns: {list(skaters_df.columns)}")
    st.stop()

skaters_df = skaters_df.rename(columns={sk_name:"name", sk_team:"team"})
skaters_df = skaters_df.loc[:, ~skaters_df.columns.duplicated()]

if sk_pos:
    skaters_df = skaters_df.rename(columns={sk_pos:"position"})
else:
    skaters_df["position"] = "F"

# ---------------------------------------------------------------
# Auto-Detect Shot Schema
# ---------------------------------------------------------------
sh_player = find_col(shots_df, ["player","shooter","name"])
sh_team   = find_col(shots_df, ["team"])
sh_game   = find_col(shots_df, ["game"])
sh_sog    = find_col(shots_df, ["sog","shots on goal"])

if not all([sh_player, sh_team, sh_game, sh_sog]):
    st.error(f"❌ Shot data schema invalid. Found columns: {list(shots_df.columns)}")
    st.stop()

shots_df = shots_df.rename(columns={
    sh_player:"player",
    sh_team:"team",
    sh_game:"game_id",
    sh_sog:"sog"
})

shots_df = shots_df.loc[:, ~shots_df.columns.duplicated()]

# ---------------------------------------------------------------
# Normalize Team Abbreviations
# ---------------------------------------------------------------
skaters_df["team"] = (
    skaters_df["team"].astype(str)
    .map(TEAM_ABBREV_MAP)
    .fillna(skaters_df["team"])
)

shots_df["team"] = (
    shots_df["team"].astype(str)
    .map(TEAM_ABBREV_MAP)
    .fillna(shots_df["team"])
)

# ---------------------------------------------------------------
# Position Matchup Adjustment (FULLY SAFE)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_position_matchups(shots, skaters):

    s = shots.copy()
    k = skaters.copy()

    def norm(x):
        return str(x).lower().replace(".","").replace(",","").strip()

    s["p"] = s["player"].astype(str).apply(norm)
    k["p"] = k["name"].astype(str).apply(norm)

    if "position" not in k.columns:
        k["position"] = "F"

    s = s.merge(k[["p","position"]], on="p", how="left")

    required = {"team","position","sog","game_id"}
    if not required.issubset(s.columns):
        return {}

    s = s.dropna(subset=list(required))

    per_game = (
        s.groupby(["team","position","game_id"])["sog"]
        .sum()
        .reset_index()
    )

    avg = (
        per_game.groupby(["team","position"])["sog"]
        .mean()
        .reset_index()
    )

    league_avg = avg.groupby("position")["sog"].mean().to_dict()

    avg["pos_factor"] = avg.apply(
        lambda r: r["sog"] / league_avg.get(r["position"], r["sog"]),
        axis=1
    )

    avg["pos_factor"] = avg["pos_factor"].clip(0.85,1.20)

    return {
        (r["team"], r["position"]): r["pos_factor"]
        for _, r in avg.iterrows()
    }

pos_matchup_adj = build_position_matchups(shots_df, skaters_df)

# ---------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------
def build_model(team_a, team_b):

    roster = skaters_df[skaters_df["team"].isin([team_a,team_b])]
    roster = roster[["name","team","position"]].drop_duplicates()

    grouped = {
        n.lower(): g
        for n,g in shots_df.groupby(shots_df["player"].str.lower())
    }

    rows = []

    for r in roster.itertuples(index=False):
        player, team, pos = r.name, r.team, r.position
        df_p = grouped.get(player.lower())
        if df_p is None:
            continue

        sog_vals = df_p.groupby("game_id")["sog"].sum().tolist()
        if len(sog_vals) < 3:
            continue

        l3,l5,l10 = np.mean(sog_vals[-3:]), np.mean(sog_vals[-5:]), np.mean(sog_vals[-10:])
        baseline = 0.55*l10 + 0.30*l5 + 0.15*l3

        opp = team_b if team == team_a else team_a
        pos_factor = pos_matchup_adj.get((opp,pos),1.0)

        lam = baseline * pos_factor

        rows.append({
            "Player":player,
            "Team":team,
            "Position":pos,
            "Final Projection":round(lam,2),
            "Pos Adj":round(pos_factor,2),
            "Season Avg":round(np.mean(sog_vals),2)
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
        st.warning("⚠️ No results generated")

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
        "💾 Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "puck_shotz_results.csv",
        "text/csv"
    )
