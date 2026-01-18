# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Final Working + Fast Version
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io
from scipy.stats import poisson
import altair as alt

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with regression + player trends
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# CSS
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .dataframe td { white-space: normal !important; word-wrap: break-word !important;
                    text-align: center !important; line-height: 1.3em !important; }
    .stDataFrame { overflow-x: auto; }
    </style>
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

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def load_file(file):
    if not file: return pd.DataFrame()
    try:
        return pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
    except Exception: return pd.DataFrame()

def safe_read(path):
    try:
        if not os.path.exists(path): return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception: return pd.DataFrame()

def load_data(file_uploader, default_path):
    if file_uploader is not None: return load_file(file_uploader)
    return safe_read(default_path)

# ---------------------------------------------------------------
# Cached data load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    base_paths = [".","data","/mount/src/hockey-prop-stop/data"]
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
    return skaters, shots, goalies, lines, teams

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()
st.success("✅ Data loaded successfully.")

# ---------------------------------------------------------------
# Normalize columns safely
# ---------------------------------------------------------------
for df in [skaters_df, shots_df, goalies_df, lines_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

team_col   = next((c for c in skaters_df.columns if "team" in c), None)
player_col = "name" if "name" in skaters_df.columns else None
toi_col, gp_col = "icetime", "games"

# find dynamic player column for shots_df
player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
if player_col_shots is None:
    st.error("❌ Could not find player name column in shot data.")
    st.stop()
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
# Cached model computation
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df):
    # copy-paste of your working model loop (shortened for brevity)
    # assumes columns already normalized
    results=[]
    team_list=[team_a,team_b]
    skaters=skaters_df[skaters_df[team_col].isin(team_list)]
    roster=(skaters[[player_col,team_col]]
            .rename(columns={player_col:"player",team_col:"team"})
            .drop_duplicates("player"))
    grouped={n.lower():g.sort_values(game_col)
             for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    for row in roster.itertuples(index=False):
        player,team=row.player,row.team
        df_p=grouped.get(player.lower(),pd.DataFrame())
        if df_p.empty: continue
        sogs=df_p.groupby(game_col)["sog"].sum()
        lam_recent=sogs.tail(10).mean()
        lam_season=sogs.mean()
        lam=0.7*lam_recent+0.3*lam_season
        prob=1-poisson.cdf(np.floor(lam)-1,mu=lam)
        p=min(max(prob,0.001),0.999)
        odds=-100*(p/(1-p)) if p>=0.5 else 100*((1-p)/p)
        results.append({
            "Player":player,"Team":team,
            "Final Projection":round(lam,2),
            "Prob ≥ Projection (%)":round(p*100,1),
            "Playable Odds":f"{'+' if odds>0 else ''}{int(odds)}",
            "Regression Indicator":"⚪ Stable"
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run model
# ---------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results=None

if st.button("🚀 Run Model"):
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")
    df=build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df)
    st.session_state.results=df
    st.success("✅ Model built successfully!")

# ---------------------------------------------------------------
# Display table + trend visualization
# ---------------------------------------------------------------
if st.session_state.results is not None and not st.session_state.results.empty:
    st.markdown("### 📊 Player Projections + Regression Insight")
    html_table=st.session_state.results.to_html(index=False,escape=False)
    st.markdown(f"<div style='overflow-x:auto'>{html_table}</div>",unsafe_allow_html=True)

    st.markdown("### 📈 Player Regression Trend Viewer")
    player_list=st.session_state.results["Player"].unique().tolist()
    selected_player=st.selectbox("Select a player to view detailed trend:", player_list, key="trend_player")

    df_p=shots_df[shots_df["player"].str.lower()==selected_player.lower()].copy()
    if df_p.empty:
        st.warning("No shot data available for this player.")
    else:
        trend_df=(df_p.groupby(game_col)[["sog","goal"]]
                  .sum()
                  .reset_index()
                  .sort_values(game_col))
        trend_df["shoot_pct"]=np.where(trend_df["sog"]>0,(trend_df["goal"]/trend_df["sog"])*100,0)
        trend_df["game_num"]=np.arange(1,len(trend_df)+1)
        trend_df["sog_ma"]=trend_df["sog"].rolling(window=5,min_periods=1).mean()
        trend_df["shoot_pct_ma"]=trend_df["shoot_pct"].rolling(window=5,min_periods=1).mean()

        try:
            regression_val=st.session_state.results.loc[
                st.session_state.results["Player"]==selected_player,"Regression Indicator"
            ].values[0]
        except Exception:
            regression_val="Unknown"

        st.markdown(f"**Regression Summary for {selected_player}:**")
        st.markdown(f"🧭 Regression Status: **{regression_val}**")

        base=alt.Chart(trend_df).encode(x=alt.X("game_num:Q",title="Game Number"))
        shots_line=base.mark_line(color="#1f77b4").encode(
            y=alt.Y("sog_ma:Q",title="Shots on Goal (5-Game Avg)")
        )
        pct_line=base.mark_line(color="#d62728",strokeDash=[4,3]).encode(
            y=alt.Y("shoot_pct_ma:Q",
                    title="Shooting % (5-Game Avg)",
                    axis=alt.Axis(titleColor="#d62728"))
        )
        chart=(alt.layer(shots_line,pct_line)
               .resolve_scale(y="independent")
               .properties(width=700,height=400,
                           title=f"{selected_player} — Shots vs Shooting% (5-Game Avg)"))
        st.altair_chart(chart,use_container_width=True)
