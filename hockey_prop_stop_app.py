import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team Trend-Weighted Shot Projections
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# File uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")

skaters_file = st.sidebar.file_uploader("SKATERS", type=["xlsx", "csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx", "csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx", "csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx", "csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx", "csv"])

def load_file(file):
    """Read Excel or CSV safely."""
    if not file:
        return pd.DataFrame()
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error reading {file.name}: {e}")
        return pd.DataFrame()

skaters_df = load_file(skaters_file)
shots_df   = load_file(shots_file)

# ---------------------------------------------------------------
# Proceed only if SKATERS + SHOT DATA uploaded
# ---------------------------------------------------------------
if not skaters_df.empty and not shots_df.empty:
    st.success("✅ SKATERS and SHOT DATA loaded")

    # Normalize headers
    skaters_df.columns = skaters_df.columns.str.lower().str.strip()
    shots_df.columns   = shots_df.columns.str.lower().str.strip()

    # Identify essential columns
    team_col   = next((c for c in skaters_df.columns if "team" in c), None)
    player_col = next((c for c in skaters_df.columns if "player" in c or "name" in c), None)

    sog_col    = next((c for c in shots_df.columns if "sog" in c), None)
    game_col   = next((c for c in shots_df.columns if "game" in c and "id" in c), None)
    player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)

    if not all([team_col, player_col, sog_col, game_col, player_col_shots]):
        st.error("⚠️ Missing required columns in uploaded files.")
        st.stop()

    # -----------------------------------------------------------
    # Team selection
    # -----------------------------------------------------------
    all_teams = sorted(skaters_df[team_col].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A", all_teams)
    with col2:
        team_b = st.selectbox("Select Team B", [t for t in all_teams if t != team_a])

    # -----------------------------------------------------------
    # Roster for selected teams
    # -----------------------------------------------------------
    roster = (
        skaters_df[skaters_df[team_col].isin([team_a, team_b])][[player_col, team_col]]
        .rename(columns={player_col: "player", team_col: "team"})
    )

    # -----------------------------------------------------------
    # Calculate SOG trends
    # -----------------------------------------------------------
    shots_df = shots_df.rename(
        columns={player_col_shots: "player", game_col: "gameid", sog_col: "sog"}
    )
    shots_df["player"] = shots_df["player"].astype(str).str.strip()
    roster["player"]   = roster["player"].astype(str).str.strip()

    results = []
    for _, row in roster.iterrows():
        player = row["player"]
        team   = row["team"]
        df_p = shots_df.loc[shots_df["player"] == player].copy()
        if df_p.empty:
            continue
        df_p = df_p.sort_values("gameid")

        l3  = df_p["sog"].tail(3).mean()
        l5  = df_p["sog"].tail(5).mean()
        l10 = df_p["sog"].tail(10).mean()
        l20 = df_p["sog"].tail(20).mean()
        season_avg = df_p["sog"].mean()

        trend = 0 if pd.isna(l10) or l10 == 0 else (l3 - l10) / l10
        base_proj = np.nanmean([0.4*l3, 0.3*l5, 0.2*l10, 0.1*l20])

        results.append({
            "Player": player,
            "Team": team,
            "Season Avg": round(season_avg, 2),
            "L3": round(l3, 2),
            "L5": round(l5, 2),
            "L10": round(l10, 2),
            "L20": round(l20, 2),
            "Trend Score": round(trend, 3),
            "Base Projection": round(base_proj, 2)
        })

    result_df = pd.DataFrame(results).sort_values("Base Projection", ascending=False)

    # -----------------------------------------------------------
    # Display results
    # -----------------------------------------------------------
    st.markdown(f"### 📊 {team_a} vs {team_b} — Player Trend Table")
    st.dataframe(result_df, use_container_width=True)

else:
    st.info("📥 Upload at least SKATERS and SHOT DATA to begin.")
