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
shots_file = st.sidebar.file_uploader("SHOT DATA", type=["xlsx", "csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx", "csv"])
lines_file = st.sidebar.file_uploader("LINE DATA", type=["xlsx", "csv"])
teams_file = st.sidebar.file_uploader("TEAMS", type=["xlsx", "csv"])

# ---------------------------------------------------------------
# Helper: Load CSV or Excel safely
# ---------------------------------------------------------------
def load_file(file):
    if not file:
        return pd.DataFrame()
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error reading {file.name}: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------------
# Load required data
# ---------------------------------------------------------------
skaters_df = load_file(skaters_file)
shots_df = load_file(shots_file)

# ---------------------------------------------------------------
# Proceed only if SKATERS + SHOT DATA uploaded
# ---------------------------------------------------------------
if not skaters_df.empty and not shots_df.empty:
    st.success("✅ SKATERS and SHOT DATA loaded successfully.")

    # Normalize headers
    skaters_df.columns = skaters_df.columns.str.lower().str.strip()
    shots_df.columns = shots_df.columns.str.lower().str.strip()

    # Identify key columns
    team_col = next((c for c in skaters_df.columns if "team" in c), None)
    player_col = "name" if "name" in skaters_df.columns else None
    sog_col = next((c for c in shots_df.columns if "sog" in c), None)
    game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)
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

    st.markdown("---")
    run_model = st.button("🚀 Run Model")

    # -----------------------------------------------------------
    # Run only when button clicked
    # -----------------------------------------------------------
    if run_model:
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

        # Roster for selected teams — ensure unique player names
        roster = (
            skaters_df[skaters_df[team_col].isin([team_a, team_b])][[player_col, team_col]]
            .rename(columns={player_col: "player", team_col: "team"})
            .drop_duplicates(subset=["player"])
            .reset_index(drop=True)
        )

        # Prepare shot data
        shots_df = shots_df.rename(
            columns={player_col_shots: "player", game_col: "gameid", sog_col: "sog"}
        )
        shots_df["player"] = shots_df["player"].astype(str).str.strip()
        roster["player"] = roster["player"].astype(str).str.strip()

        # Group shots once for performance
        grouped_shots = {
            name.lower(): g.sort_values("gameid")
            for name, g in shots_df.groupby(shots_df["player"].str.lower())
        }

        results = []
        progress = st.progress(0)
        total = len(roster)

        for i, row in enumerate(roster.itertuples(index=False), start=1):
            player, team = row.player, row.team
            df_p = grouped_shots.get(str(player).lower(), pd.DataFrame())
            if df_p.empty:
                progress.progress(i / total)
                continue

            # ✅ Aggregate to per-game level (sum SOG by game)
            game_sogs = (
                df_p.groupby("gameid")["sog"]
                .sum()
                .reset_index()
                .sort_values("gameid")
            )

            sog_values = game_sogs["sog"].tolist()

            # recent games
            last3 = sog_values[-3:]
            last5 = sog_values[-5:]
            last10 = sog_values[-10:]
            last20 = sog_values[-20:]

            l3 = np.mean(last3) if last3 else np.nan
            l5 = np.mean(last5) if last5 else np.nan
            l10 = np.mean(last10) if last10 else np.nan
            l20 = np.mean(last20) if last20 else np.nan
            season_avg = np.mean(sog_values)

            trend = 0 if pd.isna(l10) or l10 == 0 else (l3 - l10) / l10
            base_proj = np.nanmean([0.4 * l3, 0.3 * l5, 0.2 * l10, 0.1 * l20])

            # Assign matchup strength rating
            if base_proj >= 3.5:
                signal = "Strong"
            elif base_proj >= 2.0:
                signal = "Moderate"
            else:
                signal = "Weak"

            results.append(
                {
                    "Player": player,
                    "Team": team,
                    "Season Avg": round(season_avg, 2),
                    "L3 Shots": ", ".join(map(str, last3)),
                    "L3 Avg": round(l3, 2),
                    "L5 Shots": ", ".join(map(str, last5)),
                    "L5 Avg": round(l5, 2),
                    "L10 Shots": ", ".join(map(str, last10)),
                    "L10 Avg": round(l10, 2),
                    "L20 Shots": ", ".join(map(str, last20)),
                    "L20 Avg": round(l20, 2),
                    "Trend Score": round(trend, 3),
                    "Base Projection": round(base_proj, 2),
                    "Matchup Rating": signal,
                }
            )

            progress.progress(i / total)

        progress.empty()

        if not results:
            st.warning("⚠️ No matching players found for these teams.")
            st.stop()

        # Display table
        st.success(f"✅ Model built successfully for {team_a} vs {team_b}!")
        result_df = pd.DataFrame(results).sort_values("Base Projection", ascending=False)
        st.markdown(f"### 📊 {team_a} vs {team_b} — Player Trend Table")
        st.dataframe(result_df, use_container_width=True)

else:
    st.info("📥 Upload at least SKATERS and SHOT DATA to begin.")
