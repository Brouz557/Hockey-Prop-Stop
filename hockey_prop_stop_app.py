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
        Team-vs-Team matchup analytics with goalie and line impact
    </p>
    """,
    unsafe_allow_html=True,
)

# ✅ Add custom CSS for wrapped table text
st.markdown(
    """
    <style>
    .dataframe td {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: center !important;
    }
    .stDataFrame {
        overflow-x: auto;
    }
    </style>
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
goalies_df = load_file(goalies_file)
lines_df = load_file(lines_file)

# ---------------------------------------------------------------
# Proceed only if SKATERS + SHOT DATA uploaded
# ---------------------------------------------------------------
if not skaters_df.empty and not shots_df.empty:
    st.success("✅ SKATERS and SHOT DATA loaded successfully.")

    # Normalize headers
    skaters_df.columns = skaters_df.columns.str.lower().str.strip()
    shots_df.columns = shots_df.columns.str.lower().str.strip()
    goalies_df.columns = goalies_df.columns.str.lower().str.strip() if not goalies_df.empty else []
    lines_df.columns = lines_df.columns.str.lower().str.strip() if not lines_df.empty else []

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

        # -------------------------------------------------------
        # Goalie suppression factors
        # -------------------------------------------------------
        goalie_adj = {}
        if not goalies_df.empty:
            try:
                goalies_df = goalies_df[goalies_df["situation"].str.lower() == "all"]
                goalies_df["sog_allowed_per_game"] = goalies_df["unblocked attempts"] / goalies_df["games"]
                team_avg = goalies_df.groupby("team")["sog_allowed_per_game"].mean()
                league_avg = team_avg.mean()
                goalie_adj = (league_avg / team_avg).to_dict()
            except Exception as e:
                st.warning(f"⚠️ Could not calculate goalie suppression: {e}")

        # -------------------------------------------------------
        # Line matchup impact
        # -------------------------------------------------------
        line_adj = {}
        if not lines_df.empty:
            try:
                lines_df["sog_against_per_game"] = lines_df["sog against"] / lines_df["games"]
                team_avg = lines_df.groupby("team")["sog_against_per_game"].mean()
                league_avg = team_avg.mean()
                lines_df["line_factor"] = league_avg / lines_df["sog_against_per_game"]
                line_adj = lines_df.set_index("line pairings")["line_factor"].to_dict()
            except Exception as e:
                st.warning(f"⚠️ Could not calculate line matchup adjustments: {e}")

        # -------------------------------------------------------
        # Roster for selected teams
        # -------------------------------------------------------
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

            # Aggregate to per-game level (sum SOG by game)
            game_sogs = (
                df_p.groupby("gameid")["sog"]
                .sum()
                .reset_index()
                .sort_values("gameid")
            )
            sog_values = game_sogs["sog"].tolist()

            last3 = sog_values[-3:]
            last5 = sog_values[-5:]
            last10 = sog_values[-10:]

            l3 = np.mean(last3) if last3 else np.nan
            l5 = np.mean(last5) if last5 else np.nan
            l10 = np.mean(last10) if last10 else np.nan
            season_avg = np.mean(sog_values)

            trend = 0 if pd.isna(l10) or l10 == 0 else (l3 - l10) / l10

            # ✅ FIXED weighting (sum, not mean)
            base_proj = np.nansum([0.5 * l3, 0.3 * l5, 0.2 * l10])

            # ---------------------------------------------------
            # Apply Goalie Adjustment
            # ---------------------------------------------------
            opp_team = team_b if team == team_a else team_a
            goalie_factor = goalie_adj.get(opp_team, 1.0)

            # ---------------------------------------------------
            # Apply Line Adjustment
            # ---------------------------------------------------
            line_factor = 1.0
            if not lines_df.empty:
                try:
                    last_name = str(player).split()[-1]()_
