import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with goalie & weighted line adjustments
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Custom CSS for better table layout and color heatmap
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .dataframe td {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: center !important;
        line-height: 1.3em !important;
        font-size: 15px !important;
    }
    .stDataFrame {overflow-x: auto;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# File uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx", "csv"])
shots_file = st.sidebar.file_uploader("Shot Data", type=["xlsx", "csv"])
goalies_file = st.sidebar.file_uploader("Goaltenders", type=["xlsx", "csv"])
lines_file = st.sidebar.file_uploader("Line Data", type=["xlsx", "csv"])

# ---------------------------------------------------------------
# Helper: load file safely
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
# Load data
# ---------------------------------------------------------------
skaters_df = load_file(skaters_file)
shots_df = load_file(shots_file)
goalies_df = load_file(goalies_file)
lines_df = load_file(lines_file)

if not skaters_df.empty and not shots_df.empty:
    st.success("✅ Skaters and Shot Data loaded successfully.")

    # normalize columns
    skaters_df.columns = skaters_df.columns.str.lower().str.strip()
    shots_df.columns = shots_df.columns.str.lower().str.strip()
    goalies_df.columns = goalies_df.columns.str.lower().str.strip() if not goalies_df.empty else []
    lines_df.columns = lines_df.columns.str.lower().str.strip() if not lines_df.empty else []

    team_col = next((c for c in skaters_df.columns if "team" in c), None)
    player_col = "name" if "name" in skaters_df.columns else None
    sog_col = next((c for c in shots_df.columns if "sog" in c), None)
    game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)
    player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)

    if not all([team_col, player_col, sog_col, game_col, player_col_shots]):
        st.error("⚠️ Missing required columns in uploaded files.")
        st.stop()

    # -----------------------------------------------------------
    # Select teams
    # -----------------------------------------------------------
    all_teams = sorted(skaters_df[team_col].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A", all_teams)
    with col2:
        team_b = st.selectbox("Select Team B", [t for t in all_teams if t != team_a])

    st.markdown("---")
    run_model = st.button("🚀 Run Model")

    if run_model:
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

        # -------------------------------------------------------
        # 🥅 Goalie adjustments
        # -------------------------------------------------------
        goalie_adj, rebound_rate = {}, {}
        if not goalies_df.empty:
            df_g = goalies_df.copy()
            df_g = df_g[df_g["situation"].str.lower() == "all"]
            df_g["games"] = pd.to_numeric(df_g["games"], errors="coerce").fillna(0)
            df_g["unblocked attempts"] = pd.to_numeric(df_g["unblocked attempts"], errors="coerce").fillna(0)
            df_g["rebounds"] = pd.to_numeric(df_g["rebounds"], errors="coerce").fillna(0)

            df_g["shots_allowed_per_game"] = np.where(df_g["games"] > 0,
                                                      df_g["unblocked attempts"] / df_g["games"], np.nan)
            df_g["rebound_rate"] = np.where(df_g["unblocked attempts"] > 0,
                                            df_g["rebounds"] / df_g["unblocked attempts"], 0)

            team_avg = df_g.groupby("team")["shots_allowed_per_game"].mean()
            league_avg = team_avg.mean()
            goalie_adj = (league_avg / team_avg).to_dict()
            rebound_rate = df_g.groupby("team")["rebound_rate"].mean().to_dict()

        # -------------------------------------------------------
        # 🧱 Line adjustments
        # -------------------------------------------------------
        line_adj = {}
        if not lines_df.empty:
            df_l = lines_df.copy()
            df_l["games"] = pd.to_numeric(df_l["games"], errors="coerce").fillna(0)
            df_l["sog against"] = pd.to_numeric(df_l["sog against"], errors="coerce").fillna(0)
            df_l = (df_l.groupby(["line pairings", "team"], as_index=False)
                    .agg({"games": "sum", "sog against": "sum"}))
            df_l["sog_against_per_game"] = np.where(df_l["games"] > 0,
                                                    df_l["sog against"] / df_l["games"], np.nan)
            team_avg = df_l.groupby("team")["sog_against_per_game"].mean()
            league_avg = team_avg.mean()
            df_l["line_factor"] = league_avg / df_l["sog_against_per_game"]
            df_l["line_factor"] = (df_l["line_factor"]
                                   .replace([np.inf, -np.inf], np.nan)
                                   .fillna(1.0)
                                   .clip(0.7, 1.3))
            line_adj = df_l.copy()

        # -------------------------------------------------------
        # Team roster
        # -------------------------------------------------------
        roster = (skaters_df[skaters_df[team_col].isin([team_a, team_b])][[player_col, team_col]]
                  .rename(columns={player_col: "player", team_col: "team"})
                  .drop_duplicates(subset=["player"]).reset_index(drop=True))
        shots_df = shots_df.rename(columns={player_col_shots: "player", game_col: "gameid", sog_col: "sog"})
        grouped_shots = {n.lower(): g.sort_values("gameid")
                         for n, g in shots_df.groupby(shots_df["player"].str.lower())}

        # -------------------------------------------------------
        # Process players
        # -------------------------------------------------------
        results, progress = [], st.progress(0)
        total = len(roster)

        for i, row in enumerate(roster.itertuples(index=False), start=1):
            player, team = row.player, row.team
            df_p = grouped_shots.get(str(player).lower(), pd.DataFrame())
            if df_p.empty:
                progress.progress(i / total)
                continue

            game_sogs = df_p.groupby("gameid")["sog"].sum().reset_index().sort_values("gameid")
            sog_values = game_sogs["sog"].tolist()
            last3, last5, last10 = sog_values[-3:], sog_values[-5:], sog_values[-10:]
            l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)
            season_avg = np.mean(sog_values)
            trend = 0 if pd.isna(l10) or l10 == 0 else (l3 - l10) / l10
            base_proj = np.nansum([0.5 * l3, 0.3 * l5, 0.2 * l10])

            opp_team = team_b if team == team_a else team_a
            goalie_factor = np.clip(goalie_adj.get(opp_team, 1.0), 0.7, 1.3)
            rebound_factor = rebound_rate.get(opp_team, 0.0)
            line_factor = 1.0
            if not isinstance(line_adj, dict) and not line_adj.empty:
                try:
                    last_name = str(player).split()[-1].lower()
                    match = line_adj[line_adj["line pairings"].str.contains(last_name, case=False, na=False)]
                    if not match.empty:
                        line_factor = np.average(match["line_factor"], weights=match["games"])
                    line_factor = np.clip(line_factor, 0.7, 1.3)
                except Exception:
                    pass

            adj_proj = base_proj * (0.7 + 0.3 * goalie_factor) * (0.7 + 0.3 * line_factor)
            adj_proj *= (1 + rebound_factor * 0.1)
            adj_proj = max(0, round(adj_proj, 2))

            results.append({
                "Player": player, "Team": team,
                "Season Avg": round(season_avg, 2),
                "L3 Shots": ", ".join(map(str, last3)),
                "L5 Shots": ", ".join(map(str, last5)),
                "L10 Shots": ", ".join(map(str, last10)),
                "Trend": round(trend, 3),
                "Base Projection": round(base_proj, 2),
                "Goalie Adj": round(goalie_factor, 2),
                "Line Adj": round(line_factor, 2),
                "Final Projection": adj_proj,
            })
            progress.progress(i / total)
        progress.empty()

        # -------------------------------------------------------
        # ✅ Enhanced Matchup Rating
        # -------------------------------------------------------
        result_df = pd.DataFrame(results)
        result_df["Effective Projection"] = result_df.apply(
            lambda x: x["Final Projection"] / ((x["Goalie Adj"] + x["Line Adj"]) / 2), axis=1
        )

        avg_proj = result_df["Effective Projection"].mean()
        std_proj = result_df["Effective Projection"].std()

        def rate(val):
            if val >= avg_proj + std_proj:
                return "Strong"
            elif val >= avg_proj:
                return "Moderate"
            else:
                return "Weak"

        result_df["Matchup Rating"] = result_df["Effective Projection"].apply(rate)

        # -------------------------------------------------------
        # 🏒 NHL Logos (PNG CDN)
        # -------------------------------------------------------
        team_logos = {
            "Toronto Maple Leafs": "TOR", "Vancouver Canucks": "VAN", "Edmonton Oilers": "EDM",
            "Calgary Flames": "CGY", "Montreal Canadiens": "MTL", "Ottawa Senators": "OTT",
            "Boston Bruins": "BOS", "New York Rangers": "NYR", "New York Islanders": "NYI",
            "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "Chicago Blackhawks": "CHI",
            "Colorado Avalanche": "COL", "Dallas Stars": "DAL", "Vegas Golden Knights": "VGK",
            "Los Angeles Kings": "LAK", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
            "Detroit Red Wings": "DET", "Tampa Bay Lightning": "TBL", "Florida Panthers": "FLA",
            "Nashville Predators": "NSH", "Washington Capitals": "WSH", "Buffalo Sabres": "BUF",
            "St. Louis Blues": "STL", "Winnipeg Jets": "WPG", "Minnesota Wild": "MIN",
            "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI", "Columbus Blue Jackets": "CBJ",
            "New Jersey Devils": "NJD"
        }

        def get_logo_html(team_name):
            abbr = team_logos.get(team_name)
            if not abbr:
                return team_name
            url = f"https://assets.nhle.com/logos/nhl/teams-current-primary-light/{abbr}.png"
            return f"<img src='{url}' width='28' style='vertical-align:middle;margin-right:6px;'> {team_name}"

        result_df["Team"] = result_df["Team"].apply(get_logo_html)

        # -------------------------------------------------------
        # Final display
        # -------------------------------------------------------
        display_cols = [
            "Player", "Team", "Trend", "Final Projection", "Season Avg",
            "Matchup Rating", "L3 Shots", "L5 Shots", "L10 Shots",
            "Base Projection", "Goalie Adj", "Line Adj"
        ]
        visible_df = result_df[display_cols]

        st.success(f"✅ Model built successfully for {team_a} vs {team_b}")
        st.markdown(f"### 📊 {team_a} vs {team_b} — Player Projections (Adjusted)")
        st.markdown(visible_df.to_html(index=False, escape=False), unsafe_allow_html=True)

else:
    st.info("📥 Upload at least Skaters and Shot Data to begin.")
