# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Smart Data Loading + Persistent Cache
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os
import contextlib
import io

# ---------------------------------------------------------------
# Page Setup
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
# Custom CSS for readable tables
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .dataframe td {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: center !important;
        line-height: 1.3em !important;
    }
    .stDataFrame {
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# File Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("SKATERS", type=["xlsx", "csv"])
shots_file = st.sidebar.file_uploader("SHOT DATA", type=["xlsx", "csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx", "csv"])
lines_file = st.sidebar.file_uploader("LINE DATA", type=["xlsx", "csv"])
teams_file = st.sidebar.file_uploader("TEAMS", type=["xlsx", "csv"])

# ---------------------------------------------------------------
# File Loading Utilities
# ---------------------------------------------------------------
def load_file(file):
    """Read CSV or Excel safely."""
    if not file:
        return pd.DataFrame()
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error reading {file.name}: {e}")
        return pd.DataFrame()


def safe_read(path):
    """Silent reader for local repo files."""
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception:
        return pd.DataFrame()


def load_data(file_uploader, default_path):
    """Prefer uploaded file, fallback to repo default (silent)."""
    if file_uploader is not None:
        return load_file(file_uploader)
    return safe_read(default_path)


# ---------------------------------------------------------------
# Smart Cached Loader (now auto-finds files)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    """Load all five datasets once per session (cached, robust path lookup)."""
    base_paths = [
        ".",                     # repo root
        "data",                  # /data folder
        "/mount/src/hockey-prop-stop/data",
        "/mount/src/hockey-prop-stop"
    ]

    def find_file(filename):
        for p in base_paths:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return full
        return None  # not found

    with contextlib.redirect_stdout(io.StringIO()):  # silent
        skaters = load_data(skaters_file, find_file("SKATERS.xlsx") or "SKATERS.xlsx")
        shots = load_data(shots_file, find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
        goalies = load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
        lines = load_data(lines_file, find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
        teams = load_data(teams_file, find_file("TEAMS.xlsx") or "TEAMS.xlsx")

    return skaters, shots, goalies, lines, teams


# ---------------------------------------------------------------
# Load Everything Once (silent cache)
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file
)

# Confirm visible only if missing
missing = []
if skaters_df.empty:
    missing.append("SKATERS.xlsx")
if shots_df.empty:
    missing.append("SHOT DATA.xlsx")

if missing:
    st.warning(f"⚠️ Missing or unreadable file(s): {', '.join(missing)}")
else:
    st.success("✅ Data loaded successfully.")


# ---------------------------------------------------------------
# Proceed only if data available
# ---------------------------------------------------------------
if not skaters_df.empty and not shots_df.empty:

    # Normalize column names
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
    # Team Selection UI
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
    # Main Model Run
    # -----------------------------------------------------------
    if run_model:
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

        # -------------------------------------------------------
        # 🥅 Goalie Adjustments
        # -------------------------------------------------------
        goalie_adj = {}
        rebound_rate = {}
        if not goalies_df.empty:
            try:
                df_g = goalies_df.copy()
                df_g = df_g[df_g["situation"].str.lower() == "all"]
                df_g["games"] = pd.to_numeric(df_g["games"], errors="coerce").fillna(0)
                df_g["unblocked attempts"] = pd.to_numeric(df_g["unblocked attempts"], errors="coerce").fillna(0)
                df_g["rebounds"] = pd.to_numeric(df_g["rebounds"], errors="coerce").fillna(0)

                df_g["shots_allowed_per_game"] = np.where(
                    df_g["games"] > 0,
                    df_g["unblocked attempts"] / df_g["games"],
                    np.nan,
                )
                df_g["rebound_rate"] = np.where(
                    df_g["unblocked attempts"] > 0,
                    df_g["rebounds"] / df_g["unblocked attempts"],
                    0,
                )

                team_avg = df_g.groupby("team")["shots_allowed_per_game"].mean()
                league_avg = team_avg.mean()
                goalie_adj = (league_avg / team_avg).to_dict()
                rebound_rate = df_g.groupby("team")["rebound_rate"].mean().to_dict()
            except Exception as e:
                st.warning(f"⚠️ Goalie data incomplete or invalid: {e}")

        # -------------------------------------------------------
        # 🧱 Line Adjustments
        # -------------------------------------------------------
        line_adj = {}
        if not lines_df.empty:
            try:
                df_l = lines_df.copy()
                df_l["games"] = pd.to_numeric(df_l["games"], errors="coerce").fillna(0)
                df_l["sog against"] = pd.to_numeric(df_l["sog against"], errors="coerce").fillna(0)
                df_l = (
                    df_l.groupby(["line pairings", "team"], as_index=False)
                    .agg({"games": "sum", "sog against": "sum"})
                )
                df_l["sog_against_per_game"] = np.where(
                    df_l["games"] > 0,
                    df_l["sog against"] / df_l["games"],
                    np.nan,
                )
                team_avg = df_l.groupby("team")["sog_against_per_game"].mean()
                league_avg = team_avg.mean()
                df_l["line_factor"] = league_avg / df_l["sog_against_per_game"]
                df_l["line_factor"] = (
                    df_l["line_factor"]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(1.0)
                    .clip(0.7, 1.3)
                )
                line_adj = df_l.copy()
            except Exception as e:
                st.warning(f"⚠️ Line data incomplete or invalid: {e}")

        # -------------------------------------------------------
        # Build Team Roster
        # -------------------------------------------------------
        roster = (
            skaters_df[skaters_df[team_col].isin([team_a, team_b])][[player_col, team_col]]
            .rename(columns={player_col: "player", team_col: "team"})
            .drop_duplicates(subset=["player"])
            .reset_index(drop=True)
        )

        shots_df = shots_df.rename(
            columns={player_col_shots: "player", game_col: "gameid", sog_col: "sog"}
        )
        shots_df["player"] = shots_df["player"].astype(str).str.strip()
        roster["player"] = roster["player"].astype(str).str.strip()

        grouped_shots = {
            name.lower(): g.sort_values("gameid")
            for name, g in shots_df.groupby(shots_df["player"].str.lower())
        }

        # -------------------------------------------------------
        # Main Loop
        # -------------------------------------------------------
        results = []
        progress = st.progress(0)
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
            l3, l5, l10 = (
                np.mean(last3) if last3 else np.nan,
                np.mean(last5) if last5 else np.nan,
                np.mean(last10) if last10 else np.nan,
            )
            season_avg = np.mean(sog_values)
            trend = 0 if pd.isna(l10) or l10 == 0 else (l3 - l10) / l10
            base_proj = np.nansum([0.5 * l3, 0.3 * l5, 0.2 * l10])

            opp_team = team_b if team == team_a else team_a
            goalie_factor = np.clip(goalie_adj.get(opp_team, 1.0), 0.7, 1.3)
            rebound_factor = rebound_rate.get(opp_team, 0.0)
            line_factor = 1.0

            if not line_adj.empty:
                try:
                    last_name = str(player).split()[-1].lower()
                    matching = line_adj[
                        line_adj["line pairings"].str.contains(last_name, case=False, na=False)
                    ]
                    if not matching.empty:
                        line_factor = np.average(
                            matching["line_factor"], weights=matching["games"]
                        )
                    line_factor = np.clip(line_factor, 0.7, 1.3)
                except Exception:
                    pass

            adj_proj = base_proj * (0.7 + 0.3 * goalie_factor) * (0.7 + 0.3 * line_factor)
            adj_proj *= (1 + rebound_factor * 0.1)
            adj_proj = max(0, round(adj_proj, 2))

            l10_shots_formatted = (
                "<br>".join(
                    [", ".join(map(str, last10[:5])), ", ".join(map(str, last10[5:]))]
                )
                if len(last10) > 5
                else ", ".join(map(str, last10))
            )

            results.append(
                {
                    "Player": player,
                    "Team": team,
                    "Season Avg": round(season_avg, 2),
                    "L3 Shots": ", ".join(map(str, last3)),
                    "L3 Avg": round(l3, 2),
                    "L5 Shots": ", ".join(map(str, last5)),
                    "L5 Avg": round(l5, 2),
                    "L10 Shots": l10_shots_formatted,
                    "L10 Avg": round(l10, 2),
                    "Trend Score": round(trend, 3),
                    "Base Projection": round(base_proj, 2),
                    "Goalie Adj": round(goalie_factor, 2),
                    "Line Adj": round(line_factor, 2),
                    "Adj Projection": adj_proj,
                }
            )

            progress.progress(i / total)

        progress.empty()

        # Final Output
        if not results:
            st.warning("⚠️ No matching players found for these teams.")
            st.stop()

        result_df = pd.DataFrame(results)
        avg_proj, std_proj = result_df["Adj Projection"].mean(), result_df["Adj Projection"].std()

        def rate(val):
            if val >= avg_proj + std_proj:
                return "Strong"
            elif val >= avg_proj:
                return "Moderate"
            else:
                return "Weak"

        result_df["Matchup Rating"] = result_df["Adj Projection"].apply(rate)
        result_df = result_df.sort_values("Adj Projection", ascending=False)

        st.success(f"✅ Model built successfully for {team_a} vs {team_b}!")
        st.markdown(f"### 📊 {team_a} vs {team_b} — Player Projections (Adjusted)")
        st.markdown(result_df.to_html(index=False, escape=False), unsafe_allow_html=True)

else:
    st.info("📥 Upload files or use defaults from repo to begin.")
