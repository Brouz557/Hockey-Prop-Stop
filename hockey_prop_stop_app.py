import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from difflib import get_close_matches
import contextlib
import io

# ---------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="centered", page_icon="🏒")

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
# STYLE (responsive table)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .dataframe td {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: center !important;
        vertical-align: top !important;
        font-size: 0.9em !important;
    }
    @media only screen and (max-width: 600px) {
        table.dataframe {
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
        th, td {
            font-size: 0.75em !important;
            padding: 4px 6px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# FILE LOADING HELPERS (fully silent on success)
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
    """Silent reader for repo files (CSV or Excel)."""
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception as e:
        st.error(f"❌ Failed to read {path}: {e}")
        return pd.DataFrame()


def load_data(file_uploader, default_path):
    """Prefer uploaded file, fallback to repo file. Fully silent unless error."""
    try:
        if file_uploader is not None:
            return load_file(file_uploader)
        return safe_read(default_path)
    except Exception as e:
        st.error(f"❌ Error loading {default_path}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------
# SIDEBAR UPLOADS
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload New Data Files (optional)")
skaters_file = st.sidebar.file_uploader("SKATERS", type=["xlsx", "csv"])
shots_file = st.sidebar.file_uploader("SHOT DATA", type=["xlsx", "csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx", "csv"])
lines_file = st.sidebar.file_uploader("LINE DATA", type=["xlsx", "csv"])
teams_file = st.sidebar.file_uploader("TEAMS", type=["xlsx", "csv"])

# ---------------------------------------------------------------
# SMART CACHED DATA LOADING (completely silent on success)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    """Load all five datasets once per session (cached)."""
    with contextlib.redirect_stdout(io.StringIO()):  # suppress prints/spinners
        skaters = load_data(skaters_file, "SKATERS.xlsx")
        shots = load_data(shots_file, "SHOT DATA.xlsx")
        goalies = load_data(goalies_file, "GOALTENDERS.xlsx")
        lines = load_data(lines_file, "LINE DATA.xlsx")
        teams = load_data(teams_file, "TEAMS.xlsx")
    return skaters, shots, goalies, lines, teams


# ✅ Load once (silent)
skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file
)

# Optional manual refresh button
if st.sidebar.button("🔄 Reload Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# ---------------------------------------------------------------
# MAIN APP LOGIC
# ---------------------------------------------------------------
if not skaters_df.empty and not shots_df.empty:
    st.success("✅ Data loaded successfully.")

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
        st.error("⚠️ Missing required columns in files.")
        st.stop()

    # Team dropdowns
    all_teams = sorted(skaters_df[team_col].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A", all_teams)
    with col2:
        team_b = st.selectbox("Select Team B", [t for t in all_teams if t != team_a])

    st.markdown("---")
    run_model = st.button("🚀 Run Model")

    if run_model:
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")

        # -------------------------------------------------------
        # GOALIE ADJUSTMENTS
        # -------------------------------------------------------
        goalie_adj, rebound_rate = {}, {}
        if not goalies_df.empty:
            try:
                df_g = goalies_df.copy()
                df_g = df_g[df_g["situation"].str.lower() == "all"]
                df_g["games"] = pd.to_numeric(df_g["games"], errors="coerce").fillna(0)
                df_g["unblocked attempts"] = pd.to_numeric(df_g["unblocked attempts"], errors="coerce").fillna(0)
                df_g["rebounds"] = pd.to_numeric(df_g["rebounds"], errors="coerce").fillna(0)
                df_g["shots_allowed_per_game"] = np.where(
                    df_g["games"] > 0, df_g["unblocked attempts"] / df_g["games"], np.nan
                )
                df_g["rebound_rate"] = np.where(
                    df_g["unblocked attempts"] > 0, df_g["rebounds"] / df_g["unblocked attempts"], 0
                )
                team_avg = df_g.groupby("team")["shots_allowed_per_game"].mean()
                league_avg = team_avg.mean()
                goalie_adj = (league_avg / team_avg).to_dict()
                rebound_rate = df_g.groupby("team")["rebound_rate"].mean().to_dict()
            except Exception as e:
                st.warning(f"⚠️ Goalie data incomplete: {e}")

        # -------------------------------------------------------
        # LINE ADJUSTMENTS
        # -------------------------------------------------------
        line_adj = pd.DataFrame()
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
                    df_l["games"] > 0, df_l["sog against"] / df_l["games"], np.nan
                )
                team_avg = df_l.groupby("team")["sog_against_per_game"].mean()
                league_avg = team_avg.mean()
                df_l["line_factor"] = (league_avg / df_l["sog_against_per_game"]).clip(0.7, 1.3)
                line_adj = df_l
            except Exception as e:
                st.warning(f"⚠️ Line data incomplete: {e}")

        # -------------------------------------------------------
        # BUILD ROSTER & MODEL
        # -------------------------------------------------------
        roster = (
            skaters_df[skaters_df[team_col].isin([team_a, team_b])][[player_col, team_col]]
            .rename(columns={player_col: "player", team_col: "team"})
            .drop_duplicates()
        )

        shots_df = shots_df.rename(
            columns={player_col_shots: "player", game_col: "gameid", sog_col: "sog"}
        )
        shots_df["player"] = shots_df["player"].astype(str).str.strip()

        results = []
        for _, row in roster.iterrows():
            player, team = row.player, row.team
            df_p = shots_df[shots_df["player"].str.lower() == str(player).lower()]
            if df_p.empty:
                continue

            game_sogs = df_p.groupby("gameid")["sog"].sum().reset_index().sort_values("gameid")
            sog_values = game_sogs["sog"].tolist()
            if not sog_values:
                continue

            last3, last5, last10 = sog_values[-3:], sog_values[-5:], sog_values[-10:]
            l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)
            season_avg = np.mean(sog_values)
            trend = (l3 - l10) / l10 if l10 > 0 else 0
            base_proj = np.nansum([0.5 * l3, 0.3 * l5, 0.2 * l10])

            opp_team = team_b if team == team_a else team_a
            goalie_factor = np.clip(goalie_adj.get(opp_team, 1.0), 0.7, 1.3)
            line_factor = 1.0
            if not line_adj.empty:
                try:
                    last_name = str(player).split()[-1].lower()
                    match = line_adj[line_adj["line pairings"].str.contains(last_name, case=False, na=False)]
                    if not match.empty:
                        line_factor = np.average(match["line_factor"], weights=match["games"])
                    line_factor = np.clip(line_factor, 0.7, 1.3)
                except Exception:
                    pass

            adj_proj = base_proj * (0.7 + 0.3 * goalie_factor) * (0.7 + 0.3 * line_factor)
            adj_proj = round(max(0, adj_proj), 2)

            results.append({
                "Player": player,
                "Team": team,
                "Season Avg": round(season_avg, 2),
                "L3 Shots": ", ".join(map(str, last3)),
                "L3 Avg": round(l3, 2),
                "L5 Shots": ", ".join(map(str, last5)),
                "L5 Avg": round(l5, 2),
                "L10 Shots": ", ".join(map(str, last10)),
                "L10 Avg": round(l10, 2),
                "Trend Score": round(trend, 3),
                "Base Projection": round(base_proj, 2),
                "Goalie Adj": round(goalie_factor, 2),
                "Line Adj": round(line_factor, 2),
                "Adj Projection": adj_proj,
            })

        # -------------------------------------------------------
        # TABLE & RATINGS
        # -------------------------------------------------------
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
        st.markdown(result_df.to_html(index=False, escape=False), unsafe_allow_html=True)

        # -------------------------------------------------------
        # PLAYER TREND CHART
        # -------------------------------------------------------
        st.markdown("### 📈 Player Shot Trend")
        player_names = sorted(shots_df["player"].dropna().unique().tolist())
        selected_player = st.selectbox("Select a player to visualize:", player_names)

        if selected_player:
            clean_names = [str(x).strip().lower() for x in shots_df["player"]]
            close = get_close_matches(selected_player.lower().strip(), clean_names, n=1, cutoff=0.6)
            df_p = shots_df[shots_df["player"].str.lower().str.strip() == close[0]] if close else pd.DataFrame()

            if not df_p.empty:
                df_p = df_p.groupby("gameid")["sog"].sum().reset_index().sort_values("gameid")
                df_p["idx"] = np.arange(len(df_p))
                df_p["roll5"] = df_p["sog"].rolling(5, min_periods=1).mean()
                model = LinearRegression().fit(df_p[["idx"]], df_p["sog"])
                df_p["reg"] = model.predict(df_p[["idx"]])
                slope, r2 = float(model.coef_[0]), float(model.score(df_p[["idx"]], df_p["sog"]))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_p["gameid"], y=df_p["sog"], mode="lines+markers", name="SOG",
                                         line=dict(color="#00B140")))
                fig.add_trace(go.Scatter(x=df_p["gameid"], y=df_p["roll5"], mode="lines", name="5G Avg",
                                         line=dict(color="#2369FF", dash="dot")))
                fig.add_trace(go.Scatter(x=df_p["gameid"], y=df_p["reg"], mode="lines", name="Trend",
                                         line=dict(color="#FF4B4B", width=3)))
                fig.update_layout(height=360, title=f"{selected_player} — SOG Trend",
                                  xaxis_title="Game ID", yaxis_title="Shots on Goal",
                                  legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Slope: {slope:.3f} | R²: {r2:.3f} | Last-5 Avg: {df_p['roll5'].iloc[-1]:.2f}")
            else:
                st.warning(f"⚠️ No shot data found for {selected_player}.")

else:
    st.info("📥 Upload files or use defaults from repo to begin.")
