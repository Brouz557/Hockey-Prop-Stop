import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

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
# Custom CSS for table readability
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

    if run_model:
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")

        # -------------------------------------------------------
        # 🥅 GOALIE ADJUSTMENTS
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
        # 🧱 LINE ADJUSTMENTS (Weighted by games)
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
                line_adj = df_l
            except Exception as e:
                st.warning(f"⚠️ Line data incomplete or invalid: {e}")

        # -------------------------------------------------------
        # Build team roster and results
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
            l3, l5, l10 = np.mean(last3) if last3 else np.nan, np.mean(last5) if last5 else np.nan, np.mean(last10) if last10 else np.nan
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
                    matching = line_adj[line_adj["line pairings"].str.contains(last_name, case=False, na=False)]
                    if not matching.empty:
                        line_factor = np.average(matching["line_factor"], weights=matching["games"])
                    line_factor = np.clip(line_factor, 0.7, 1.3)
                except Exception:
                    pass

            adj_proj = base_proj * (0.7 + 0.3 * goalie_factor) * (0.7 + 0.3 * line_factor)
            adj_proj *= (1 + rebound_factor * 0.1)
            adj_proj = max(0, round(adj_proj, 2))

            l10_shots_formatted = (
                "<br>".join([", ".join(map(str, last10[:5])), ", ".join(map(str, last10[5:]))])
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
            return "Weak"

        result_df["Matchup Rating"] = result_df["Adj Projection"].apply(rate)
        result_df = result_df.sort_values("Adj Projection", ascending=False)
        st.session_state["result_df"], st.session_state["shots_df"] = result_df, shots_df

        st.success(f"✅ Model built successfully for {team_a} vs {team_b}!")
        st.markdown(f"### 📊 {team_a} vs {team_b} — Player Projections (Adjusted)")
        st.markdown(result_df.to_html(index=False, escape=False), unsafe_allow_html=True)

# ---------------------------------------------------------------
# 🔥 Player SOG Heatmap — Recent Game Trends
# ---------------------------------------------------------------
if "result_df" in st.session_state and "shots_df" in st.session_state:
    result_df = st.session_state["result_df"]
    shots_df = st.session_state["shots_df"]

    st.markdown("### 🔥 Player Shot Trend Heatmap")
    selected_player = st.selectbox("Select a player to visualize:", result_df["Player"].unique())

    if selected_player:
        df_p = shots_df[shots_df["player"].str.lower() == selected_player.lower()].copy()
        if not df_p.empty:
            df_p = df_p.groupby("gameid")["sog"].sum().reset_index().sort_values("gameid").tail(20)
            data = df_p.set_index("gameid").T
            fig, ax = plt.subplots(figsize=(10, 1.5))
            sns.heatmap(
                data,
                cmap="YlGnBu",
                cbar_kws={"orientation": "horizontal", "label": "Shots on Goal"},
                linewidths=0.5,
                linecolor="gray",
                ax=ax,
            )
            ax.set_xlabel("Game ID")
            ax.set_yticks([])
            ax.set_title(f"{selected_player} — Last {len(df_p)} Games SOG Heatmap", pad=10)
            st.pyplot(fig)

            avg_sog = df_p["sog"].mean()
            rolling = df_p["sog"].rolling(5).mean()
            trend = rolling.iloc[-1] - rolling.iloc[0] if len(rolling.dropna()) >= 5 else 0
            direction = "📈 rising" if trend > 0 else "📉 falling" if trend < 0 else "➡ steady"
            st.caption(f"🔥 Average SOG: {avg_sog:.2f} | Trend (5-game): {direction}")
        else:
            st.warning("No shot data found for that player.")
else:
    st.info("📥 Upload at least SKATERS and SHOT DATA to begin.")
