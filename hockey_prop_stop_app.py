# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — L5 Probability + Form Indicator + Dashboard
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, datetime, pytz, subprocess
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page Setup + Tabs
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with blended regression and L5-based probabilities
    </p>
    """,
    unsafe_allow_html=True,
)
tab1, tab2 = st.tabs(["🏒 Model", "📊 Results Dashboard"])

# ===============================================================
# 🏒 MODEL TAB
# ===============================================================
with tab1:
    # Sidebar Uploaders
    st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
    skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
    shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
    goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
    lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
    teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])

    # Helper Functions
    def load_file(file):
        if not file: return pd.DataFrame()
        try:
            return pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
        except Exception:
            return pd.DataFrame()

    def safe_read(path):
        try:
            if not os.path.exists(path): return pd.DataFrame()
            return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def load_data(file_uploader, default_path):
        return load_file(file_uploader) if file_uploader is not None else safe_read(default_path)

    # Cached Data Load
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

    # Load Data
    skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
        skaters_file, shots_file, goalies_file, lines_file, teams_file
    )
    if skaters_df.empty or shots_df.empty:
        st.warning("⚠️ Missing required data. Please upload or verify repo files.")
        st.stop()
    st.success("✅ Data loaded successfully.")

    # Normalize Columns
    for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
        if not df.empty: df.columns = df.columns.str.lower().str.strip()

    team_col = next((c for c in skaters_df.columns if "team" in c), None)
    player_col = "name" if "name" in skaters_df.columns else None
    toi_col, gp_col = "icetime", "games"

    player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
    shots_df = shots_df.rename(columns={player_col_shots: "player"})
    shots_df["player"] = shots_df["player"].astype(str).str.strip()

    sog_col  = next((c for c in shots_df.columns if "sog" in c), None)
    goal_col = next((c for c in shots_df.columns if "goal" in c), None)
    game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)

    # Team Selection
    teams = sorted(skaters_df[team_col].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1: team_a = st.selectbox("Select Team A", teams)
    with col2: team_b = st.selectbox("Select Team B", [t for t in teams if t != team_a])
    st.markdown("---")

    # Model Build
    @st.cache_data(show_spinner=True)
    def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df):
        results = []
        skaters = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
        roster = skaters[[player_col, team_col]].rename(columns={player_col:"player", team_col:"team"}).drop_duplicates("player")
        grouped = {n.lower():g.sort_values(game_col) for n,g in shots_df.groupby(shots_df["player"].str.lower())}

        for row in roster.itertuples(index=False):
            player, team = row.player, row.team
            df_p = grouped.get(player.lower(), pd.DataFrame())
            if df_p.empty: continue

            game_sogs = df_p.groupby(game_col)[["sog","goal"]].sum().reset_index().sort_values(game_col)
            sog_values = game_sogs["sog"].tolist()
            if not sog_values: continue

            last3 = sog_values[-3:] if len(sog_values)>=3 else sog_values
            last5 = sog_values[-5:] if len(sog_values)>=5 else sog_values
            last10 = sog_values[-10:] if len(sog_values)>=10 else sog_values

            l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)
            season_avg = np.mean(sog_values)
            trend = (l5 - l10)/l10 if l10>0 else 0

            lam = l5
            line = round(lam, 2)
            prob = 1 - poisson.cdf(np.floor(line) - 1, mu=lam)
            p = min(max(prob, 0.001), 0.999)
            odds = -100*(p/(1-p)) if p>=0.5 else 100*((1-p)/p)
            implied_odds = f"{'+' if odds>0 else ''}{int(odds)}"

            # Form Indicator
            form_flag = "⚪ Neutral Form"
            try:
                season_toi = pd.to_numeric(
                    skaters_df.loc[skaters_df[player_col].str.lower() == player.lower(), "icetime"],
                    errors="coerce"
                ).mean()
                games_played = pd.to_numeric(
                    skaters_df.loc[skaters_df[player_col].str.lower() == player.lower(), "games"],
                    errors="coerce"
                ).mean()
                if season_toi > 0 and games_played >= 10:
                    avg_toi = (season_toi / games_played) / 60.0
                    sog_per60 = (season_avg / avg_toi) * 60
                    blended_recent = 0.7 * l5 + 0.3 * l10
                    recent_per60 = (blended_recent / avg_toi) * 60 if avg_toi > 0 else 0
                    opponent = team_b if team == team_a else team_a
                    if not teams_df.empty and "shots_against_per_game" in teams_df.columns:
                        opp_rate = teams_df.loc[
                            teams_df["team"].str.lower() == opponent.lower(), "shots_against_per_game"
                        ]
                        league_avg = teams_df["shots_against_per_game"].mean()
                        opp_factor = float(league_avg / opp_rate.mean()) if not opp_rate.empty else 1.0
                    else:
                        opp_factor = 1.0
                    recent_adj = recent_per60 * opp_factor
                    usage_delta = (recent_adj - sog_per60) / sog_per60 if sog_per60 > 0 else 0
                    if usage_delta > 0.10:
                        form_flag = "🟢 Above-Baseline Form"
                    elif usage_delta < -0.10:
                        form_flag = "🔴 Below-Baseline Form"
            except Exception:
                pass

            results.append({
                "Player": player,
                "Team": team,
                "Season Avg": round(season_avg, 2),
                "L3 Shots": ", ".join(map(str, last3)),
                "L5 Shots": ", ".join(map(str, last5)),
                "L10 Shots": ", ".join(map(str, last10)),
                "Trend Score": round(trend, 3),
                "Final Projection": round(line, 2),
                "Prob ≥ Projection (%) L5": round(p * 100, 1),
                "Playable Odds": implied_odds,
                "Form Indicator": form_flag
            })
        return pd.DataFrame(results)

    # Run Model
    if st.button("🚀 Run Model"):
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")
        df = build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df)
        df = df.sort_values("Final Projection", ascending=False).reset_index(drop=True)
        st.session_state.results_raw = df.copy()
        st.success("✅ Model built successfully!")

    # Display + Save
    if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
        df = st.session_state.results_raw.copy()
        st.dataframe(df)

        # Save Projections
        st.markdown("### 💾 Save Projections")
        selected_date = st.date_input("Select game date:", datetime.date.today())
        if st.button("💾 Save Projections for Selected Date"):
            out_df = df.copy()
            out_df["Date_Game"] = selected_date.strftime("%Y-%m-%d")
            out_df["Matchup"] = f"{team_a} vs {team_b}"
            os.makedirs("projections", exist_ok=True)
            master_path = "model_results.csv"
            if os.path.exists(master_path):
                existing = pd.read_csv(master_path)
                missing = [c for c in out_df.columns if c not in existing.columns]
                for c in missing: existing[c] = np.nan
                pd.concat([existing, out_df], ignore_index=True).to_csv(master_path, index=False)
            else:
                out_df.to_csv(master_path, index=False)
            matchup_file = f"projections/{team_a}_vs_{team_b}_{out_df['Date_Game'].iloc[0]}.csv"
            out_df.to_csv(matchup_file, index=False)
            st.success(f"✅ Saved projections for {selected_date} to both {master_path} and {matchup_file}")

# ===============================================================
# 📊 RESULTS DASHBOARD TAB
# ===============================================================
with tab2:
    st.header("📊 Model Results Dashboard")

    # Clear Results Section
    st.markdown("### 🧹 Manage Results Data")
    if os.path.exists("model_results.csv"):
        with st.expander("⚙️ Clear or Manage Saved Results"):
            st.warning("Clearing will permanently delete your saved projection history.")
            clear_master = st.checkbox("Delete rolling file (model_results.csv)")
            clear_matchups = st.checkbox("Delete individual matchup CSVs in /projections")
            if st.button("🧹 Clear Results Table"):
                if clear_master and os.path.exists("model_results.csv"):
                    os.remove("model_results.csv")
                    st.success("✅ Deleted model_results.csv")
                if clear_matchups and os.path.exists("projections"):
                    for f in os.listdir("projections"):
                        if f.endswith(".csv"):
                            os.remove(os.path.join("projections", f))
                    st.success("✅ Cleared projections folder")
                if not clear_master and not clear_matchups:
                    st.info("No selections made — nothing deleted.")

    master_path = "model_results.csv"
    if not os.path.exists(master_path):
        st.info("No results logged yet. Run and save projections first.")
    else:
        df = pd.read_csv(master_path)
        st.markdown("### 📅 Projection History")
        st.dataframe(df.tail(30))
        st.markdown("#### Upload Actual Results")
        actuals_file = st.file_uploader("Upload updated SHOT DATA to evaluate accuracy", type=["csv","xlsx"])
        if actuals_file:
            actuals = pd.read_excel(actuals_file) if actuals_file.name.endswith(".xlsx") else pd.read_csv(actuals_file)
            actuals.columns = actuals.columns.str.lower().str.strip()
            player_col = next((c for c in actuals.columns if "player" in c or "name" in c), None)
            sog_col = next((c for c in actuals.columns if "sog" in c), None)
            if player_col and sog_col:
                latest = actuals.groupby(player_col)[sog_col].mean().reset_index()
                latest.columns = ["Player", "Actual_SOG"]
                df_eval = df.merge(latest, on="Player", how="left")
                df_eval["Hit"] = (df_eval["Actual_SOG"] >= df_eval["Final Projection"]).astype(int)
                hit_rate = df_eval["Hit"].mean() * 100
                mae = np.mean(np.abs(df_eval["Actual_SOG"] - df_eval["Final Projection"]))
                st.metric("Overall Hit Rate", f"{hit_rate:.1f}%")
                st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")

                st.markdown("#### 🟢 Top Performing Players")
                st.dataframe(
                    df_eval.groupby("Player")["Hit"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )

                st.markdown("#### 🔴 Bottom Performing Players")
                st.dataframe(
                    df_eval.groupby("Player")["Hit"]
                    .mean()
                    .sort_values(ascending=True)
                    .head(10)
                    .reset_index()
                )

                if "Date_Game" in df_eval.columns:
                    daily = df_eval.groupby("Date_Game")["Hit"].mean().reset_index()
                    daily["Hit"] = daily["Hit"] * 100
                    st.markdown("#### 📈 Daily Hit Rate Trend")
                    st.line_chart(daily, x="Date_Game", y="Hit", use_container_width=True)
            else:
                st.warning("Could not find 'player' or 'sog' column in uploaded file.")
        else:
            st.info("Upload updated shot data to evaluate accuracy.")
