# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — L5 Probability Update + Manual Results + Safe Widget IDs
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
    st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
    skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
    shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
    goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
    lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
    teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])

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
        if file_uploader is not None:
            return load_file(file_uploader)
        return safe_read(default_path)

    @st.cache_data(show_spinner=False)
    def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
        base_paths = [".", "data", "/mount/src/hockey-prop-stop/data"]
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

    skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
        skaters_file, shots_file, goalies_file, lines_file, teams_file
    )
    if skaters_df.empty or shots_df.empty:
        st.warning("⚠️ Missing required data. Please upload or verify repo files.")
        st.stop()
    st.success("✅ Data loaded successfully.")

    for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
        if not df.empty: df.columns = df.columns.str.lower().str.strip()

    team_col = next((c for c in skaters_df.columns if "team" in c), None)
    player_col = "name" if "name" in skaters_df.columns else None
    player_col_shots = next((c for c in shots_df.columns if "player" in c or "name" in c), None)
    shots_df = shots_df.rename(columns={player_col_shots: "player"})
    shots_df["player"] = shots_df["player"].astype(str).str.strip()
    sog_col  = next((c for c in shots_df.columns if "sog" in c), None)
    goal_col = next((c for c in shots_df.columns if "goal" in c), None)
    game_col = next((c for c in shots_df.columns if "game" in c and "id" in c), None)

    teams = sorted(skaters_df[team_col].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1: team_a = st.selectbox("Select Team A", teams)
    with col2: team_b = st.selectbox("Select Team B", [t for t in teams if t != team_a])
    st.markdown("---")

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
            last5 = sog_values[-5:] if len(sog_values)>=5 else sog_values
            last10 = sog_values[-10:] if len(sog_values)>=10 else sog_values
            l5 = np.mean(last5); l10 = np.mean(last10)
            season_avg = np.mean(sog_values)
            lam = l5
            line = round(lam, 2)
            prob = 1 - poisson.cdf(np.floor(line) - 1, mu=lam)
            p = min(max(prob, 0.001), 0.999)
            odds = -100*(p/(1-p)) if p>=0.5 else 100*((1-p)/p)
            implied_odds = f"{'+' if odds>0 else ''}{int(odds)}"
            results.append({
                "Player": player, "Team": team,
                "Season Avg": round(season_avg,2),
                "Final Projection": round(line,2),
                "Prob ≥ Projection (%) L5": round(p*100,1),
                "Playable Odds": implied_odds
            })
        return pd.DataFrame(results)

    if st.button("🚀 Run Model"):
        st.info(f"Building model for matchup: **{team_a} vs {team_b}** …")
        df = build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df)
        df = df.sort_values("Final Projection", ascending=False).reset_index(drop=True)
        st.session_state.results_raw = df.copy()
        st.success("✅ Model built successfully!")

    if "results_raw" in st.session_state and not st.session_state.results_raw.empty:
        df = st.session_state.results_raw.copy()
        st.dataframe(df)

        selected_date = st.date_input("Select game date:", datetime.date.today())
        if st.button("💾 Save Projections for Selected Date"):
            df_to_save = df.copy()
            df_to_save["Date_Game"] = selected_date.strftime("%Y-%m-%d")
            df_to_save["Matchup"] = f"{team_a} vs {team_b}"
            os.makedirs("projections", exist_ok=True)
            master = "model_results.csv"
            if os.path.exists(master):
                existing = pd.read_csv(master)
                combined = pd.concat([existing, df_to_save], ignore_index=True)
                combined.to_csv(master, index=False)
            else:
                df_to_save.to_csv(master, index=False)
            matchup_file = f"projections/{team_a}_vs_{team_b}_{selected_date.strftime('%Y-%m-%d')}.csv"
            df_to_save.to_csv(matchup_file, index=False)
            st.success(f"✅ Saved projections for {selected_date}!")

# ===============================================================
# 📊 RESULTS DASHBOARD TAB
# ===============================================================
with tab2:
    st.header("📊 Results Dashboard")

    master = "model_results.csv"
    if not os.path.exists(master):
        st.info("No saved projections yet. Run model and click 'Save Projections' first.")
    else:
        df = pd.read_csv(master)
        st.markdown("### 📅 Saved Projection History")
        st.dataframe(df.tail(50))

        # ---------------------- Manual Entry ----------------------
        st.markdown("### 📝 Enter Results Manually")
        if not df.empty:
            available_games = df[["Matchup","Date_Game"]].drop_duplicates()
            available_games["Label"] = available_games["Matchup"] + " — " + available_games["Date_Game"].astype(str)
            game_selection = st.selectbox("Select matchup to enter results for:", available_games["Label"])

            if game_selection:
                sel_row = available_games.loc[available_games["Label"] == game_selection].iloc[0]
                matchup, date_game = sel_row["Matchup"], sel_row["Date_Game"]
                sub_df = df[(df["Matchup"] == matchup) & (df["Date_Game"] == date_game)].copy()

                st.write(f"**Entering results for {matchup} ({date_game})**")
                actuals = {}

                # Generate a stable unique run_id for widget key names
                if "run_id" not in st.session_state:
                    st.session_state["run_id"] = int(datetime.datetime.now().timestamp())

                run_id = st.session_state["run_id"]

                for i, player in enumerate(sub_df["Player"].unique()):
                    safe_key = f"res_{run_id}_{i}_{abs(hash(player)) % 1000000}"
                    actuals[player] = st.number_input(
                        f"{player} SOG:",
                        min_value=0,
                        step=1,
                        key=safe_key
                    )

                if st.button(f"💾 Save Entered Results for {matchup} ({date_game})"):
                    sub_df["Actual_SOG"] = sub_df["Player"].map(actuals)
                    sub_df["Hit"] = (sub_df["Actual_SOG"] >= sub_df["Final Projection"]).astype(int)
                    sub_df["Evaluation_Date"] = datetime.date.today().strftime("%Y-%m-%d")

                    eval_path = "results_evaluated.csv"
                    if os.path.exists(eval_path):
                        existing_eval = pd.read_csv(eval_path)
                        combined_eval = pd.concat([existing_eval, sub_df], ignore_index=True)
                        combined_eval.to_csv(eval_path, index=False)
                    else:
                        sub_df.to_csv(eval_path, index=False)

                    hit_rate = sub_df["Hit"].mean() * 100
                    st.success(f"✅ Results saved. Hit rate for this matchup: {hit_rate:.1f}%")

        # ---------------------- Clear Data ----------------------
        st.markdown("### 🧹 Clear Saved Results")
        if st.button("🧹 Clear Saved Results"):
            for path in ["model_results.csv", "results_evaluated.csv"]:
                if os.path.exists(path): os.remove(path)
            if os.path.exists("projections"):
                for f in os.listdir("projections"):
                    if f.endswith(".csv"):
                        os.remove(os.path.join("projections", f))
            st.success("✅ All saved projections and results cleared.")
