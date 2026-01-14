# hockey_prop_stop_app.py — Streamlit debug loader
import importlib.util
import os
import traceback
import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Hockey Prop Stop — Debug", layout="wide", page_icon="🏒")
st.title("🏒 Hockey Prop Stop — Debug Loader")

# --- Show the exact model path being used
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
st.code(f"Loading model from:\n{module_path}", language="bash")
print("DEBUG: Loading model from path:", module_path)

# --- Dynamic import with error display in UI
try:
    spec = importlib.util.spec_from_file_location("hockey_model", module_path)
    hockey_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hockey_model)
    st.sidebar.success("✅ hockey_model loaded successfully.")
except Exception as e:
    st.sidebar.error("❌ Failed to load hockey_model.py.")
    st.exception(e)
    st.stop()

parse_raw_files = hockey_model.parse_raw_files
project_matchup  = hockey_model.project_matchup

# --- Uploads
st.sidebar.header("📂 Upload CSV Files")
uploaded_skaters = st.sidebar.file_uploader("NHL Skaters.csv", type=["csv"])
uploaded_teams   = st.sidebar.file_uploader("NHL TEAMs.csv", type=["csv"])
uploaded_shots   = st.sidebar.file_uploader("shots.csv", type=["csv"])
uploaded_goalies = st.sidebar.file_uploader("goalies.csv", type=["csv"])
uploaded_lines   = st.sidebar.file_uploader("lines.csv", type=["csv"])

raw_files = {
    "skaters": pd.read_csv(uploaded_skaters) if uploaded_skaters else pd.DataFrame(),
    "teams":   pd.read_csv(uploaded_teams)   if uploaded_teams   else pd.DataFrame(),
    "shots":   pd.read_csv(uploaded_shots)   if uploaded_shots   else pd.DataFrame(),
    "goalies": pd.read_csv(uploaded_goalies) if uploaded_goalies else pd.DataFrame(),
    "lines":   pd.read_csv(uploaded_lines)   if uploaded_lines   else pd.DataFrame(),
}

if all([uploaded_skaters, uploaded_teams, uploaded_shots, uploaded_goalies, uploaded_lines]):
    st.success("✅ All five files uploaded.")

    # Wrap parse in try/except so any error shows in the UI
    try:
        skaters_df, teams_df, shots_df, goalies_df, lines_df, all_teams = parse_raw_files(raw_files)
        st.success(f"✅ Parsed data. Detected {len(all_teams)} team(s).")
        st.write("**Teams list:**", all_teams)
        st.write("**Shots columns (first 25):**", list(shots_df.columns)[:25])
    except Exception as e:
        st.error("❌ Error during parse_raw_files()")
        st.exception(e)
        st.stop()

    # team selectors
    if len(all_teams) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            team_a = st.selectbox("Select Team A", options=all_teams)
        with col2:
            team_b = st.selectbox("Select Team B", options=[t for t in all_teams if t != team_a])
    else:
        st.error("❌ Not enough teams detected to build a matchup. Check your skaters/teams files.")
        st.stop()

    if st.button("🚀 Run Model"):
        st.info(f"Building model for **{team_a} vs {team_b}** ...")
        try:
            data = project_matchup(skaters_df, teams_df, shots_df, goalies_df, lines_df, team_a, team_b)
        except Exception as e:
            st.error("❌ Error during project_matchup()")
            st.exception(e)
            st.stop()

        if data is None or data.empty:
            st.error("⚠️ No valid projections generated.")
            # EXTRA DEBUG SURFACE
            try:
                st.write("**Unique teams in SHOTS (raw):**", sorted(shots_df.get("teamCode", pd.Series(dtype=object)).dropna().unique().tolist()))
                st.write("**Unique teams in SKATERS:**", sorted(skaters_df.get("team", pd.Series(dtype=object)).dropna().unique().tolist()))
                st.write("**First 10 rows of player_form (if available):**")
                # Try to call the internal function if accessible:
                if hasattr(hockey_model, "build_player_form"):
                    pf = hockey_model.build_player_form(shots_df)
                    st.dataframe(pf.head(10))
            except Exception as e:
                st.warning("Could not render extra debug info.")
                st.exception(e)
        else:
            st.success(f"✅ Generated {len(data)} projections.")
            st.dataframe(data, use_container_width=True)

            out = BytesIO()
            data.to_excel(out, index=False)
            st.download_button(
                label="Download Excel",
                data=out.getvalue(),
                file_name=f"HockeyPropStop_{team_a}_vs_{team_b}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
else:
    st.info("📥 Upload all five CSV files to begin.")

st.caption("© Hockey Prop Stop — debug build")
