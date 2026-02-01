# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Production Version
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Team Abbreviation Normalization
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ": "NJD","LA": "LAK","SJ": "SJS","TB": "TBL"
}

st.set_page_config(page_title="Puck Shotz Hockey Analytics", layout="wide", page_icon="🏒")
st.warning("Production Version")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>Shots on Goal projections with confidence context</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])
injuries_file= st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def load_file(f):
    if not f: return pd.DataFrame()
    return pd.read_excel(f) if f.name.endswith(".xlsx") else pd.read_csv(f)

@st.cache_data(show_spinner=False)
def load_all():
    return (
        load_file(skaters_file),
        load_file(shots_file),
        load_file(goalies_file),
        load_file(lines_file),
        load_file(teams_file),
        load_file(injuries_file)
    )

skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all()
if skaters_df.empty or shots_df.empty:
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df]:
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name"
game_col = next(c for c in shots_df.columns if "game" in c and "id" in c)

shots_df = shots_df.rename(columns={next(c for c in shots_df.columns if "player" in c):"player"})
shots_df["player"] = shots_df["player"].str.lower().str.strip()

# ---------------------------------------------------------------
# Fighting Chance Helper
# ---------------------------------------------------------------
def fighting_chance(last10, line_adj, threshold):
    arr = np.array(last10)
    hits = np.sum(arr >= threshold)
    floors = np.sum(arr <= 1)

    if threshold == 3:
        freq = 5 if hits >= 7 else 4 if hits == 6 else 3 if hits == 5 else 2 if hits == 4 else 1 if hits == 3 else 0
    else:
        freq = 5 if hits >= 5 else 4 if hits == 4 else 3 if hits == 3 else 2 if hits == 2 else 1 if hits == 1 else 0

    score = freq - min(floors,3) + (line_adj - 1) * 5

    label = "🟢 Strong" if score >= 4 else "🟡 Live" if score >= 2 else "🔴 Fragile"
    return round(score,2), label

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
def build_model(team_a, team_b):
    results = []
    roster = skaters_df[skaters_df[team_col].isin([team_a,team_b])][[player_col,team_col]].drop_duplicates()
    grouped = {n:g for n,g in shots_df.groupby("player")}

    for _, r in roster.iterrows():
        player, team = r[player_col], r[team_col]
        df_p = grouped.get(player.lower())
        if df_p is None: continue

        sog_vals = df_p.groupby(game_col)["sog"].sum().tolist()
        if len(sog_vals) < 5: continue

        last3, last5, last10 = sog_vals[-3:], sog_vals[-5:], sog_vals[-10:]
        l3, l5, l10 = map(np.mean, [last3,last5,last10])

        baseline = 0.55*l10 + 0.3*l5 + 0.15*l3

        line_adj = 1.0
        if not lines_df.empty and "line pairings" in lines_df.columns:
            m = lines_df[lines_df["line pairings"].str.contains(player.split()[-1],case=False,na=False)]
            if not m.empty:
                line_adj = m["line_factor"].mean()

        fc3s, fc3 = fighting_chance(last10, line_adj, 3)
        fc4s, fc4 = fighting_chance(last10, line_adj, 4)

        lam = baseline * line_adj
        prob = 1 - poisson.cdf(2, lam)

        results.append({
            "Player": player,
            "Team": team,
            "Final Projection": round(lam,2),
            "FC 3SOG": fc3,
            "FC 3SOG Score": fc3s,
            "FC 4SOG": fc4,
            "FC 4SOG Score": fc4s,
            "L10 Shots": ", ".join(map(str,last10))
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if st.button("🚀 Run Model"):
    all_games = []
    for t in skaters_df[team_col].unique():
        df = build_model(t,t)
        all_games.append(df)

    final = pd.concat(all_games).sort_values("FC 3SOG Score",ascending=False)
    st.dataframe(final, use_container_width=True)


# ---------------------------------------------------------------
# Run Model + Combine Games
# ---------------------------------------------------------------
if run_model:
    all_tables=[]
    for m in games:
        team_a,team_b=m["away"],m["home"]
        df=build_model(team_a,team_b,skaters_df,shots_df,goalies_df,lines_df,teams_df,injuries_df)
        if not df.empty:
            df["Matchup"]=f"{team_a}@{team_b}"
            all_tables.append(df)
    if all_tables:
        combined=pd.concat(all_tables,ignore_index=True)
        st.session_state.results=combined
        st.session_state.matchups=games
        st.success("✅ Model built for all games.")
        st.experimental_rerun()
    else:
        st.warning("⚠️ No valid data generated.")

# ---------------------------------------------------------------
# Display Buttons + Table
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results.copy()
    games=st.session_state.matchups

    # Matchup buttons
    cols = st.columns(3)
    for i, m in enumerate(games):
        team_a, team_b = m["away"], m["home"]
        match_id = f"{team_a}@{team_b}"
        is_selected = st.session_state.get("selected_match") == match_id
        btn_color = "#2F7DEB" if is_selected else "#1C5FAF"
        border = "2px solid #FF4B4B" if is_selected else "1px solid #1C5FAF"
        glow = "0 0 12px #FF4B4B" if is_selected else "none"
        with cols[i % 3]:
            form_key = f"form_{i}"
            with st.form(form_key):
                st.markdown(f"""
                <div style="background-color:{btn_color};border:{border};border-radius:8px 8px 0 0;
                            color:#fff;font-weight:600;font-size:15px;padding:10px 14px;width:100%;
                            box-shadow:{glow};display:flex;align-items:center;justify-content:center;gap:6px;">
                    <img src="{m['away_logo']}" height="22">
                    <span>{m['away']}</span>
                    <span style="color:#D6D6D6;">@</span>
                    <span>{m['home']}</span>
                    <img src="{m['home_logo']}" height="22">
                </div>
                """, unsafe_allow_html=True)
                clicked = st.form_submit_button("Click to view", use_container_width=True, type="secondary")
                if clicked:
                    if is_selected:
                        st.session_state.selected_match=None
                        st.session_state.selected_teams=None
                    else:
                        st.session_state.selected_match=match_id
                        st.session_state.selected_teams={team_a,team_b}
                    st.rerun()

    sel_teams=st.session_state.get("selected_teams")
    if sel_teams:
        df=df[df["Team"].isin(sel_teams)]
        st.markdown(f"### Showing results for: **{' vs '.join(sel_teams)}**")
    else:
        st.markdown("### Showing results for: **All Teams**")

    def color_trend(v):
        if v>0.05: return "<span style='color:#00FF00;font-weight:bold;'>▲</span>"
        elif v<-0.05: return "<span style='color:#FF4B4B;font-weight:bold;'>▼</span>"
        else: return "<span style='color:#D6D6D6;'>–</span>"

    def color_form(v):
        if "Above" in v: return "<span style='color:#00FF00;font-weight:bold;'>🟢 Above Baseline</span>"
        elif "Below" in v: return "<span style='color:#FF4B4B;font-weight:bold;'>🔴 Below Baseline</span>"
        else: return "<span style='color:#D6D6D6;'>⚪ Neutral</span>"

    df["Trend"]=df["Trend Score"].apply(color_trend)
    df["Form Indicator"]=df["Form Indicator"].apply(color_form)

    if "line_test_val" in st.session_state:
        test_line=st.session_state.line_test_val
        df["Prob ≥ Line (%)"]=df["Final Projection"].apply(
            lambda lam:round((1-poisson.cdf(test_line-1,mu=max(lam,0.01)))*100,1))
        def safe_odds(p):
            p=np.clip(p,0.1,99.9)
            if p>=50: odds_val=-100*((p/100)/(1-p/100))
            else: odds_val=100*((1-p/100)/(p/100))
            return f"{'+' if odds_val>0 else ''}{int(round(odds_val))}"
        df["Playable Odds"]=df["Prob ≥ Line (%)"].apply(safe_odds)

    df=df.sort_values(["Team","Final Projection","Line Adj"],ascending=[True,False,False])
    cols=["Player","Team","Injury","Trend","Final Projection","Prob ≥ Line (%)",
          "Playable Odds","Season Avg","Line Adj","Exp Goals (xG)","Shooting %",
          "Form Indicator","L3 Shots","L5 Shots","L10 Shots"]
    existing_cols=[c for c in cols if c in df.columns]
    html_table=df[existing_cols].to_html(index=False,escape=False)
    csv = df[existing_cols].to_csv(index=False).encode("utf-8")

    st.download_button(
    label="💾 Download Results (CSV)",
    data=csv,
    file_name="puck_shotz_results.csv",
    mime="text/csv"
)


    components.html(f"""
    <style>
    table {{
        width:100%;border-collapse:collapse;font-family:'Source Sans Pro',sans-serif;color:#D6D6D6;
    }}
    th {{
        background-color:#0A3A67;color:#FFFFFF;padding:6px;text-align:center;position:sticky;top:0;
        border-bottom:2px solid #1E5A99;
    }}
    td:first-child,th:first-child {{
        position:sticky;left:0;background-color:#1E5A99;color:#FFFFFF;font-weight:bold;
    }}
    td {{
        background-color:#0F2743;color:#D6D6D6;padding:4px;text-align:center;
    }}
    tr:nth-child(even) td {{background-color:#142F52;}}
    td:nth-child(10), td:nth-child(11) {{
        color:#7FFF00;font-weight:bold;
    }}
    </style>
    <div style='overflow-x:auto;height:650px;'>{html_table}</div>
    """,height=700,scrolling=True)
