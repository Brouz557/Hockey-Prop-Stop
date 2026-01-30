# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (Instant Filter + Logos + Injuries + xG)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Team Abbreviation Normalization (ESPN -> Data)
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ":"NJD","LA":"LAK","SJ":"SJS","TB":"TBL","ARI":"ARI","ANA":"ANA","BOS":"BOS","BUF":"BUF",
    "CAR":"CAR","CBJ":"CBJ","CGY":"CGY","CHI":"CHI","COL":"COL","DAL":"DAL","DET":"DET","EDM":"EDM",
    "FLA":"FLA","MIN":"MIN","MTL":"MTL","NSH":"NSH","NYI":"NYI","NYR":"NYR","OTT":"OTT","PHI":"PHI",
    "PIT":"PIT","SEA":"SEA","STL":"STL","TOR":"TOR","VAN":"VAN","VGK":"VGK","WSH":"WSH","WPG":"WPG"
}

st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("Production Version")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups with inline logos, instant filters, injuries, and expected goals.</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])
injuries_file= st.sidebar.file_uploader("INJURIES", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def load_file(f):
    if not f: return pd.DataFrame()
    try:
        return pd.read_excel(f) if f.name.lower().endswith(".xlsx") else pd.read_csv(f)
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
def load_all(skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file):
    base_paths=[".","data","/mount/src/hockey-prop-stop/data"]
    def find_file(name):
        for p in base_paths:
            fp=os.path.join(p,name)
            if os.path.exists(fp): return fp
        return None
    skaters=load_data(skaters_file, find_file("Skaters.xlsx") or "Skaters.xlsx")
    shots  =load_data(shots_file,   find_file("SHOT DATA.xlsx") or "SHOT DATA.xlsx")
    goalies=load_data(goalies_file, find_file("GOALTENDERS.xlsx") or "GOALTENDERS.xlsx")
    lines  =load_data(lines_file,   find_file("LINE DATA.xlsx") or "LINE DATA.xlsx")
    teams  =load_data(teams_file,   find_file("TEAMS.xlsx") or "TEAMS.xlsx")

    injuries=pd.DataFrame()
    for p in ["injuries.xlsx","Injuries.xlsx","data/injuries.xlsx"]:
        if os.path.exists(p):
            injuries=load_file(open(p,"rb"));break
    if injuries.empty:
        injuries=load_file(injuries_file)
    if not injuries.empty:
        injuries.columns=injuries.columns.str.lower().str.strip()
        injuries["player"]=injuries["player"].astype(str).str.strip().str.lower()
    return skaters,shots,goalies,lines,teams,injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file)

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty:
        df.columns=df.columns.str.lower().str.strip()

team_col=next((c for c in skaters_df.columns if "team" in c),None)
player_col="name"
shots_df["player"]=shots_df["player"].astype(str).str.strip()
game_col=next((c for c in shots_df.columns if "game" in c and "id" in c),None)

# ---------------------------------------------------------------
# 🔥 OPPONENT POSITION MATCHUP PROFILE (NEW)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_position_matchups(shots_df, skaters_df, game_col):

    def norm(x):
        return str(x).lower().replace(".","").replace(",","").strip()

    s=shots_df.copy()
    k=skaters_df.copy()

    s["player_norm"]=s["player"].apply(norm)
    k["player_norm"]=k["name"].apply(norm)

    s=s.merge(k[["player_norm","position"]],on="player_norm",how="left")
    s=s.dropna(subset=["position","sog",game_col])

    per_game=s.groupby(["team","position",game_col])["sog"].sum().reset_index()
    avg=per_game.groupby(["team","position"])["sog"].mean().reset_index()

    league_avg=avg.groupby("position")["sog"].mean().to_dict()
    avg["pos_factor"]=avg.apply(lambda r:r["sog"]/league_avg.get(r["position"],r["sog"]),axis=1)
    avg["pos_factor"]=avg["pos_factor"].clip(0.85,1.20)

    return {(r["team"],r["position"]):r["pos_factor"] for _,r in avg.iterrows()}

pos_matchup_adj=build_position_matchups(shots_df,skaters_df,game_col)

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):

    results=[]
    skaters=skaters_df[skaters_df[team_col].isin([team_a,team_b])]
    roster=skaters[[player_col,team_col,"position"]].rename(
        columns={player_col:"player",team_col:"team"}
    ).drop_duplicates("player")

    grouped={n.lower():g for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    goalie_adj={}
    if {"team","shots against","games"}.issubset(goalies_df.columns):
        g=goalies_df.copy()
        g["shots against"]=pd.to_numeric(g["shots against"],errors="coerce").fillna(0)
        g["games"]=pd.to_numeric(g["games"],errors="coerce").fillna(1)
        g["shots_per_game"]=g["shots against"]/g["games"]
        g["goalie_factor"]=(g["shots_per_game"]/g["shots_per_game"].mean()).clip(0.7,1.3)
        goalie_adj=g.groupby("team")["goalie_factor"].mean().to_dict()

    for r in roster.itertuples(index=False):
        player,team,position=r.player,r.team,r.position
        df_p=grouped.get(player.lower(),pd.DataFrame())
        if df_p.empty: continue

        sog_vals=df_p.groupby(game_col)["sog"].sum().tolist()
        if not sog_vals: continue

        l3,l5,l10=np.mean(sog_vals[-3:]),np.mean(sog_vals[-5:]),np.mean(sog_vals[-10:])
        baseline=0.55*l10+0.3*l5+0.15*l3

        opp=team_b if team==team_a else team_a
        goalie_factor=goalie_adj.get(opp,1.0)
        pos_factor=pos_matchup_adj.get((opp,position),1.0)

        lam=baseline*(1+(goalie_factor-1)*0.2)*pos_factor

        results.append({
            "Player":player,
            "Team":team,
            "Position":position,
            "Final Projection":round(lam,2),
            "Pos Adj":round(pos_factor,2),
            "Season Avg":round(np.mean(sog_vals),2)
        })

    return pd.DataFrame(results)
# ---------------------------------------------------------------
# Run Model + Combine Games
# ---------------------------------------------------------------
if run_model:
    all_tables=[]
    for m in games:
        team_a,team_b=m["away"],m["home"]
        df=build_model(
            team_a,team_b,
            skaters_df,shots_df,goalies_df,
            lines_df,teams_df,injuries_df
        )
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
            with st.form(f"form_{i}"):
                st.markdown(f"""
                <div style="background-color:{btn_color};border:{border};
                            border-radius:8px 8px 0 0;
                            color:#fff;font-weight:600;font-size:15px;
                            padding:10px 14px;
                            box-shadow:{glow};
                            display:flex;align-items:center;
                            justify-content:center;gap:6px;">
                    <img src="{m['away_logo']}" height="22">
                    <span>{m['away']}</span>
                    <span style="color:#D6D6D6;">@</span>
                    <span>{m['home']}</span>
                    <img src="{m['home_logo']}" height="22">
                </div>
                """, unsafe_allow_html=True)

                clicked = st.form_submit_button(
                    "Click to view",
                    use_container_width=True,
                    type="secondary"
                )

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

    # Formatting helpers
    def color_form(v):
        if "Above" in str(v): return "<span style='color:#00FF00;font-weight:bold;'>🟢</span>"
        if "Below" in str(v): return "<span style='color:#FF4B4B;font-weight:bold;'>🔴</span>"
        return "<span style='color:#D6D6D6;'>⚪</span>"

    if "line_test_val" in st.session_state:
        test_line=st.session_state.line_test_val
        df["Prob ≥ Line (%)"]=df["Final Projection"].apply(
            lambda lam: round((1-poisson.cdf(test_line-1,mu=max(lam,0.01)))*100,1)
        )

        def safe_odds(p):
            p=np.clip(p,0.1,99.9)
            if p>=50:
                odds_val=-100*((p/100)/(1-p/100))
            else:
                odds_val=100*((1-p/100)/(p/100))
            return f"{'+' if odds_val>0 else ''}{int(round(odds_val))}"

        df["Playable Odds"]=df["Prob ≥ Line (%)"].apply(safe_odds)

    df=df.sort_values(["Team","Final Projection"],ascending=[True,False])

    cols=[
        "Player","Team","Position","Matchup",
        "Final Projection","Pos Adj",
        "Prob ≥ Line (%)","Playable Odds",
        "Season Avg"
    ]
    existing=[c for c in cols if c in df.columns]

    html_table=df[existing].to_html(index=False,escape=False)
    csv=df[existing].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="💾 Download Results (CSV)",
        data=csv,
        file_name="puck_shotz_results.csv",
        mime="text/csv"
    )

    components.html(f"""
    <style>
    table {{
        width:100%;
        border-collapse:collapse;
        font-family:'Source Sans Pro',sans-serif;
        color:#D6D6D6;
    }}
    th {{
        background-color:#0A3A67;
        color:#FFFFFF;
        padding:6px;
        text-align:center;
        position:sticky;
        top:0;
    }}
    td {{
        background-color:#0F2743;
        padding:4px;
        text-align:center;
    }}
    tr:nth-child(even) td {{
        background-color:#142F52;
    }}
    </style>
    <div style='overflow-x:auto;height:650px;'>
        {html_table}
    </div>
    """,height=700,scrolling=True)
