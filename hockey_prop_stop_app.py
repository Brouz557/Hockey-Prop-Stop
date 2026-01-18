# ---------------------------------------------------------------
# 🏒 Hockey Prop Stop — Goal Projections + Calibration
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os, contextlib, io, altair as alt

# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")
st.markdown(
    """
    <h1 style='text-align:center;color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center;color:#BFC0C0;'>
        Team-vs-Team matchup analytics with improved goal projections
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# CSS
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
    .stDataFrame { overflow-x: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar Uploaders
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Data Files (.xlsx or .csv)")
skaters_file = st.sidebar.file_uploader("Skaters", type=["xlsx","csv"])
shots_file   = st.sidebar.file_uploader("SHOT DATA", type=["xlsx","csv"])
goalies_file = st.sidebar.file_uploader("GOALTENDERS", type=["xlsx","csv"])
lines_file   = st.sidebar.file_uploader("LINE DATA", type=["xlsx","csv"])
teams_file   = st.sidebar.file_uploader("TEAMS", type=["xlsx","csv"])

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
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
    if file_uploader is not None: return load_file(file_uploader)
    return safe_read(default_path)

# ---------------------------------------------------------------
# Cached load
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data(skaters_file, shots_file, goalies_file, lines_file, teams_file):
    base_paths = [".","data","/mount/src/hockey-prop-stop/data"]
    def find_file(filename):
        for p in base_paths:
            full=os.path.join(p,filename)
            if os.path.exists(full): return full
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        skaters=load_data(skaters_file,find_file("Skaters.xlsx")or"Skaters.xlsx")
        shots=load_data(shots_file,find_file("SHOT DATA.xlsx")or"SHOT DATA.xlsx")
        goalies=load_data(goalies_file,find_file("GOALTENDERS.xlsx")or"GOALTENDERS.xlsx")
        lines=load_data(lines_file,find_file("LINE DATA.xlsx")or"LINE DATA.xlsx")
        teams=load_data(teams_file,find_file("TEAMS.xlsx")or"TEAMS.xlsx")
    return skaters,shots,goalies,lines,teams

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all_data(
    skaters_file, shots_file, goalies_file, lines_file, teams_file
)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing required data. Please upload or verify repo files.")
    st.stop()
st.success("✅ Data loaded successfully.")

# ---------------------------------------------------------------
# Normalize columns
# ---------------------------------------------------------------
skaters_df.columns=skaters_df.columns.str.lower().str.strip()
shots_df.columns=shots_df.columns.str.lower().str.strip()
if not goalies_df.empty: goalies_df.columns=goalies_df.columns.str.lower().str.strip()
if not lines_df.empty: lines_df.columns=lines_df.columns.str.lower().str.strip()

team_col=next((c for c in skaters_df.columns if "team" in c),None)
player_col="name" if "name" in skaters_df.columns else None
sog_col=next((c for c in shots_df.columns if "sog" in c),None)
goal_col=next((c for c in shots_df.columns if "goal" in c),None)
game_col=next((c for c in shots_df.columns if "game" in c and "id" in c),None)
player_col_shots=next((c for c in shots_df.columns if "player" in c or "name" in c),None)

# ---------------------------------------------------------------
# Team Selection
# ---------------------------------------------------------------
all_teams=sorted(skaters_df[team_col].dropna().unique().tolist())
col1,col2=st.columns(2)
with col1: team_a=st.selectbox("Select Team A",all_teams)
with col2: team_b=st.selectbox("Select Team B",[t for t in all_teams if t!=team_a])
st.markdown("---")

if "results" not in st.session_state: st.session_state.results=None
run_model=st.button("🚀 Run Model")

# ---------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------
if run_model:
    st.info(f"Building model for matchup: **{team_a} vs {team_b}** ...")

    goalie_adj,rebound_rate={},{}
    if not goalies_df.empty:
        g=goalies_df.copy()
        g=g[g["situation"].str.lower()=="all"]
        g["games"]=pd.to_numeric(g["games"],errors="coerce").fillna(0)
        g["unblocked attempts"]=pd.to_numeric(g["unblocked attempts"],errors="coerce").fillna(0)
        g["rebounds"]=pd.to_numeric(g["rebounds"],errors="coerce").fillna(0)
        g["shots_allowed_per_game"]=np.where(g["games"]>0,g["unblocked attempts"]/g["games"],np.nan)
        g["rebound_rate"]=np.where(g["unblocked attempts"]>0,g["rebounds"]/g["unblocked attempts"],0)
        team_avg=g.groupby("team")["shots_allowed_per_game"].mean()
        league_avg=team_avg.mean()
        goalie_adj=(league_avg/team_avg).to_dict()
        rebound_rate=g.groupby("team")["rebound_rate"].mean().to_dict()

    line_adj={}
    if not lines_df.empty:
        l=lines_df.copy()
        l["games"]=pd.to_numeric(l["games"],errors="coerce").fillna(0)
        l["sog against"]=pd.to_numeric(l["sog against"],errors="coerce").fillna(0)
        l=(l.groupby(["line pairings","team"],as_index=False)
             .agg({"games":"sum","sog against":"sum"}))
        l["sog_against_per_game"]=np.where(l["games"]>0,l["sog against"]/l["games"],np.nan)
        team_avg=l.groupby("team")["sog_against_per_game"].mean()
        league_avg=team_avg.mean()
        l["line_factor"]=(league_avg/l["sog_against_per_game"]).clip(0.7,1.3)
        line_adj=l.copy()

    roster=(skaters_df[skaters_df[team_col].isin([team_a,team_b])]
            [[player_col,team_col]]
            .rename(columns={player_col:"player",team_col:"team"})
            .drop_duplicates("player").reset_index(drop=True))

    shots_df=shots_df.rename(columns={player_col_shots:"player",game_col:"gameid",sog_col:"sog",goal_col:"goal"})
    shots_df["player"]=shots_df["player"].astype(str).str.strip()
    roster["player"]=roster["player"].astype(str).str.strip()
    grouped={n.lower():g.sort_values("gameid") for n,g in shots_df.groupby(shots_df["player"].str.lower())}

    results=[]
    progress=st.progress(0)
    total=len(roster)
    league_avg_shooting=shots_df["goal"].sum()/shots_df["sog"].sum()

    for i,row in enumerate(roster.itertuples(index=False),1):
        player,team=row.player,row.team
        df_p=grouped.get(str(player).lower(),pd.DataFrame())
        if df_p.empty: progress.progress(i/total); continue
        game_sogs=df_p.groupby("gameid")[["sog","goal"]].sum().reset_index().sort_values("gameid")
        sog_values=game_sogs["sog"].tolist()
        l3,l5,l10=np.mean(sog_values[-3:]),np.mean(sog_values[-5:]),np.mean(sog_values[-10:])
        season_avg=np.mean(sog_values)
        trend=0 if pd.isna(l10) or l10==0 else (l3-l10)/l10
        base_proj=np.nansum([0.5*l3,0.3*l5,0.2*l10])
        opp=team_b if team==team_a else team_a
        goalie_factor=np.clip(goalie_adj.get(opp,1.0),0.7,1.3)
        rebound_factor=rebound_rate.get(opp,0.0)
        line_factor=1.0
        if not isinstance(line_adj,dict):
            last_name=str(player).split()[-1].lower()
            m=line_adj[line_adj["line pairings"].str.contains(last_name,case=False,na=False)]
            if not m.empty: line_factor=np.average(m["line_factor"],weights=m["games"])
            line_factor=np.clip(line_factor,0.7,1.3)
        adj_proj=base_proj*(0.7+0.3*goalie_factor)*(0.7+0.3*line_factor)
        adj_proj*=(1+rebound_factor*0.1)
        adj_proj=max(0,round(adj_proj,2))
        total_sogs=df_p["sog"].sum(); total_goals=df_p["goal"].sum()
        season_shoot_pct=(total_goals/total_sogs) if total_sogs>0 else 0
        recent_df=df_p.tail(10)
        recent_sogs=recent_df["sog"].sum(); recent_goals=recent_df["goal"].sum()
        recent_shoot_pct=(recent_goals/recent_sogs) if recent_sogs>0 else season_shoot_pct
        adj_shoot_pct=(0.7*recent_shoot_pct+0.2*season_shoot_pct+0.1*league_avg_shooting)
        adj_shoot_pct=np.clip(adj_shoot_pct,0.05,0.22)*(1/goalie_factor)
        proj_goals=adj_proj*adj_shoot_pct
        results.append({"Player":player,"Team":team,"Season Avg":round(season_avg,2),
                        "Trend Score":round(trend,3),"Base Projection":round(base_proj,2),
                        "Goalie Adj":round(goalie_factor,2),"Line Adj":round(line_factor,2),
                        "Final Projection":adj_proj,"Shooting %":round(adj_shoot_pct*100,1),
                        "Projected Goals":round(proj_goals,2)})
        progress.progress(i/total)
    progress.empty()
    df=pd.DataFrame(results)
    if not df.empty:
        df["Goal Signal"]=((df["Projected Goals"]/df["Projected Goals"].max())*0.6+
                            (df["Shooting %"]/df["Shooting %"].max())*0.4).fillna(0)
        df["Scoring Tier"]=pd.qcut(df["Goal Signal"].rank(method="first"),4,
                                   labels=["Low","Medium","High","Elite"])
    st.session_state.results=df

# ---------------------------------------------------------------
# Display Results + Calibration
# ---------------------------------------------------------------
if st.session_state.results is not None:
    df=st.session_state.results.copy()
    st.markdown("### 📊 Player Projections (Adjusted)")
    html_table=df.to_html(index=False,escape=False)
    st.markdown(f"<div style='overflow-x:auto'>{html_table}</div>",unsafe_allow_html=True)

    st.markdown("### 🔁 Interactive Table (Sortable)")
    display_cols=[c for c in ["Player","Final Projection","Projected Goals",
                              "Shooting %","Scoring Tier"] if c in df.columns]
    st.dataframe(df[display_cols],use_container_width=True,hide_index=True)

    # --- Scatter Plot ---
    st.markdown("### 🎯 Shot vs Goal Visualization")
    chart=alt.Chart(df).mark_circle(size=80).encode(
        x="Final Projection",y="Projected Goals",
        color=alt.Color("Shooting %",scale=alt.Scale(scheme="viridis")),
        tooltip=["Player","Team","Projected Goals","Shooting %","Scoring Tier"]
    ).interactive()
    st.altair_chart(chart,use_container_width=True)

    # --- Calibration Chart ---
    st.markdown("### 🧮 Model Calibration (Projected vs Actual Goals)")
    if "goal" in shots_df.columns and "sog" in shots_df.columns:
        calib_df=shots_df.groupby("player",as_index=False).agg({"goal":"sum","sog":"sum"})
        calib_df["actual_goal_rate"]=np.where(calib_df["sog"]>0,calib_df["goal"]/calib_df["sog"],0)
        merged=df[["Player","Projected Goals"]].rename(columns={"Player":"player"})
        merged=pd.merge(merged,calib_df,on="player",how="left")
        bin_count=6
        bins=np.linspace(0,max(0.6,merged["Projected Goals"].max()),bin_count+1)
        merged["goal_bin"]=pd.cut(merged["Projected Goals"],bins=bins,include_lowest=True)
        calib_summary=(merged.groupby("goal_bin",as_index=False)
                       .agg({"Projected Goals":"mean","actual_goal_rate":"mean","player":"count"})
                       .rename(columns={"player":"Player Count"}))
        calib_melt=calib_summary.melt(id_vars=["goal_bin","Player Count"],
                                      value_vars=["Projected Goals","actual_goal_rate"],
                                      var_name="Type",value_name="Value")
        chart_calib=(alt.Chart(calib_melt).mark_bar(opacity=0.8).encode(
            x=alt.X("goal_bin:N",title="Projected Goals Range"),
            y=alt.Y("Value:Q",title="Average Goals / Goal Rate"),
            color=alt.Color("Type:N",
                            scale=alt.Scale(domain=["Projected Goals","actual_goal_rate"],
                                            range=["#00B140","#BFC0C0"])),
            tooltip=["goal_bin","Player Count","Type","Value"]
        ).properties(height=400))
        st.altair_chart(chart_calib,use_container_width=True)
        st.caption("Green = Model projection | Gray = Actual scoring rate from data")
    else:
        st.info("Calibration chart unavailable (need 'goal' and 'sog' columns).")
