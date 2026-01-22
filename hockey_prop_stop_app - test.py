# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Test Mode (Instant Filter + Logos + Injuries)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

st.set_page_config(page_title="Puck Shotz Hockey Analytics (Test)", layout="wide", page_icon="🏒")
st.warning("🧪 TEST MODE — Sandbox version. Changes here won’t affect your main app.")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px; margin-bottom:10px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>Automatically runs all of today’s NHL matchups with inline logos and instant team filters.</p>
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
        if "player" in injuries.columns:
            injuries["player"]=injuries["player"].astype(str).str.strip().str.lower()
    return skaters,shots,goalies,lines,teams,injuries

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df = load_all(
    skaters_file, shots_file, goalies_file, lines_file, teams_file, injuries_file)
if skaters_df.empty or shots_df.empty:
    st.warning("⚠️ Missing data. Upload required files.")
    st.stop()
st.success("✅ Data loaded successfully.")

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    if not df.empty: df.columns=df.columns.str.lower().str.strip()
team_col=next((c for c in skaters_df.columns if "team" in c),None)
player_col="name" if "name" in skaters_df.columns else None
shots_df=shots_df.rename(columns={next((c for c in shots_df.columns if "player" in c or "name" in c),"player"):"player"})
shots_df["player"]=shots_df["player"].astype(str).str.strip()
game_col=next((c for c in shots_df.columns if "game" in c and "id" in c),None)

# ---------------------------------------------------------------
# Matchup Pull (ESPN)
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_todays_games():
    url="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    r=requests.get(url,timeout=10)
    data=r.json()
    games=[]
    for e in data.get("events",[]):
        comps=e.get("competitions",[{}])[0].get("competitors",[])
        if len(comps)==2:
            away,home=comps[0],comps[1]
            games.append({
                "away":away["team"]["abbreviation"],
                "home":home["team"]["abbreviation"],
                "away_logo":away["team"]["logo"],
                "home_logo":home["team"]["logo"]
            })
    return games

games=get_todays_games()
if not games:
    st.warning("No games found today.")
    st.stop()

# ---------------------------------------------------------------
# Run Button / Line Input
# ---------------------------------------------------------------
col_run,col_line=st.columns([3,1])
with col_run: run_model=st.button("🚀 Run Model (All Games)")
with col_line:
    line_test=st.number_input("Line to Test (Probability Update)",0.0,10.0,3.5,0.5,key="line_test")
    if "line_test_val" not in st.session_state:
        st.session_state.line_test_val=line_test
    elif st.session_state.line_test_val!=line_test:
        st.session_state.line_test_val=line_test
        if "results" in st.session_state:
            st.rerun()

# ---------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_model(team_a, team_b, skaters_df, shots_df, goalies_df, lines_df, teams_df, injuries_df):
    results = []
    skaters = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
    roster = skaters[[player_col, team_col]].rename(columns={player_col:"player", team_col:"team"}).drop_duplicates("player")
    grouped = {n.lower(): g for n, g in shots_df.groupby(shots_df["player"].str.lower())}

    line_adj = {}
    if not lines_df.empty and "line pairings" in lines_df.columns:
        l = lines_df.copy()
        l["games"] = pd.to_numeric(l["games"], errors="coerce").fillna(0)
        l["sog against"] = pd.to_numeric(l["sog against"], errors="coerce").fillna(0)
        l = l.groupby(["line pairings","team"], as_index=False).agg({"games":"sum","sog against":"sum"})
        l["sog_against_per_game"] = np.where(l["games"] > 0, l["sog against"]/l["games"], np.nan)
        team_avg = l.groupby("team")["sog_against_per_game"].mean()
        league_avg = team_avg.mean()
        l["line_factor"] = (league_avg / l["sog_against_per_game"]).clip(0.7, 1.3)
        line_adj = l.copy()

    goalie_adj = {}
    if not goalies_df.empty and {"team","shots against","games"}.issubset(goalies_df.columns):
        g = goalies_df.copy()
        g["shots against"] = pd.to_numeric(g["shots against"], errors="coerce").fillna(0)
        g["games"] = pd.to_numeric(g["games"], errors="coerce").fillna(1)
        g["shots_per_game"] = g["shots against"] / g["games"]
        league_avg_sa = g["shots_per_game"].mean()
        g["goalie_factor"] = (g["shots_per_game"] / league_avg_sa).clip(0.7, 1.3)
        goalie_adj = g.groupby("team")["goalie_factor"].mean().to_dict()

    for row in roster.itertuples(index=False):
        player, team = row.player, row.team
        df_p = grouped.get(player.lower(), pd.DataFrame())
        if df_p.empty: continue
        sog_vals = df_p.groupby(game_col)["sog"].sum().tolist()
        if not sog_vals: continue

        last3 = sog_vals[-3:] if len(sog_vals) >= 3 else sog_vals
        last5 = sog_vals[-5:] if len(sog_vals) >= 5 else sog_vals
        last10 = sog_vals[-10:] if len(sog_vals) >= 10 else sog_vals
        l3, l5, l10 = np.mean(last3), np.mean(last5), np.mean(last10)
        baseline = (0.55 * l10) + (0.3 * l5) + (0.15 * l3)
        trend = (l5 - l10) / l10 if l10 > 0 else 0

        if trend > 0.05:
            form_flag = "🟢 Above Baseline"
        elif trend < -0.05:
            form_flag = "🔴 Below Baseline"
        else:
            form_flag = "⚪ Neutral"

        # --- Injury detection and tooltip ---
        injury_html = ""
        if not injuries_df.empty and {"player","team"}.issubset(injuries_df.columns):
            player_lower = player.lower().strip()
            last_name = player_lower.split()[-1]
            team_lower = team.lower().strip()
            match = injuries_df[
                injuries_df["team"].str.lower().str.strip().eq(team_lower)
                & injuries_df["player"].str.lower().str.endswith(last_name)
            ]
            if not match.empty:
                note = str(match.iloc[0].get("injury note","")).strip()
                injury_type = str(match.iloc[0].get("injury type","")).strip()
                date_injury = str(match.iloc[0].get("date of injury","")).strip()
                tooltip = "\n".join([p for p in [injury_type,note,date_injury] if p]) or "Injury info unavailable"
                safe = html.escape(tooltip)
                injury_html = f"<span style='cursor:pointer;' onclick='alert({json.dumps(safe)})' title='Tap or click for injury info'>🚑</span>"

        line_factor_internal = 1.0
        if isinstance(line_adj, pd.DataFrame) and not line_adj.empty:
            last_name = str(player).split()[-1].lower()
            m = line_adj[line_adj["line pairings"].str.contains(last_name, case=False, na=False)]
            if not m.empty:
                line_factor_internal = np.average(m["line_factor"], weights=m["games"])

        opp_team = team_b if team == team_a else team_a
        goalie_factor = goalie_adj.get(opp_team, 1.0)
        lam = baseline * (1 + (goalie_factor - 1.0) * 0.2) * line_factor_internal
        poisson_prob = float(np.clip(1 - poisson.cdf(np.floor(lam) - 1, mu=max(lam, 0.01)), 0.0001, 0.9999))
        odds = -100 * (poisson_prob / (1 - poisson_prob)) if poisson_prob >= 0.5 else 100 * ((1 - poisson_prob) / poisson_prob)
        odds = float(np.clip(odds, -10000, 10000))
        playable_odds = f"{'+' if odds > 0 else ''}{int(odds)}"

        results.append({
            "Player": player, "Team": team, "Injury": injury_html,
            "Trend Score": round(trend,3),
            "Final Projection": round(lam,2),
            "Prob ≥ Projection (%) L5": round(poisson_prob*100,1),
            "Playable Odds": playable_odds,
            "Season Avg": round(np.mean(sog_vals),2),
            "Line Adj": round(line_factor_internal,2),
            "Form Indicator": form_flag,
            "L3 Shots": ", ".join(map(str,last3)),
            "L5 Shots": ", ".join(map(str,last5)),
            "L10 Shots": ", ".join(map(str,last10))
        })
    return pd.DataFrame(results)

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
        st.warning("No valid data generated.")

# ---------------------------------------------------------------
# Display Buttons + Table
# ---------------------------------------------------------------
if "results" in st.session_state:
    df=st.session_state.results.copy()
    games=st.session_state.matchups

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
                        st.session_state.selected_match = None
                        st.session_state.selected_teams = None
                    else:
                        st.session_state.selected_match = match_id
                        st.session_state.selected_teams = {team_a, team_b}
                    st.rerun()

    sel_teams = st.session_state.get("selected_teams")
    if sel_teams:
        df = df[df["Team"].isin(sel_teams)]
        st.markdown(f"### Showing results for: **{' vs '.join(sel_teams)}**")
    else:
        st.markdown("### Showing results for: **All Teams**")

    # Colorize trend + form
    def color_trend(v):
        if v > 0.05:
            return "<span style='color:#00FF00;font-weight:bold;'>▲</span>"
        elif v < -0.05:
            return "<span style='color:#FF4B4B;font-weight:bold;'>▼</span>"
        else:
            return "<span style='color:#D6D6D6;'>–</span>"

    def color_form(v):
        if "Above" in v:
            return "<span style='color:#00FF00;font-weight:bold;'>🟢 Above Baseline</span>"
        elif "Below" in v:
            return "<span style='color:#FF4B4B;font-weight:bold;'>🔴 Below Baseline</span>"
        else:
            return "<span style='color:#D6D6D6;'>⚪ Neutral</span>"

    df["Trend"] = df["Trend Score"].apply(color_trend)
    df["Form Indicator"] = df["Form Indicator"].apply(color_form)

    if "line_test_val" in st.session_state:
        test_line = st.session_state.line_test_val
        df["Prob ≥ Line (%)"] = df["Final Projection"].apply(
            lambda lam: round((1 - poisson.cdf(test_line - 1, mu=max(lam, 0.01))) * 100, 1)
        )
        def safe_odds(p):
            p = np.clip(p, 0.1, 99.9)
            if p >= 50:
                odds_val = -100 * ((p/100) / (1 - p/100))
            else:
                odds_val = 100 * ((1 - p/100) / (p/100))
            return f"{'+' if odds_val > 0 else ''}{int(round(odds_val))}"
        df["Playable Odds"] = df["Prob ≥ Line (%)"].apply(safe_odds)

    df = df.sort_values(["Team","Final Projection","Line Adj"],ascending=[True,False,False])

    # ✅ Safe column handling to prevent KeyError
    cols = [
        "Player","Team","Injury","Trend","Final Projection","Prob ≥ Line (%)",
        "Playable Odds","Season Avg","Line Adj","Form Indicator",
        "L3 Shots","L5 Shots","L10 Shots"
    ]
    existing_cols = [c for c in cols if c in df.columns]
    html_table = df[existing_cols].to_html(index=False, escape=False)

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
    </style>
    <div style='overflow-x:auto;height:650px;'>{html_table}</div>
    """,height=700,scrolling=True)
    
    # ---------------------------------------------------------------
# 🎯 Parlay Probability Calculator
# ---------------------------------------------------------------
st.markdown("### 🎯 Parlay Probability Calculator")
st.caption("Select multiple players from the table above to estimate the combined probability and fair odds of all selected players hitting their shot lines. This is a statistical combination only (not a bet).")

# Ensure 'Prob ≥ Line (%)' exists in dataframe
if "Prob ≥ Line (%)" in df.columns:
    # Multiselect player picker
    selected_players = st.multiselect(
        "Select Players for Parlay",
        options=df["Player"].unique(),
        help="Choose players to include in your parlay"
    )

    # Correlation adjustment slider
    corr_adj = st.slider(
        "Correlation Adjustment (%)",
        min_value=-10,
        max_value=10,
        value=0,
        step=1,
        help="Adjust for overlap between legs (e.g. teammates or related outcomes)"
    )

    if selected_players:
        subset = df[df["Player"].isin(selected_players)][["Player", "Prob ≥ Line (%)"]].copy()
        subset["Prob ≥ Line (%)"] = pd.to_numeric(subset["Prob ≥ Line (%)"], errors="coerce").fillna(0)
        probs = subset["Prob ≥ Line (%)"].to_numpy() / 100.0

        # Apply correlation adjustment
        adjustment_factor = 1 + (corr_adj / 100)
        combined_prob = np.prod(probs) * adjustment_factor
        combined_prob = np.clip(combined_prob, 0, 1)

        # Convert to American odds
        if combined_prob >= 0.5:
            american_odds = -100 * (combined_prob / (1 - combined_prob))
        else:
            american_odds = 100 * ((1 - combined_prob) / combined_prob)

        st.markdown(f"**Selected Legs:** {len(selected_players)}")
        st.markdown(f"**Combined Probability:** {combined_prob*100:.2f}%")

        odds_str = f\"{int(round(american_odds, -1)):+}\"
        st.markdown(f"**Fair Odds:** {odds_str}")

        # Optional table of legs
        st.dataframe(subset.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Select one or more players to calculate a parlay probability.")
else:
    st.warning("⚠️ Probability column not found. Run the model first to enable parlay calculations.")


