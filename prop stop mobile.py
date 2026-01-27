# ---------------------------------------------------------------
# 🏒 Puck Shotz Hockey Analytics — Mobile (CACHED + FULL PARITY)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os, requests
from scipy.stats import poisson
import streamlit.components.v1 as components

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Puck Shotz Hockey Analytics (Mobile)",
    layout="wide",
    page_icon="🏒"
)

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='text-align:center; background-color:#0A3A67;
            padding:14px; border-radius:8px; margin-bottom:12px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png'
       width='200'>
</div>
<h3 style='text-align:center;color:#1E5A99;margin-top:0;'>
    Puck Shotz Hockey Analytics
</h3>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Team Abbreviation Normalization
# ---------------------------------------------------------------
TEAM_ABBREV_MAP = {
    "NJ": "NJD",
    "LA": "LAK",
    "SJ": "SJS",
    "TB": "TBL"
}

# ---------------------------------------------------------------
# Auto-load data
# ---------------------------------------------------------------
def safe_read(path):
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def find_file(name):
    for p in [".", "data", "/mount/src/hockey-prop-stop/data"]:
        fp = os.path.join(p, name)
        if os.path.exists(fp):
            return fp
    return None

@st.cache_data(show_spinner=False)
def load_all():
    return (
        safe_read(find_file("Skaters.xlsx")),
        safe_read(find_file("SHOT DATA.xlsx")),
        safe_read(find_file("GOALTENDERS.xlsx")),
        safe_read(find_file("LINE DATA.xlsx")),
        safe_read(find_file("TEAMS.xlsx")),
    )

skaters_df, shots_df, goalies_df, lines_df, teams_df = load_all()
if skaters_df.empty or shots_df.empty:
    st.error("Missing required data files.")
    st.stop()

for df in [skaters_df, shots_df, goalies_df, lines_df, teams_df]:
    df.columns = df.columns.str.lower().str.strip()

team_col = next(c for c in skaters_df.columns if "team" in c)
player_col = "name" if "name" in skaters_df.columns else skaters_df.columns[0]

shots_player_col = next(c for c in shots_df.columns if "player" in c or "name" in c)
shots_df = shots_df.rename(columns={shots_player_col: "player"})
shots_df["player"] = shots_df["player"].astype(str).str.strip()
game_col = next(c for c in shots_df.columns if "game" in c)

# ---------------------------------------------------------------
# ESPN Matchups
# ---------------------------------------------------------------
@st.cache_data(ttl=300)
def get_games():
    url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    data = requests.get(url, timeout=10).json()
    games = []

    for e in data.get("events", []):
        comps = e.get("competitions", [{}])[0].get("competitors", [])
        if len(comps) == 2:
            a, h = comps
            games.append({
                "away": TEAM_ABBREV_MAP.get(a["team"]["abbreviation"], a["team"]["abbreviation"]),
                "home": TEAM_ABBREV_MAP.get(h["team"]["abbreviation"], h["team"]["abbreviation"]),
                "away_logo": a["team"]["logo"],
                "home_logo": h["team"]["logo"]
            })
    return games

games = get_games()
if not games:
    st.warning("No games found today.")
    st.stop()

# ---------------------------------------------------------------
# Line Test
# ---------------------------------------------------------------
st.session_state.setdefault("line_test_val", 3.5)

line_test = st.number_input(
    "Line to Test",
    min_value=0.0,
    max_value=10.0,
    step=0.5,
    value=st.session_state.line_test_val
)
st.session_state.line_test_val = line_test

# ---------------------------------------------------------------
# Run model
# ---------------------------------------------------------------
if st.button("Run Model (All Games)", use_container_width=True):
    results = []

    for g in games:
        team_a, team_b = g["away"], g["home"]
        roster = skaters_df[skaters_df[team_col].isin([team_a, team_b])]
        grouped = {n.lower(): d for n, d in shots_df.groupby(shots_df["player"].str.lower())}

        line_adj = pd.DataFrame()
        if not lines_df.empty and {"line pairings","team","games","sog against"}.issubset(lines_df.columns):
            l = lines_df.copy()
            l["games"] = pd.to_numeric(l["games"], errors="coerce").fillna(0)
            l["sog against"] = pd.to_numeric(l["sog against"], errors="coerce").fillna(0)
            l = l.groupby(["line pairings","team"], as_index=False).agg({"games":"sum","sog against":"sum"})
            l["sog_against_per_game"] = np.where(l["games"]>0, l["sog against"]/l["games"], np.nan)
            league_avg = l.groupby("team")["sog_against_per_game"].mean().mean()
            l["line_factor"] = (league_avg / l["sog_against_per_game"]).clip(0.7,1.3)
            line_adj = l

        for _, r in roster.iterrows():
            player = str(r[player_col])
            team = r[team_col]
            df_p = grouped.get(player.lower())
            if df_p is None or "sog" not in df_p.columns:
                continue

            sog_vals = df_p.groupby(game_col)["sog"].sum().tolist()
            if len(sog_vals) < 3:
                continue

            if "goal" in df_p.columns:
                agg = df_p.groupby(game_col).agg({"sog":"sum","goal":"sum"})
                total_shots = agg["sog"].sum()
                total_goals = agg["goal"].sum()
                shooting_pct = (total_goals / total_shots) if total_shots > 0 else np.nan
            else:
                shooting_pct = np.nan

            l3, l5, l10 = np.mean(sog_vals[-3:]), np.mean(sog_vals[-5:]), np.mean(sog_vals[-10:])
            baseline = 0.55*l10 + 0.3*l5 + 0.15*l3

            trend = (l5 - l10) / l10 if l10 > 0 else 0
            form = "Above Baseline" if trend > 0.05 else "Below Baseline" if trend < -0.05 else "Neutral"

            line_factor = 1.0
            if not line_adj.empty:
                last = player.split()[-1].lower()
                m = line_adj[line_adj["line pairings"].str.contains(last, case=False, na=False)]
                if not m.empty:
                    line_factor = np.average(m["line_factor"], weights=m["games"])

            lam = baseline * line_factor

            results.append({
                "Player": player,
                "Team": team,
                "Matchup": f"{team_a}@{team_b}",
                "Final Projection": round(lam,2),
                "Line Adj": round(line_factor,2),
                "Form": form,
                "Season Avg": round(np.mean(sog_vals),2),
                "Shooting %": round(shooting_pct*100,2) if not np.isnan(shooting_pct) else "",
                "L3": ", ".join(map(str, sog_vals[-3:])),
                "L5": ", ".join(map(str, sog_vals[-5:])),
                "L10": ", ".join(map(str, sog_vals[-10:])),
            })

    st.session_state.base_results = pd.DataFrame(results)
    st.success("Model built and cached")

# ---------------------------------------------------------------
# Apply line test
# ---------------------------------------------------------------
def apply_line_test(df, line):
    df = df.copy()
    probs, odds = [], []

    for lam in df["Final Projection"]:
        p = 1 - poisson.cdf(line - 1, mu=max(lam, 0.01))
        p = np.clip(p, 0.0001, 0.9999)
        probs.append(round(p*100,1))
        o = -100*(p/(1-p)) if p>=0.5 else 100*((1-p)/p)
        odds.append(f"{'+' if o>0 else ''}{int(round(o))}")

    df["Prob ≥ Line (%)"] = probs
    df["Playable Odds"] = odds
    return df

# ---------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------
if "base_results" in st.session_state:
    df = apply_line_test(st.session_state.base_results, st.session_state.line_test_val)

    if "selected_match" not in st.session_state:
        st.session_state.selected_match = f"{games[0]['away']}@{games[0]['home']}"

    st.markdown("## Matchups")
    cols = st.columns(3)

    for i, g in enumerate(games):
        with cols[i % 3]:
            if st.button(f"{g['away']} @ {g['home']}", use_container_width=True):
                st.session_state.selected_match = f"{g['away']}@{g['home']}"

    team_a, team_b = st.session_state.selected_match.split("@")
    tabs = st.tabs([team_a, team_b])

    def team_logo(team):
        for g in games:
            if g["away"] == team:
                return g["away_logo"]
            if g["home"] == team:
                return g["home_logo"]
        return ""

    def render(team, tab):
        with tab:
            tdf = (
                df[(df["Team"]==team)&(df["Matchup"]==st.session_state.selected_match)]
                .drop_duplicates("Player")
                .sort_values("Final Projection", ascending=False)
            )

            for _, r in tdf.iterrows():
                components.html(
                    f"""
                    <div style="background:#0F2743;border:1px solid #1E5A99;
                                border-radius:16px;padding:16px;margin-bottom:16px;color:#fff;">
                        <div style="display:flex;align-items:center;margin-bottom:6px;">
                            <img src="{team_logo(team)}" style="width:32px;height:32px;margin-right:10px;">
                            <b>{r['Player']} – {r['Team']}</b>
                        </div>
                        <div>Final Projection: <b>{r['Final Projection']}</b></div>
                        <div>Line Adj: <b>{r['Line Adj']}</b></div>
                        <div>Form: <b>{r['Form']}</b></div>
                        <div>Prob ≥ Line: <b>{r['Prob ≥ Line (%)']}%</b></div>
                        <div>Playable Odds: <b>{r['Playable Odds']}</b></div>
                        <div>Season Avg: {r['Season Avg']}</div>
                        <div>Shooting %: <b>{r['Shooting %']}%</b></div>
                        <div>L3: {r['L3']}</div>
                        <div>L5: {r['L5']}</div>
                        <div>L10: {r['L10']}</div>
                    </div>
                    """,
                    height=340
                )

    render(team_a, tabs[0])
    render(team_b, tabs[1])
