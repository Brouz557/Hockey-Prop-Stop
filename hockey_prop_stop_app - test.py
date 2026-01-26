# ---------------------------------------------------------------
# 🏒 PUCK SHOTZ HOCKEY ANALYTICS — v1.0 (LOCKED)
# ESPN + MoneyPuck | Full Feature Parity | Calibrated
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests, html, json
from scipy.stats import poisson
import streamlit.components.v1 as components

# ===============================================================
# MODEL VERSIONING (LOCKED)
# ===============================================================
MODEL_VERSION = "v1.0.0"

MODEL_CONFIG = {
    "weights": {"L10": 0.55, "L5": 0.30, "L3": 0.15},
    "goalie_strength": 0.20,
    "line_clip": (0.7, 1.3),
    "pace_clip": (0.85, 1.15),
    "penalty_clip": (0.8, 1.25),
    "lambda_alpha": 0.97   # ← from backtest calibration
}

# ===============================================================
# STREAMLIT SETUP
# ===============================================================
st.set_page_config(
    page_title="Puck Shotz Hockey Analytics",
    layout="wide",
    page_icon="🏒"
)

st.warning(f"Production Model — {MODEL_VERSION}")

# ===============================================================
# CONSTANTS / ENDPOINTS
# ===============================================================
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
MP_PLAYER_GAMES = "https://api.moneypuck.com/v2/playerGameStats"
MP_GOALIES = "https://api.moneypuck.com/v2/goalieStats"
MP_SHIFTS = "https://api.moneypuck.com/v2/shiftData"

TEAM_ABBREV_MAP = {
    "NJ": "NJD", "LA": "LAK", "SJ": "SJS", "TB": "TBL"
}

# ===============================================================
# HEADER
# ===============================================================
st.markdown("""
<div style='text-align:center; background-color:#0A3A67; padding:15px; border-radius:6px;'>
  <img src='https://raw.githubusercontent.com/Brouz557/Hockey-Prop-Stop/694ae2a448204908099ce2899bd479052d01b518/modern%20hockey%20puck%20l.png' width='220'>
</div>
<h1 style='text-align:center;color:#1E5A99;'>Puck Shotz Hockey Analytics</h1>
<p style='text-align:center;color:#D6D6D6;'>
ESPN Matchups • MoneyPuck Data • v1.0 Locked Model
</p>
""", unsafe_allow_html=True)

# ===============================================================
# ESPN — TODAY’S GAMES
# ===============================================================
@st.cache_data(ttl=300)
def get_todays_games():
    r = requests.get(ESPN_SCOREBOARD, timeout=15)
    r.raise_for_status()
    data = r.json()

    games = []
    for e in data.get("events", []):
        c = e["competitions"][0]["competitors"]
        away, home = c[0], c[1]
        games.append({
            "away": TEAM_ABBREV_MAP.get(
                away["team"]["abbreviation"], away["team"]["abbreviation"]
            ),
            "home": TEAM_ABBREV_MAP.get(
                home["team"]["abbreviation"], home["team"]["abbreviation"]
            ),
            "away_logo": away["team"]["logo"],
            "home_logo": home["team"]["logo"]
        })
    return games

games = get_todays_games()
if not games:
    st.warning("No NHL games today.")
    st.stop()

# ===============================================================
# MONEYPUCK — PLAYER GAME DATA (EV + PP)
# ===============================================================
@st.cache_data(ttl=3600)
def load_player_games():
    r = requests.get(MP_PLAYER_GAMES, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())

    df = df.rename(columns={
        "gameId": "game_id",
        "gameDate": "game_date",
        "playerId": "player_id",
        "playerName": "player",
        "team": "team",
        "shotsOnGoal": "sog",
        "goals": "goal",
        "timeOnIce": "toi",
        "powerPlayShotsOnGoal": "pp_sog",
        "powerPlayTimeOnIce": "pp_toi"
    })

    for c in ["sog","goal","toi","pp_sog","pp_toi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ev_sog"] = df["sog"] - df["pp_sog"]
    df["ev_toi"] = df["toi"] - df["pp_toi"]

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["team"] = df["team"].str.upper()
    df["player"] = df["player"].str.strip()

    return df[[
        "game_id","game_date","player_id","player","team",
        "sog","goal","toi","pp_sog","pp_toi","ev_sog","ev_toi"
    ]]

# ===============================================================
# MONEYPUCK — GOALIE SUPPRESSION
# ===============================================================
@st.cache_data(ttl=3600)
def load_goalie_adj():
    r = requests.get(MP_GOALIES, timeout=30)
    r.raise_for_status()
    g = pd.DataFrame(r.json())

    g["shots_per_game"] = g["shotsAgainst"] / g["gamesPlayed"]
    league_avg = g["shots_per_game"].mean()
    g["goalie_factor"] = (g["shots_per_game"] / league_avg).clip(0.7, 1.3)

    return g.groupby("team")["goalie_factor"].mean().to_dict()

# ===============================================================
# MONEYPUCK — LINE SUPPRESSION (SHIFT DATA)
# ===============================================================
@st.cache_data(ttl=3600)
def load_line_adj():
    r = requests.get(MP_SHIFTS, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())

    df = df.rename(columns={
        "team": "team",
        "onIceShotsAgainst": "sog_against",
        "timeOnIce": "toi"
    })

    df["toi"] = pd.to_numeric(df["toi"], errors="coerce").fillna(0)
    df["sog_against"] = pd.to_numeric(df["sog_against"], errors="coerce").fillna(0)

    df["sog60"] = np.where(df["toi"] > 0, df["sog_against"] / df["toi"] * 60, np.nan)

    team_def = df.groupby("team")["sog60"].mean().reset_index()
    league_avg = team_def["sog60"].mean()
    team_def["line_factor"] = (league_avg / team_def["sog60"]).clip(
        *MODEL_CONFIG["line_clip"]
    )

    return team_def.set_index("team")["line_factor"].to_dict()

# ===============================================================
# PACE & PENALTY CONTEXT
# ===============================================================
def build_team_pace(shots):
    g = shots.groupby(["team","game_id"])["sog"].sum().reset_index()
    pace = g.groupby("team")["sog"].mean()
    league = pace.mean()
    return (pace / league).clip(*MODEL_CONFIG["pace_clip"]).to_dict()

def build_penalty_adj(shots):
    g = shots.groupby(["team","game_id"])["pp_toi"].sum().reset_index()
    pp = g.groupby("team")["pp_toi"].mean()
    league = pp.mean()
    return (pp / league).clip(*MODEL_CONFIG["penalty_clip"]).to_dict()

# ===============================================================
# MODEL CORE — v1.0 LOCKED
# ===============================================================
def american_odds(p):
    p = np.clip(p, 0.001, 0.999)
    return int(-100*(p/(1-p))) if p >= 0.5 else int(100*((1-p)/p))

def project_lambda_v1(r, opp, adj):
    ev_lam = r.EV_L10_rate * r.EV_L10_toi
    pp_lam = r.PP_L10_rate * (r.PP_L10_toi * adj["penalty"].get(opp, 1.0))

    lam = ev_lam + pp_lam
    lam *= adj["pace"]
    lam *= adj["line"]
    lam *= (1 + MODEL_CONFIG["goalie_strength"] * (adj["goalie"] - 1))
    lam *= MODEL_CONFIG["lambda_alpha"]

    return max(lam, 0.01)

# ===============================================================
# RUN CONTROLS
# ===============================================================
col_run, col_line = st.columns([3,1])
with col_run:
    run = st.button("🚀 Run Model (All Games)")
with col_line:
    test_line = st.number_input("Line to Test", 0.5, 10.0, 3.5, 0.5)

# ===============================================================
# RUN MODEL
# ===============================================================
if run:
    with st.spinner("Loading MoneyPuck data…"):
        shots = load_player_games()
        goalie_adj = load_goalie_adj()
        line_adj = load_line_adj()

    pace_adj = build_team_pace(shots)
    penalty_adj = build_penalty_adj(shots)

    shots = shots.sort_values("game_date")
    g = shots.groupby("player_id", group_keys=False)

    shots["EV_L10_rate"] = g["ev_sog"].rolling(10).sum() / g["ev_toi"].rolling(10).sum()
    shots["PP_L10_rate"] = g["pp_sog"].rolling(10).sum() / g["pp_toi"].rolling(10).sum()
    shots["EV_L10_toi"] = g["ev_toi"].rolling(10).mean()
    shots["PP_L10_toi"] = g["pp_toi"].rolling(10).mean()

    shots = shots.dropna(subset=["EV_L10_rate","PP_L10_rate"])

    results = []

    for m in games:
        teams = {m["away"], m["home"]}
        df = shots[shots["team"].isin(teams)]
        latest = df.groupby("player_id").tail(1)

        for _, r in latest.iterrows():
            opp = next(t for t in teams if t != r.team)

            lam = project_lambda_v1(
                r,
                opp,
                adj={
                    "pace": np.sqrt(pace_adj.get(r.team,1.0)*pace_adj.get(opp,1.0)),
                    "line": line_adj.get(opp,1.0),
                    "goalie": goalie_adj.get(opp,1.0),
                    "penalty": penalty_adj
                }
            )

            p = 1 - poisson.cdf(test_line-1, mu=lam)

            results.append({
                "Player": r.player,
                "Team": r.team,
                "Final Projection": round(lam,2),
                "Prob ≥ Line (%)": round(p*100,1),
                "Playable Odds": f"{'+' if american_odds(p)>0 else ''}{american_odds(p)}",
                "Model Version": MODEL_VERSION
            })

    out = pd.DataFrame(results).sort_values(
        ["Team","Final Projection"], ascending=[True,False]
    )

    st.success("✅ Model Complete")

    components.html(
        f"<div style='height:650px;overflow:auto'>{out.to_html(index=False)}</div>",
        height=700
    )
