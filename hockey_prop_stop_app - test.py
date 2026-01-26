# ---------------------------------------------------------------
# 🏒 PUCK SHOTZ HOCKEY ANALYTICS — v1.0 (SAFE / CLOUD-READY)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import streamlit.components.v1 as components

# ===============================================================
# MODEL VERSIONING (LOCKED)
# ===============================================================
MODEL_VERSION = "v1.0.1"

MODEL_CONFIG = {
    "goalie_strength": 0.20,
    "line_clip": (0.7, 1.3),
    "pace_clip": (0.85, 1.15),
    "penalty_clip": (0.8, 1.25),
    "lambda_alpha": 0.97
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
# ENDPOINTS
# ===============================================================
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
MP_PLAYER_GAMES = "https://api.moneypuck.com/v2/playerGameStats"
MP_GOALIES = "https://api.moneypuck.com/v2/goalieStats"
MP_SHIFTS = "https://api.moneypuck.com/v2/shiftData"

# ===============================================================
# SAFE REQUEST WRAPPER  ⭐⭐⭐ FIX
# ===============================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (PuckShotz/1.0; +https://github.com)"
}

def safe_get(url, timeout=30):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch data from: {url}")
        st.stop()

# ===============================================================
# ESPN — TODAY’S GAMES
# ===============================================================
@st.cache_data(ttl=300)
def get_todays_games():
    data = safe_get(ESPN_SCOREBOARD)
    games = []

    for e in data.get("events", []):
        c = e["competitions"][0]["competitors"]
        away, home = c[0], c[1]
        games.append({
            "away": away["team"]["abbreviation"],
            "home": home["team"]["abbreviation"],
            "away_logo": away["team"]["logo"],
            "home_logo": home["team"]["logo"]
        })
    return games

games = get_todays_games()
if not games:
    st.warning("No NHL games today.")
    st.stop()

# ===============================================================
# MONEYPUCK — PLAYER GAME DATA  ⭐⭐⭐ FIX
# ===============================================================
@st.cache_data(ttl=3600)
def load_player_games():
    data = safe_get(MP_PLAYER_GAMES)
    df = pd.DataFrame(data)

    df = df.rename(columns={
        "gameId": "game_id",
        "gameDate": "game_date",
        "playerId": "player_id",
        "playerName": "player",
        "team": "team",
        "shotsOnGoal": "sog",
        "timeOnIce": "toi",
        "powerPlayShotsOnGoal": "pp_sog",
        "powerPlayTimeOnIce": "pp_toi"
    })

    for c in ["sog", "toi", "pp_sog", "pp_toi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ev_sog"] = df["sog"] - df["pp_sog"]
    df["ev_toi"] = df["toi"] - df["pp_toi"]

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["team"] = df["team"].str.upper()
    df["player"] = df["player"].str.strip()

    return df

# ===============================================================
# GOALIE SUPPRESSION (SAFE FALLBACK)
# ===============================================================
@st.cache_data(ttl=3600)
def load_goalie_adj():
    try:
        data = safe_get(MP_GOALIES)
        g = pd.DataFrame(data)
        g["shots_per_game"] = g["shotsAgainst"] / g["gamesPlayed"]
        league = g["shots_per_game"].mean()
        g["goalie_factor"] = (g["shots_per_game"] / league).clip(0.7, 1.3)
        return g.groupby("team")["goalie_factor"].mean().to_dict()
    except:
        return {}

# ===============================================================
# LINE SUPPRESSION (SAFE FALLBACK)
# ===============================================================
@st.cache_data(ttl=3600)
def load_line_adj():
    try:
        data = safe_get(MP_SHIFTS)
        df = pd.DataFrame(data)
        df["toi"] = pd.to_numeric(df["timeOnIce"], errors="coerce").fillna(0)
        df["sog"] = pd.to_numeric(df["onIceShotsAgainst"], errors="coerce").fillna(0)
        df["sog60"] = np.where(df["toi"] > 0, df["sog"] / df["toi"] * 60, np.nan)

        team = df.groupby("team")["sog60"].mean()
        league = team.mean()
        return (league / team).clip(*MODEL_CONFIG["line_clip"]).to_dict()
    except:
        return {}

# ===============================================================
# MODEL CORE
# ===============================================================
def american_odds(p):
    p = np.clip(p, 0.001, 0.999)
    return int(-100*(p/(1-p))) if p >= 0.5 else int(100*((1-p)/p))

def project_lambda(r, opp, adj):
    ev = r.EV_L10_rate * r.EV_L10_toi
    pp = r.PP_L10_rate * r.PP_L10_toi
    lam = (ev + pp) * adj["pace"] * adj["line"]
    lam *= (1 + MODEL_CONFIG["goalie_strength"] * (adj["goalie"] - 1))
    lam *= MODEL_CONFIG["lambda_alpha"]
    return max(lam, 0.01)

# ===============================================================
# UI CONTROLS
# ===============================================================
col_run, col_line = st.columns([3,1])
with col_run:
    run = st.button("🚀 Run Model")
with col_line:
    test_line = st.number_input("Line", 0.5, 10.0, 3.5, 0.5)

# ===============================================================
# RUN MODEL
# ===============================================================
if run:
    shots = load_player_games()
    goalie_adj = load_goalie_adj()
    line_adj = load_line_adj()

    shots = shots.sort_values("game_date")
    g = shots.groupby("player_id", group_keys=False)

    shots["EV_L10_rate"] = g["ev_sog"].rolling(10).sum() / g["ev_toi"].rolling(10).sum()
    shots["PP_L10_rate"] = g["pp_sog"].rolling(10).sum() / g["pp_toi"].rolling(10).sum()
    shots["EV_L10_toi"] = g["ev_toi"].rolling(10).mean()
    shots["PP_L10_toi"] = g["pp_toi"].rolling(10).mean()

    shots = shots.dropna(subset=["EV_L10_rate"])

    rows = []
    for m in games:
        teams = {m["away"], m["home"]}
        latest = shots[shots["team"].isin(teams)].groupby("player_id").tail(1)

        for _, r in latest.iterrows():
            opp = next(t for t in teams if t != r.team)
            lam = project_lambda(
                r,
                opp,
                {
                    "pace": 1.0,
                    "line": line_adj.get(opp, 1.0),
                    "goalie": goalie_adj.get(opp, 1.0)
                }
            )
            p = 1 - poisson.cdf(test_line-1, mu=lam)

            rows.append({
                "Player": r.player,
                "Team": r.team,
                "Projection": round(lam,2),
                "Prob ≥ Line (%)": round(p*100,1),
                "Odds": american_odds(p)
            })

    out = pd.DataFrame(rows).sort_values("Projection", ascending=False)
    st.dataframe(out, use_container_width=True)
