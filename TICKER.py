import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🏒 Live Player Stat Ticker (Shots on Goal)")

# --------------------------------------------------
# MOCK LIVE FEED (replace later with real API)
# --------------------------------------------------
def mock_live_feed(players, current_stats):
    """
    Simulates live stat updates.
    Randomly adds shots to players.
    """
    new_stats = current_stats.copy()

    for p in players:
        if np.random.rand() < 0.35:  # 35% chance per refresh
            new_stats[p] += np.random.choice([0, 1])

    return new_stats


# --------------------------------------------------
# PLAYER SELECTION
# --------------------------------------------------
players_available = [
    "Auston Matthews",
    "Connor McDavid",
    "Nathan MacKinnon",
    "David Pastrnak",
    "Artemi Panarin"
]

selected_players = st.multiselect(
    "Select players to track",
    players_available,
    default=players_available[:2]
)

if not selected_players:
    st.stop()

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "current_stats" not in st.session_state:
    st.session_state.current_stats = {p: 0 for p in selected_players}

if "previous_stats" not in st.session_state:
    st.session_state.previous_stats = st.session_state.current_stats.copy()

if "ticker_events" not in st.session_state:
    st.session_state.ticker_events = []

# --------------------------------------------------
# AUTO REFRESH
# --------------------------------------------------
st.autorefresh(interval=5000, key="live_refresh")  # every 5 seconds

# --------------------------------------------------
# FETCH LIVE DATA
# --------------------------------------------------
new_stats = mock_live_feed(
    selected_players,
    st.session_state.current_stats
)

# --------------------------------------------------
# DELTA DETECTION (THIS IS THE KEY PART)
# --------------------------------------------------
for player in selected_players:
    prev = st.session_state.previous_stats.get(player, 0)
    curr = new_stats.get(player, 0)
    delta = curr - prev

    if delta > 0:
        st.session_state.ticker_events.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "player": player,
                "delta": delta,
                "total": curr
            }
        )

# Update state AFTER comparison
st.session_state.previous_stats = st.session_state.current_stats.copy()
st.session_state.current_stats = new_stats.copy()

# --------------------------------------------------
# DISPLAY CURRENT STATS
# --------------------------------------------------
st.subheader("📊 Current Live Stats")

stats_df = pd.DataFrame.from_dict(
    st.session_state.current_stats,
    orient="index",
    columns=["Shots on Goal"]
)

st.dataframe(stats_df, use_container_width=True)

# --------------------------------------------------
# LIVE TICKER
# --------------------------------------------------
st.subheader("🔥 Live Stat Ticker")

if not st.session_state.ticker_events:
    st.info("No stat changes yet…")
else:
    for event in st.session_state.ticker_events[:10]:
        st.markdown(
            f"""
            **{event['time']}** — 🏒 **{event['player']}**
            +{event['delta']} SOG  
            _Total: {event['total']}_
            """
        )
