# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Hockey Prop Stop - Streamlit app
# Dark green / silver theme with sample data
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import sys, os

# ---- Ensure local imports work on Streamlit Cloud ----
sys.path.append(os.path.dirname(__file__))
from hockey_model import build_model, project_matchup


# ---- Sample data function (for when no files uploaded) ----
def load_sample():
    np.random.seed(42)
    players = ["Aho", "Larkin", "Necas", "Burns", "Raymond", "Walman"]
    pos = ["F", "F", "F", "D", "F", "D"]
    team = ["CAR", "DET", "CAR", "CAR", "DET", "DET"]
    sog = np.random.uniform(1.0, 4.0, 6).round(2)
    prob = np.random.uniform(0.45, 0.8, 6).round(2)
    strength = ["Strong" if p > 0.7 else "Moderate" if p > 0.55 else "Weak" for p in prob]
    match = ["Favorable", "Neutral", "Favorable", "Favorable", "Unfavorable", "Unfavorable"]
    odds = (1 / prob - 1) * 100
    df = pd.DataFrame({
        "Player": players, "Team": team, "Pos": pos,
        "Projected SOG": sog, "Probability (Over)": prob,
        "Signal Strength": strength, "Matchup Favorability": match,
        "Lowest Playable Odds": odds.round(0)
    })
    return df


# ---- Streamlit setup ----
st.set_page_config(
    page_title="Hockey Prop Stop",
    layout="wide",
    page_icon="🏒"
)

# ---- Header ----
st.markdown(
    """
    <h1 style='text-align:center; color:#BFC0C0;'>
        <span style='color:#00B140;'>🏒 Hockey Prop Stop</span>
    </h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Data-driven shots-on-goal analytics dashboard
    </p>
    """,
    unsafe_allow_html=True
)

# ---- Sidebar for uploads ----
st.sidebar.header("📂 Upload Daily Data Files"

