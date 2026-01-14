# ---------------------------------------------------------------
# hockey_prop_stop_app.py
# Streamlit UI for Hockey Prop Stop
# ---------------------------------------------------------------

# --- Core Imports ---
import importlib.util
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ---------------------------------------------------------------
# Dynamically import hockey_model.py (safe for Streamlit Cloud)
# ---------------------------------------------------------------
module_path = os.path.join(os.path.dirname(__file__), "hockey_model.py")
spec = importlib.util.spec_from_file_location("hockey_model", module_path)
hockey_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hockey_model)

# Get functions from the loaded module
parse_raw_files = hockey_model.parse_raw_files
project_matchup = hockey_model.project_matchup

# ---------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

st.markdown(
    """
    <h1 style='text-align:center; color:#00B140;'>🏒 Hockey Prop Stop</h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Team-vs-Team matchup analytics with exponential regression weighting
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Raw NHL Data Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSVs (skaters, teams, shots, goalies, lines)",
    type=["csv"],
    accept_multiple_files=True
)

data = pd.DataFrame()
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded. Parsing raw data...")

    raw_files = {}
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, nrows=200000)
            raw_files[f.name] = df
        except Exception as e:
            st.warning(f"⚠️ Could not read {f.name}: {e}")

    skaters_df, teams_df, shots_df, goalies_df, lines_df = parse_raw_fil_
