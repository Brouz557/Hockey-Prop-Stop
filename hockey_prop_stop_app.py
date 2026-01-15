# ---------------------------------------------------------------
# hockey_prop_stop_app.py — Streamlit Front End (Excel/CSV Ready)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hockey_model

st.set_page_config(page_title="Hockey Prop Stop", layout="wide", page_icon="🏒")

st.markdown(
    """
    <h1 style='text-align:center; color:#BFC0C0;'>
        <span style='color:#00B140;'>🏒 Hockey Prop Stop</span>
    </h1>
    <p style='text-align:center; color:#BFC0C0;'>
        Team-vs-Team matchup analytics with adaptive regression weighting
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Sidebar: Upload Daily Data Files (CSV or Excel)
# ---------------------------------------------------------------
st.sidebar.header("📂 Upload Daily Data Files")

uploaded_skaters = st.sidebar.file_uploader("NHL Skaters (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_teams   = st.sidebar.file_uploader("NHL TEAMs (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_shots   = st.sidebar.file_uploader("shots (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_goalies = st.sidebar.file_uploader("goalies (CSV or Excel)", type=["csv", "xlsx", "xls"])
uploaded_lines   = st.sidebar.file_uploader("lines (CSV or Excel)", type=["csv", "xlsx", "xls"])

def load_file(file):
    if not file:
        return pd.DataFrame()
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error reading {file.name}: {e}")
        return pd.DataFrame()

raw_files = {
    "skaters": load_file(uploaded_skaters),
    "teams": load_file(uploaded_teams),
    "shots": load_file(uploaded_shots),
    "goalies": load_file(uploaded_goalies),
    "lines": load_file(uploaded_lines),
}

# -----------
