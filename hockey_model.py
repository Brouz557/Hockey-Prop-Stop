# ---------------------------------------------------------------
# hockey_model.py
# Core analytics engine for Hockey Prop Stop / Hockey Bot
# Fully raw-data tolerant + matchup-aware model
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson
import re
import chardet

# ---------------------------------------------------------------
# Safe CSV reader
# ---------------------------------------------------------------
def safe_read_csv(uploaded_file):
    """Read any uploaded CSV safely and return a cleaned DataFrame."""
    try:
        raw_bytes = uploaded_file.read()
        if not raw_bytes:
            return pd.DataFrame()
        enc_guess = chardet.detect(raw_bytes)
        encoding = enc_guess.get("encoding", "utf-8")
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding,
                         on_bad_lines="skip", engine="python")
        df = df.dropna(how="all")
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        print(f"⚠️ Failed to parse {getattr(uploaded_file, 'name', 'unknown')} — {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------
# Smart parser for raw NHL data
# ---------------------------------------------------------
