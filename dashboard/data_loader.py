"""Shared, cached data loading for every dashboard page.

Also puts the project root on sys.path (Streamlit only adds each page's own
directory) and loads .env, so every page can `from src...` and read
GEMINI_API_KEY etc. after importing this module first.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

FEATURES_DIR = str(_PROJECT_ROOT / "data" / "features")


@st.cache_data
def load_conversation_features() -> pd.DataFrame:
    return pd.read_parquet(f"{FEATURES_DIR}/conversation_features.parquet")


@st.cache_data
def load_customer_features() -> pd.DataFrame:
    return pd.read_parquet(f"{FEATURES_DIR}/customer_features.parquet")


@st.cache_data
def load_agent_features() -> pd.DataFrame:
    return pd.read_parquet(f"{FEATURES_DIR}/agent_features.parquet")


@st.cache_data
def load_customer_segments() -> pd.DataFrame:
    return pd.read_parquet(f"{FEATURES_DIR}/customer_segments.parquet")


def data_is_ready() -> bool:
    import os
    required = ["conversation_features.parquet", "customer_features.parquet",
                "agent_features.parquet", "customer_segments.parquet"]
    return all(os.path.exists(f"{FEATURES_DIR}/{f}") for f in required)
