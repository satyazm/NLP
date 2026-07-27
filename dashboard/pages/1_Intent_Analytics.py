import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import data_is_ready, load_conversation_features

st.title("Intent Analytics")

if not data_is_ready():
    st.warning("Run the pipeline first: `python -m src.pipeline --input data/raw/sample_twcs.csv`")
    st.stop()

df = load_conversation_features()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Intent Distribution")
    counts = df["intent"].value_counts().reset_index()
    counts.columns = ["intent", "conversations"]
    st.plotly_chart(px.pie(counts, names="intent", values="conversations"), width="stretch")

with col2:
    st.subheader("Avg Sentiment by Intent")
    by_intent = df.groupby("intent")["final_sentiment"].mean().reset_index()
    st.plotly_chart(px.bar(by_intent, x="intent", y="final_sentiment"), width="stretch")

st.subheader("Weekly Trend")
trend_df = df.copy()
trend_df["started_at"] = pd.to_datetime(trend_df["started_at"])
trend_df["week"] = trend_df["started_at"].dt.to_period("W").dt.start_time
weekly = trend_df.groupby(["week", "intent"]).size().reset_index(name="conversations")
if weekly["week"].nunique() < 2:
    st.caption("Only one week of data in this sample — trend will look more interesting "
               "once you run the pipeline on the full multi-week dataset.")
st.plotly_chart(px.line(weekly, x="week", y="conversations", color="intent", markers=True),
                 width="stretch")

st.subheader("Raw Table")
st.dataframe(df[["conversation_id", "intent", "num_messages", "final_sentiment"]],
             width="stretch")
