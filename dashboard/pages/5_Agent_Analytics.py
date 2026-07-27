import plotly.express as px
import streamlit as st

from data_loader import data_is_ready, load_agent_features

st.title("Agent Analytics")

if not data_is_ready():
    st.warning("Run the pipeline first: `python -m src.pipeline --input data/raw/sample_twcs.csv`")
    st.stop()

df = load_agent_features()

st.subheader("Leaderboard")
st.dataframe(
    df.style.format({
        "avg_response_time_sec": "{:.0f}",
        "avg_resolution_time_sec": "{:.0f}",
        "avg_sentiment_improvement": "{:+.2f}",
        "resolution_rate": "{:.0%}",
        "escalation_rate": "{:.0%}",
    }),
    width="stretch",
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Avg Response Time by Agent")
    st.plotly_chart(px.bar(df, x="agent_id", y="avg_response_time_sec"), width="stretch")

with col2:
    st.subheader("Sentiment Improvement by Agent")
    st.plotly_chart(px.bar(df, x="agent_id", y="avg_sentiment_improvement"), width="stretch")
