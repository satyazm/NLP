import plotly.express as px
import streamlit as st

from data_loader import data_is_ready, load_customer_segments

st.title("Customer Segments")

if not data_is_ready():
    st.warning("Run the pipeline first: `python -m src.pipeline --input data/raw/sample_twcs.csv`")
    st.stop()

df = load_customer_segments()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Segment Sizes")
    counts = df["segment_name"].value_counts().reset_index()
    counts.columns = ["segment", "customers"]
    st.plotly_chart(px.bar(counts, x="segment", y="customers"), width="stretch")

with col2:
    st.subheader("Segments by Sentiment vs Complaint Volume")
    st.plotly_chart(
        px.scatter(df, x="conversation_count", y="avg_sentiment", color="segment_name",
                   size="refund_requests", hover_data=["customer_id"]),
        width="stretch",
    )

st.subheader("Customer Profiles")
st.dataframe(df, width="stretch")
