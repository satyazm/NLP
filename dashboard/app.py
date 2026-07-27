import plotly.express as px
import streamlit as st

from data_loader import data_is_ready, load_agent_features, load_conversation_features

st.set_page_config(page_title="Customer Support Intelligence", layout="wide")
st.title("Customer Support Intelligence — Overview")

if not data_is_ready():
    st.warning(
        "No processed data found yet. Run the pipeline first:\n\n"
        "`python -m src.pipeline --input data/raw/sample_twcs.csv`"
    )
    st.stop()

conv_df = load_conversation_features()
agent_df = load_agent_features()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Conversations", len(conv_df))
col2.metric("Avg Final Sentiment", f"{conv_df['final_sentiment'].mean():.2f}")
col3.metric("Avg First Response Time", f"{conv_df['first_response_time_sec'].mean() / 60:.0f} min")
col4.metric("Avg Sentiment Improvement", f"{conv_df['sentiment_improvement'].mean():+.2f}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Top Complaint Categories")
    intent_counts = conv_df["intent"].value_counts().reset_index()
    intent_counts.columns = ["intent", "conversations"]
    fig = px.bar(intent_counts, x="intent", y="conversations")
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Sentiment: Initial vs Final")
    fig = px.scatter(conv_df, x="initial_sentiment", y="final_sentiment",
                      hover_data=["conversation_id", "intent"],
                      labels={"initial_sentiment": "Initial", "final_sentiment": "Final"})
    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Agent Leaderboard (by sentiment improvement)")
st.dataframe(agent_df, width="stretch")

st.caption("Use the sidebar to explore intents, sentiment trends, topics, customer segments, "
           "agent analytics, individual conversations, and the Ask-the-Data assistant.")
