import plotly.express as px
import streamlit as st

from data_loader import data_is_ready, load_conversation_features

st.title("Sentiment Analytics")

if not data_is_ready():
    st.warning("Run the pipeline first: `python -m src.pipeline --input data/raw/sample_twcs.csv`")
    st.stop()

df = load_conversation_features()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Improvement Distribution")
    st.plotly_chart(px.histogram(df, x="sentiment_improvement", nbins=15), width="stretch")

with col2:
    st.subheader("Positive vs Negative Endings")
    df["outcome"] = df["final_sentiment"].apply(lambda s: "Positive" if s >= 0 else "Negative")
    counts = df["outcome"].value_counts().reset_index()
    counts.columns = ["outcome", "conversations"]
    st.plotly_chart(px.pie(counts, names="outcome", values="conversations",
                            color="outcome",
                            color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728"}),
                     width="stretch")

st.subheader("Sentiment Arc by Conversation")
melted = df.melt(id_vars=["conversation_id", "intent"],
                  value_vars=["initial_sentiment", "final_sentiment"],
                  var_name="stage", value_name="sentiment")
st.plotly_chart(
    px.line(melted, x="stage", y="sentiment", color="conversation_id",
            hover_data=["intent"], markers=True),
    width="stretch",
)

st.subheader("Product-wise Sentiment")
exploded = df.explode("products").dropna(subset=["products"])
if exploded.empty:
    st.caption("No product entities were extracted from this sample — this fills in once "
               "the NER stage runs over a larger, product-mention-heavy dataset.")
else:
    by_product = exploded.groupby("products")["final_sentiment"].mean().reset_index()
    st.plotly_chart(px.bar(by_product, x="products", y="final_sentiment"), width="stretch")
