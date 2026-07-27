import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import data_is_ready, load_conversation_features

st.title("Topic Discovery")
st.caption("Runs BERTopic over conversation transcripts to discover themes without "
           "manually defining categories. Needs a decent number of documents to form "
           "stable clusters — small samples may collapse into one topic.")

if not data_is_ready():
    st.warning("Run the pipeline first: `python -m src.pipeline --input data/raw/sample_twcs.csv`")
    st.stop()

df = load_conversation_features()

if st.button("Run topic discovery"):
    # Scale cluster size to corpus size: a fixed min_topic_size=2 that lets a
    # 6-doc sample form any topic at all produces thousands of 2-3 doc
    # micro-topics on tens of thousands of real conversations. Targeting
    # roughly one topic per 1000 docs keeps the result readable either way.
    min_topic_size = max(10, len(df) // 1000)

    with st.spinner(f"Embedding conversations and clustering (min_topic_size={min_topic_size})..."):
        from src.nlp.topic_model import fit_topics, topic_labels

        try:
            model, topics, probs = fit_topics(df["full_text"].tolist(), min_topic_size=min_topic_size)
            df = df.assign(topic=topics)
            labels = topic_labels(model)

            sizes = df[df["topic"] != -1]["topic"].value_counts().reset_index()
            sizes.columns = ["topic", "conversations"]
            sizes = sizes.sort_values("conversations", ascending=False)

            st.subheader(f"Discovered {len(labels)} topics")

            st.subheader("Top Topic Keywords")
            for topic_id in sizes["topic"].head(20):
                st.write(f"**Topic {topic_id}:** {', '.join(labels[topic_id])}")

            st.subheader("Topic Sizes (top 20)")
            if not sizes.empty:
                st.plotly_chart(px.bar(sizes.head(20), x="topic", y="conversations"), width="stretch")

            st.subheader("Conversations by Topic")
            st.dataframe(df[["conversation_id", "intent", "topic"]], width="stretch")
        except Exception as e:
            st.error(f"Topic modeling needs more data to form stable clusters: {e}")
else:
    st.info("Click the button above to run topic discovery (downloads a small embedding "
            "model on first use).")
