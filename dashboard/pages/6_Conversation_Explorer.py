import json

import streamlit as st

from data_loader import data_is_ready, load_conversation_features

st.title("Conversation Explorer")

if not data_is_ready():
    st.warning("Run the pipeline first: `python -m src.pipeline --input data/raw/sample_twcs.csv`")
    st.stop()

df = load_conversation_features()

with open("data/processed/conversations.jsonl") as f:
    conversations = {json.loads(line)["conversation_id"]: json.loads(line)["turns"] for line in f}

col1, col2 = st.columns(2)
with col1:
    intent_filter = st.multiselect("Filter by intent", sorted(df["intent"].unique()))
with col2:
    customer_filter = st.text_input("Filter by customer id (contains)")

filtered = df.copy()
if intent_filter:
    filtered = filtered[filtered["intent"].isin(intent_filter)]
if customer_filter:
    filtered = filtered[filtered["customer_id"].str.contains(customer_filter, case=False, na=False)]

st.dataframe(
    filtered[["conversation_id", "customer_id", "agent_id", "intent", "final_sentiment", "started_at"]],
    width="stretch",
)

conv_id = st.selectbox("Select a conversation to inspect", filtered["conversation_id"].tolist())

if conv_id is not None:
    row = df[df["conversation_id"] == conv_id].iloc[0]
    turns = conversations[conv_id]

    st.subheader(f"Conversation #{conv_id} — {row['intent']}")
    for t in turns:
        speaker = "🧑 Customer" if t["speaker"] == "customer" else "🎧 Agent"
        st.markdown(f"**{speaker}:** {t['text']}")

    st.write("**Entities:**", {
        "products": row["products"], "locations": row["locations"], "order_numbers": row["order_numbers"],
    })
    st.write(f"**Sentiment:** {row['initial_sentiment']:.2f} → {row['final_sentiment']:.2f} "
             f"({row['sentiment_improvement']:+.2f})")

    if st.button("Generate AI summary"):
        try:
            from src.llm.summarizer import summarize_conversation
            with st.spinner("Asking Gemini..."):
                summary = summarize_conversation(turns)
            for key, label in [("summary", "Summary"), ("customer_problem", "Customer Problem"),
                                ("root_cause", "Root Cause"), ("suggested_resolution", "Suggested Resolution"),
                                ("recommended_follow_up", "Recommended Follow-up")]:
                if summary.get(key):
                    st.markdown(f"**{label}:** {summary[key]}")
        except Exception as e:
            st.error(f"Couldn't reach Gemini — check GEMINI_API_KEY in your .env. ({e})")
