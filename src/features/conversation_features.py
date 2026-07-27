"""Per-conversation metrics: timing, sentiment arc, intent, and entities.

This is the central feature table — customer and agent features are just
aggregations of these rows.
"""
import pandas as pd

from src.nlp.intent import classify_conversation_intent
from src.nlp.ner import extract_entities_batch
from src.nlp.sentiment import score_conversation_timeline


def _response_time_seconds(t1: dict, t2: dict) -> float:
    return (pd.Timestamp(t2["created_at"]) - pd.Timestamp(t1["created_at"])).total_seconds()


def build_conversation_features(conversations: list[dict], use_zero_shot_intent: bool = False) -> pd.DataFrame:
    full_texts = [" ".join(t["text"] for t in conv["turns"]) for conv in conversations]
    entities_per_conv = extract_entities_batch(full_texts)

    rows = []
    for conv, full_text, entities in zip(conversations, full_texts, entities_per_conv):
        turns = conv["turns"]
        customer_turns = [t for t in turns if t["speaker"] == "customer"]
        agent_turns = [t for t in turns if t["speaker"] == "agent"]

        sentiment = score_conversation_timeline(turns)
        intent = classify_conversation_intent(turns, use_zero_shot=use_zero_shot_intent)

        first_response_time = None
        for i in range(len(turns) - 1):
            if turns[i]["speaker"] == "customer" and turns[i + 1]["speaker"] == "agent":
                first_response_time = _response_time_seconds(turns[i], turns[i + 1])
                break

        resolution_time = _response_time_seconds(turns[0], turns[-1]) if len(turns) > 1 else 0.0

        rows.append({
            "conversation_id": conv["conversation_id"],
            "started_at": turns[0]["created_at"],
            "customer_id": customer_turns[0]["author_id"] if customer_turns else None,
            "agent_id": agent_turns[0]["author_id"] if agent_turns else None,
            "num_messages": len(turns),
            "num_customer_messages": len(customer_turns),
            "num_agent_messages": len(agent_turns),
            "avg_message_length": sum(len(t["text"]) for t in turns) / len(turns),
            "first_response_time_sec": first_response_time,
            "resolution_time_sec": resolution_time,
            "initial_sentiment": sentiment["initial_sentiment"],
            "final_sentiment": sentiment["final_sentiment"],
            "sentiment_improvement": sentiment["sentiment_improvement"],
            "intent": intent,
            "products": entities["product"],
            "locations": entities["location"],
            "order_numbers": entities["order_number"],
            "full_text": full_text,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.preprocessing.thread_builder import load_conversations

    convs = load_conversations("data/processed/conversations.jsonl")
    df = build_conversation_features(convs)
    df.to_parquet("data/features/conversation_features.parquet", index=False)
    print(df.head())
