"""Aggregate conversation-level features up to one row per customer."""
import pandas as pd


def build_customer_features(conv_features: pd.DataFrame) -> pd.DataFrame:
    grouped = conv_features.groupby("customer_id")

    customer_df = grouped.agg(
        conversation_count=("conversation_id", "count"),
        avg_sentiment=("final_sentiment", "mean"),
        avg_sentiment_improvement=("sentiment_improvement", "mean"),
        avg_conversation_length=("num_messages", "mean"),
        avg_response_time_sec=("first_response_time_sec", "mean"),
        negative_conversations=("final_sentiment", lambda s: (s < 0).sum()),
    ).reset_index()

    refund_counts = (
        conv_features[conv_features["intent"] == "Refund"]
        .groupby("customer_id").size().rename("refund_requests")
    )
    customer_df = customer_df.merge(refund_counts, on="customer_id", how="left")
    customer_df["refund_requests"] = customer_df["refund_requests"].fillna(0).astype(int)

    return customer_df


if __name__ == "__main__":
    conv_df = pd.read_parquet("data/features/conversation_features.parquet")
    df = build_customer_features(conv_df)
    df.to_parquet("data/features/customer_features.parquet", index=False)
    print(df)
