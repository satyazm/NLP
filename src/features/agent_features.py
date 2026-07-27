"""Aggregate conversation-level features up to one row per support agent."""
import pandas as pd


def build_agent_features(conv_features: pd.DataFrame) -> pd.DataFrame:
    df = conv_features.dropna(subset=["agent_id"])
    grouped = df.groupby("agent_id")

    agent_df = grouped.agg(
        conversations_handled=("conversation_id", "count"),
        avg_response_time_sec=("first_response_time_sec", "mean"),
        avg_resolution_time_sec=("resolution_time_sec", "mean"),
        avg_sentiment_improvement=("sentiment_improvement", "mean"),
    ).reset_index()

    # Proxy for resolution: did the customer's sentiment end non-negative?
    resolved_rate = grouped.apply(
        lambda g: (g["final_sentiment"] >= 0).mean(), include_groups=False
    ).rename("resolution_rate")
    agent_df = agent_df.merge(resolved_rate, on="agent_id")

    escalation_rate = grouped.apply(
        lambda g: (g["sentiment_improvement"] < 0).mean(), include_groups=False
    ).rename("escalation_rate")
    agent_df = agent_df.merge(escalation_rate, on="agent_id")

    return agent_df.sort_values("avg_sentiment_improvement", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    conv_df = pd.read_parquet("data/features/conversation_features.parquet")
    df = build_agent_features(conv_df)
    df.to_parquet("data/features/agent_features.parquet", index=False)
    print(df)
