"""Cluster customers into behavioral segments (loyal, high-complaint, refund-seeker, VIP)."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "conversation_count",
    "avg_sentiment",
    "avg_sentiment_improvement",
    "avg_conversation_length",
    "avg_response_time_sec",
    "negative_conversations",
    "refund_requests",
]


def fit_segments(customer_features: pd.DataFrame, k: int = 4, random_state: int = 42):
    X = customer_features[FEATURE_COLUMNS].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    result = customer_features.copy()
    result["segment"] = labels
    return result, model, scaler


def label_segments(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a human-readable name to each numeric cluster based on its
    average refund requests, complaint volume, and sentiment."""
    profile = segmented_df.groupby("segment")[FEATURE_COLUMNS].mean()

    names = {}
    for seg_id, row in profile.iterrows():
        if row["refund_requests"] >= profile["refund_requests"].median() and row["refund_requests"] > 0:
            names[seg_id] = "Refund Seekers"
        elif row["negative_conversations"] > profile["negative_conversations"].median():
            names[seg_id] = "High Complaint Customers"
        elif row["avg_sentiment"] > profile["avg_sentiment"].median():
            names[seg_id] = "Loyal Customers"
        else:
            names[seg_id] = "New / Occasional Customers"

    segmented_df = segmented_df.copy()
    segmented_df["segment_name"] = segmented_df["segment"].map(names)
    return segmented_df


if __name__ == "__main__":
    customer_df = pd.read_parquet("data/features/customer_features.parquet")
    k = min(4, len(customer_df))
    segmented, _, _ = fit_segments(customer_df, k=k)
    segmented = label_segments(segmented)
    segmented.to_parquet("data/features/customer_segments.parquet", index=False)
    print(segmented[["customer_id", "segment", "segment_name"]])
