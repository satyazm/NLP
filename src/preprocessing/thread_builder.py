"""Turn a flat table of tweets into ordered customer/agent conversations."""
import json

import pandas as pd


def _first_id(cell) -> int | None:
    """response_tweet_id / in_response_to_tweet_id can be 'NaN', a single id,
    or a comma-separated list of ids. Return the first valid one."""
    if cell is None or pd.isna(cell) or str(cell).strip().lower() == "nan":
        return None
    first = str(cell).split(",")[0].strip()
    return int(float(first)) if first else None


def build_conversations(df: pd.DataFrame) -> list[dict]:
    tweets = df.set_index("tweet_id").to_dict(orient="index")

    openers = [
        tid for tid, row in tweets.items()
        if row["inbound"] and _first_id(row.get("in_response_to_tweet_id")) is None
    ]
    openers.sort(key=lambda tid: tweets[tid]["created_at"])

    conversations = []
    for conv_id, opener_id in enumerate(openers, start=1):
        turns = []
        tid = opener_id
        seen = set()
        while tid is not None and tid in tweets and tid not in seen:
            seen.add(tid)
            row = tweets[tid]
            turns.append({
                "tweet_id": int(tid),
                "speaker": "customer" if row["inbound"] else "agent",
                "author_id": row["author_id"],
                "text": row.get("clean_text", row["text"]),
                "created_at": str(row["created_at"]),
            })
            tid = _first_id(row.get("response_tweet_id"))

        if len(turns) >= 2:  # keep only conversations with at least one reply
            conversations.append({"conversation_id": conv_id, "turns": turns})

    return conversations


def save_conversations(conversations: list[dict], output_path: str) -> None:
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


def load_conversations(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    tweets_df = pd.read_parquet("data/processed/tweets_clean.parquet")
    convs = build_conversations(tweets_df)
    save_conversations(convs, "data/processed/conversations.jsonl")
    print(f"Built {len(convs)} conversations")
