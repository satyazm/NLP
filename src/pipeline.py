"""End-to-end orchestrator: raw tweets -> conversations -> features -> segments.

Run:
    python -m src.pipeline --input data/raw/sample_twcs.csv
    python -m src.pipeline --input data/raw/twcs.csv --max-rows 200000   # cap a huge file
    python -m src.pipeline --input data/raw/sample_twcs.csv --with-llm   # builds the RAG index
"""
import argparse
import time

import pandas as pd

from src.features.agent_features import build_agent_features
from src.features.conversation_features import build_conversation_features
from src.features.customer_features import build_customer_features
from src.models.segmentation import fit_segments, label_segments
from src.preprocessing.clean import clean_pipeline
from src.preprocessing.thread_builder import build_conversations, save_conversations


def run(input_path: str, with_llm: bool = False, max_rows: int | None = None) -> None:
    t0 = time.time()
    print(f"[1/6] Cleaning raw tweets from {input_path}" + (f" (first {max_rows} rows)" if max_rows else ""))
    tweets_df = clean_pipeline(input_path, "data/processed/tweets_clean.parquet", max_rows=max_rows)
    print(f"      {len(tweets_df)} tweets after cleaning ({time.time() - t0:.0f}s)")

    t0 = time.time()
    print("[2/6] Building conversation threads")
    conversations = build_conversations(tweets_df)
    save_conversations(conversations, "data/processed/conversations.jsonl")
    print(f"      {len(conversations)} conversations ({time.time() - t0:.0f}s)")

    t0 = time.time()
    print("[3/6] Building conversation features (sentiment, intent, entities) — the slow step")
    conv_features = build_conversation_features(conversations)
    conv_features.to_parquet("data/features/conversation_features.parquet", index=False)
    print(f"      done ({time.time() - t0:.0f}s)")

    print("[4/6] Building customer and agent features")
    customer_features = build_customer_features(conv_features)
    customer_features.to_parquet("data/features/customer_features.parquet", index=False)

    agent_features = build_agent_features(conv_features)
    agent_features.to_parquet("data/features/agent_features.parquet", index=False)

    print("[5/6] Segmenting customers")
    k = min(4, len(customer_features)) or 1
    segmented, _, _ = fit_segments(customer_features, k=k)
    segmented = label_segments(segmented)
    segmented.to_parquet("data/features/customer_segments.parquet", index=False)

    if with_llm:
        print("[6/6] Building RAG index over conversations")
        from src.llm.rag_chatbot import build_index
        build_index(conv_features)
    else:
        print("[6/6] Skipping RAG index (pass --with-llm to build it)")

    print("\nDone. Outputs written to data/processed/ and data/features/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/sample_twcs.csv")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Only read the first N rows of --input. Useful for huge files.")
    args = parser.parse_args()
    run(args.input, with_llm=args.with_llm, max_rows=args.max_rows)
