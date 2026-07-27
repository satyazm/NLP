"""RAG assistant: ask natural-language questions over historical conversations.

Conversations -> embeddings -> FAISS index -> retrieve top-k -> Gemini answers
the question grounded in those transcripts and cites conversation ids.
"""
import os

# faiss and torch (pulled in by sentence-transformers) each bundle their own
# OpenMP runtime. Suppressing the duplicate-runtime abort isn't enough on its
# own — two independent thread pools can still crash when both try to run
# parallel regions in the same process. Forcing everything single-threaded
# sidesteps the race entirely; these env vars must be set before numpy/faiss/
# torch are imported for the first time.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pickle

import faiss
import numpy as np
import pandas as pd

faiss.omp_set_num_threads(1)

from src.llm.client import ask_llm
from src.nlp.embeddings import embed_texts

ANSWER_PROMPT = """Answer the customer support question below using ONLY the \
retrieved conversation excerpts as evidence. Cite conversation ids like [#12] \
inline. If the excerpts don't contain the answer, say so.

Question: {question}

Retrieved conversations:
{context}
"""


def build_index(conv_features: pd.DataFrame, index_path: str = "vectorstore/conversations.faiss",
                 meta_path: str = "vectorstore/conversations_meta.pkl") -> None:
    embeddings = embed_texts(conv_features["full_text"].tolist()).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    meta = conv_features[["conversation_id", "intent", "full_text"]].to_dict(orient="records")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)


def _load(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def retrieve(query: str, k: int = 5, index_path: str = "vectorstore/conversations.faiss",
             meta_path: str = "vectorstore/conversations_meta.pkl") -> list[dict]:
    index, meta = _load(index_path, meta_path)
    query_vec = embed_texts([query]).astype("float32")
    faiss.normalize_L2(query_vec)

    _, indices = index.search(query_vec, min(k, len(meta)))
    return [meta[i] for i in indices[0] if i != -1]


def ask_question(query: str, k: int = 5) -> str:
    hits = retrieve(query, k=k)
    context = "\n\n".join(f"[#{h['conversation_id']}] ({h['intent']}): {h['full_text']}" for h in hits)
    prompt = ANSWER_PROMPT.format(question=query, context=context)
    return ask_llm(prompt, max_tokens=600)


if __name__ == "__main__":
    df = pd.read_parquet("data/features/conversation_features.parquet")
    build_index(df)
    print(ask_question("What are customers complaining about most?"))
