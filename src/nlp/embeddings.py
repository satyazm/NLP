"""Sentence embeddings for topic modeling and RAG retrieval."""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Single-threaded to avoid a crash where torch's OpenMP thread pool races
# with faiss's independent one in the same process (see rag_chatbot.py).
torch.set_num_threads(1)

_model = None


def _get_model():
    global _model
    if _model is None:
        # Force CPU: PyTorch's MPS (Apple GPU) backend segfaults on some macOS
        # versions when a long-lived process (e.g. the Streamlit server) moves
        # tensors to/from it repeatedly across reruns. This model is small
        # enough that CPU is plenty fast for this workload.
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    return _get_model().encode(texts, show_progress_bar=False)
