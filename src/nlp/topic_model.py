"""Unsupervised topic discovery over conversation openers using BERTopic.

BERTopic needs a reasonable number of documents to form stable clusters
(a few dozen at minimum). On very small datasets it may collapse everything
into a single topic — that's expected, not a bug; it just means there isn't
enough data yet to discover distinct themes.
"""
from bertopic import BERTopic

from src.nlp.embeddings import embed_texts


def fit_topics(docs: list[str], min_topic_size: int = 2):
    embeddings = embed_texts(docs)
    model = BERTopic(min_topic_size=min_topic_size, verbose=False)
    topics, probs = model.fit_transform(docs, embeddings)
    return model, topics, probs


def topic_labels(model: BERTopic) -> dict:
    """Map topic id -> top keywords, e.g. {0: ['refund', 'payment', 'bank']}."""
    info = model.get_topic_info()
    labels = {}
    for topic_id in info["Topic"]:
        if topic_id == -1:
            continue  # -1 is BERTopic's "outlier / no topic" bucket
        labels[topic_id] = [word for word, _ in model.get_topic(topic_id)][:5]
    return labels
