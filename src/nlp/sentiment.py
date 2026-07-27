"""Turn-level sentiment scoring using VADER (lightweight, no model download).

VADER is tuned for short, informal text like tweets, which makes it a good
default here. Swap in a transformer (e.g. cardiffnlp/twitter-roberta-base-sentiment)
in `score_text` if you need higher accuracy and can afford the extra latency.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """Return a compound sentiment score in [-1, 1]."""
    return _analyzer.polarity_scores(text)["compound"]


def score_conversation_timeline(turns: list[dict]) -> dict:
    """Score every turn and summarize the conversation's sentiment arc."""
    scored = [{**t, "sentiment": score_text(t["text"])} for t in turns]
    customer_scores = [t["sentiment"] for t in scored if t["speaker"] == "customer"]

    initial = customer_scores[0] if customer_scores else 0.0
    final = customer_scores[-1] if customer_scores else 0.0

    return {
        "turns": scored,
        "initial_sentiment": initial,
        "final_sentiment": final,
        "sentiment_improvement": final - initial,
    }
