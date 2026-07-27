"""Conversation-level intent classification.

Two implementations are provided:
- `rule_based_intent`: instant, no downloads, good enough for a first pass and
  for keeping the pipeline runnable offline.
- `zero_shot_intent`: a transformer zero-shot classifier (facebook/bart-large-mnli)
  for higher accuracy once you're ready to pay the model download + inference cost.
"""

INTENT_LABELS = [
    "Refund",
    "Delivery Delay",
    "Payment Failure",
    "Cancellation",
    "Technical Issue",
    "Subscription",
    "Account Access",
]

_KEYWORDS = {
    "Refund": ["refund", "money back", "reimburse"],
    "Delivery Delay": ["hasn't arrived", "not arrived", "delayed", "late delivery", "still waiting", "shipping"],
    "Payment Failure": ["payment failed", "charged twice", "deducted", "duplicate charge", "declined"],
    "Cancellation": ["cancel"],
    "Technical Issue": ["crash", "bug", "not working", "error", "broken"],
    "Subscription": ["subscription", "renew", "plan", "membership"],
    "Account Access": ["log in", "login", "can't access", "otp", "password", "locked out", "suspended"],
}


def rule_based_intent(text: str) -> str:
    lowered = text.lower()
    for intent, keywords in _KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return intent
    return "Other"


_zero_shot_pipeline = None


def zero_shot_intent(text: str, labels: list[str] = INTENT_LABELS) -> str:
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        from transformers import pipeline
        _zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    result = _zero_shot_pipeline(text, candidate_labels=labels)
    return result["labels"][0]


def classify_conversation_intent(turns: list[dict], use_zero_shot: bool = False) -> str:
    """Classify a conversation's intent from its opening customer message."""
    opener = next((t["text"] for t in turns if t["speaker"] == "customer"), "")
    if use_zero_shot:
        return zero_shot_intent(opener)
    return rule_based_intent(opener)
