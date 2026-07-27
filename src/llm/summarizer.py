"""Turn a raw conversation transcript into a structured summary via Gemini."""
import json

from src.llm.client import ask_llm

PROMPT_TEMPLATE = """You are analyzing a customer support conversation. Read the \
transcript below and respond with ONLY a JSON object (no markdown fences) with \
these keys: "summary", "customer_problem", "root_cause", "suggested_resolution", \
"recommended_follow_up". Keep each value to one or two sentences.

Transcript:
{transcript}
"""


def _format_transcript(turns: list[dict]) -> str:
    return "\n".join(f"{t['speaker'].capitalize()}: {t['text']}" for t in turns)


def summarize_conversation(turns: list[dict]) -> dict:
    prompt = PROMPT_TEMPLATE.format(transcript=_format_transcript(turns))
    raw = ask_llm(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": raw, "customer_problem": None, "root_cause": None,
                "suggested_resolution": None, "recommended_follow_up": None}
