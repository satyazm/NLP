"""Find which issue categories are driving negative sentiment, and explain why.

Stage 1 is plain aggregation (cheap, deterministic). Stage 2 hands the
aggregated stats to Gemini to turn into a short, readable narrative — this
is the "business report" a product manager would actually read.
"""
import pandas as pd

from src.llm.client import ask_llm

NARRATIVE_PROMPT = """Here is a table of customer support issue categories with \
their average sentiment and conversation volume:

{stats_table}

In 3-4 sentences, identify which categories are the biggest problem areas and \
suggest one concrete action for the most urgent one. Be direct and specific.
"""


def summarize_by_intent(conv_features: pd.DataFrame) -> pd.DataFrame:
    stats = conv_features.groupby("intent").agg(
        conversations=("conversation_id", "count"),
        avg_initial_sentiment=("initial_sentiment", "mean"),
        avg_final_sentiment=("final_sentiment", "mean"),
        avg_sentiment_improvement=("sentiment_improvement", "mean"),
        avg_resolution_time_sec=("resolution_time_sec", "mean"),
    ).reset_index()
    return stats.sort_values("avg_final_sentiment")


def explain_root_causes(stats: pd.DataFrame) -> str:
    table_str = stats.round(2).to_string(index=False)
    return ask_llm(NARRATIVE_PROMPT.format(stats_table=table_str))
