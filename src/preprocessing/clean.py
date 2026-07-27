"""Load and clean the raw Customer Support on Twitter export."""
import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
DELETED_MARKERS = ("tweet is unavailable", "tweet you shared is no longer available")

# Twitter's native export format, e.g. "Tue Oct 31 22:10:47 +0000 2017".
# Parsing with this format explicitly is ~15x faster than letting pandas fall
# back to per-row dateutil parsing, which matters at millions of rows.
TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def _parse_created_at(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, format=TWITTER_DATE_FORMAT)
    except (ValueError, TypeError):
        return pd.to_datetime(series)


def load_raw(path: str, max_rows: int | None = None) -> pd.DataFrame:
    # response_tweet_id / in_response_to_tweet_id are read as strings because the
    # full Kaggle dataset sometimes stores multiple comma-separated reply ids.
    # nrows is passed straight to pandas so a --max-rows run never reads the
    # rest of a multi-GB file off disk in the first place.
    df = pd.read_csv(path, dtype={"tweet_id": "Int64", "response_tweet_id": "str",
                                   "in_response_to_tweet_id": "str"}, nrows=max_rows)
    df["created_at"] = _parse_created_at(df["created_at"])
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="tweet_id").reset_index(drop=True)


def remove_deleted(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].fillna("").str.lower()
    is_deleted = text.str.contains("|".join(DELETED_MARKERS))
    return df[~is_deleted & (text.str.len() > 0)].reset_index(drop=True)


def clean_text(text: str) -> str:
    text = URL_RE.sub("", str(text))
    text = MENTION_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_pipeline(input_path: str, output_path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = load_raw(input_path, max_rows=max_rows)
    df = remove_duplicates(df)
    df = remove_deleted(df)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    df.to_parquet(output_path, index=False)
    return df


if __name__ == "__main__":
    clean_pipeline("data/raw/sample_twcs.csv", "data/processed/tweets_clean.parquet")
