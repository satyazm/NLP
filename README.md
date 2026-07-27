# Customer Support Intelligence Platform

Turns raw Customer Support on Twitter conversations into conversation-level
analytics, customer segments, agent performance metrics, and an LLM-powered
summarization + RAG assistant — with a Streamlit dashboard on top.

Ships with a small hand-built sample dataset (`data/raw/sample_twcs.csv`, 6
conversations) so the whole pipeline runs end to end immediately. Point it at
the full [Kaggle Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(same column schema) for real results.

## Project Structure

```
data/
  raw/            input CSVs (tweet_id, author_id, inbound, created_at, text,
                  response_tweet_id, in_response_to_tweet_id)
  processed/      cleaned tweets + built conversation threads
  features/       conversation/customer/agent feature tables, segments

src/
  preprocessing/  clean.py, thread_builder.py
  nlp/            sentiment.py, intent.py, ner.py, topic_model.py, embeddings.py
  features/       conversation_features.py, customer_features.py, agent_features.py
  models/         segmentation.py (KMeans)
  llm/            client.py, summarizer.py, root_cause.py, rag_chatbot.py
  pipeline.py     end-to-end orchestrator

dashboard/
  app.py          Overview page
  pages/          Intent, Sentiment, Topics, Segments, Agents, Explorer, Ask-the-Data
```

## Setup

spaCy requires Python 3.10+. If your default `python3` is older, use a
specific interpreter (e.g. `python3.11`) to create the virtualenv.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Copy `.env.example` to `.env` and add a Gemini API key (from
[Google AI Studio](https://aistudio.google.com/apikey)) if you want the GenAI
features (conversation summaries, root-cause narratives, Ask-the-Data):

```bash
cp .env.example .env
```

## Run the pipeline

```bash
python -m src.pipeline --input data/raw/sample_twcs.csv
```

This cleans the raw tweets, threads them into conversations, and writes
conversation/customer/agent feature tables plus customer segments to
`data/processed/` and `data/features/`.

Add `--with-llm` to also build the FAISS index used by the Ask-the-Data page.

To run on the full Kaggle dataset instead, download `twcs.csv` into
`data/raw/` and pass `--input data/raw/twcs.csv`.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

Pages: Overview, Intent Analytics, Sentiment Analytics, Topic Discovery
(BERTopic — needs a reasonably sized dataset to form stable clusters),
Customer Segments, Agent Analytics, Conversation Explorer (with on-demand
Gemini summaries), and Ask the Data (RAG over your conversations).

## Notes

- Sentiment uses VADER (fast, no model download, tuned for short/informal
  text). Intent has a zero-download keyword classifier by default and an
  optional transformer zero-shot classifier (`use_zero_shot=True`) for higher
  accuracy at the cost of a model download.
- The sample dataset is intentionally tiny — segmentation and topic modeling
  will look far more meaningful once you run this against the full 3M-row
  Kaggle dataset (or your own support export in the same schema).
