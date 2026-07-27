"""Named entity extraction: products, locations, money, dates, orgs, order numbers."""
import re

import spacy

ORDER_NUMBER_RE = re.compile(r"#\s?\d{4,}|\b[A-Z]{2,}-\d{4,}\b")

_SPACY_LABEL_MAP = {
    "GPE": "location",
    "LOC": "location",
    "ORG": "company",
    "MONEY": "money",
    "DATE": "date",
    "PRODUCT": "product",
}

# spaCy's small English model doesn't reliably tag consumer product names
# (iPhone, MacBook, etc.), so we backstop it with a small known-brand list.
_KNOWN_PRODUCTS = ["iphone", "macbook", "ipad", "samsung tv", "airpods", "apple watch"]

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        # Only tok2vec + ner are needed for entity extraction; disabling the
        # tagger/parser/lemmatizer/attribute_ruler roughly halves per-doc cost,
        # which matters once you're running this over tens of thousands of docs.
        _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    return _nlp


def _entities_from_doc(doc, text: str) -> dict:
    entities: dict[str, list[str]] = {"product": [], "location": [], "money": [],
                                       "date": [], "company": [], "order_number": []}

    for ent in doc.ents:
        key = _SPACY_LABEL_MAP.get(ent.label_)
        if key and ent.text not in entities[key]:
            entities[key].append(ent.text)

    lowered = text.lower()
    for product in _KNOWN_PRODUCTS:
        if product in lowered and product not in [p.lower() for p in entities["product"]]:
            entities["product"].append(product)

    for match in ORDER_NUMBER_RE.findall(text):
        if match not in entities["order_number"]:
            entities["order_number"].append(match)

    return entities


def extract_entities(text: str) -> dict:
    return _entities_from_doc(_get_nlp()(text), text)


def extract_entities_batch(texts: list[str], batch_size: int = 256) -> list[dict]:
    """Batched entity extraction via nlp.pipe — much faster than calling
    extract_entities() in a loop once you're past a few thousand documents."""
    docs = _get_nlp().pipe(texts, batch_size=batch_size)
    return [_entities_from_doc(doc, text) for doc, text in zip(docs, texts)]
