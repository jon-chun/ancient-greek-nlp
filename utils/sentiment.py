"""
Option 1: Lexicon translate‑score
Option 2: Multilingual transformer fine‑tune / zero‑shot
"""

from transformers import pipeline          # transformers 4.51.x :contentReference[oaicite:8]{index=8}
import json, pathlib

def _lexicon_score(docs):
    # naive: translate lemmas via LSJ JSON + VADER
    ...

def _transformer_score(docs, model_name):
    clf = pipeline("sentiment-analysis", model=model_name, truncation=True)
    return [clf(d) for d in docs]

def run(cfg, docs):
    if cfg["sentiment_method"] == "lexicon":
        return _lexicon_score(docs)
    else:
        return _transformer_score(docs, cfg["sentiment_model"])
