"""
Aspect‑Based Sentiment through:
  • rule‑based dependency extraction  (default)
  • or LLM prompt (if cfg["absa_method"] == "llm")
"""

import stanza, collections
from transformers import pipeline

def _dependency_absa(cfg, docs, sent_results):
    stanza.download("grc", processors="tokenize,pos,depparse,ner")
    nlp = stanza.Pipeline(lang="grc", processors="tokenize,pos,depparse,ner")
    absa_rows = []
    for sent, sent_meta in zip(docs, sent_results):
        doc = nlp(sent)
        aspects = [e.text for e in doc.entities if e.type in cfg["absa_entity_types"]]
        opinions = [w.text for w in doc.iter_words() if w.upos == "ADJ"]
        absa_rows.append({"sentence": sent,
                          "aspects": aspects,
                          "opinions": opinions,
                          "sentiment": sent_meta})
    return absa_rows

def _llm_absa(cfg, docs):
    llm = pipeline("text-generation", model="gpt-4o", api_key="...")
    ...

def run(cfg, docs, sent_results):
    if cfg["absa_method"] == "llm":
        return _llm_absa(cfg, docs)
    else:
        return _dependency_absa(cfg, docs, sent_results)
