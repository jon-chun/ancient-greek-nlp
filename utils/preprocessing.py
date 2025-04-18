"""
Tokenise, segment, POS‑tag and lemmatise Ancient‑Greek texts.

Relies on:
    • CLTK 2.2+  (Ancient‑Greek pipeline)  :contentReference[oaicite:2]{index=2}
    • stanza 1.8.1 for sentence segmentation  :contentReference[oaicite:3]{index=3}
    • spaCy‑grc (optional)  :contentReference[oaicite:4]{index=4}
"""

from pathlib import Path
from cltk import NLP as CLTK_NLP
import stanza, regex as re

def _strip_diacritics(text): ...

def run(cfg):
    text = Path(cfg["raw_corpus"]).read_text(encoding="utf‑8")
    if cfg["remove_diacritics"]:
        text = _strip_diacritics(text)

    # sentence splitting ----------------------------------------------------
    if cfg["sentence_segmenter"] == "stanza":
        stanza.download("grc", processors="tokenize")
        nlp_sent = stanza.Pipeline(lang="grc", processors="tokenize")
        sents = [s.text for s in nlp_sent(text).sentences]
    else:  # regex fallback
        sents = re.split(r"(?u)[··.;;]\s*", text)

    # lemmatisation ---------------------------------------------------------
    if cfg["lemmatizer"] == "cltk":
        nlp = CLTK_NLP(language="ancient_greek")
        docs = [" ".join([t.lemma_ for t in nlp(s).tokens
                          if len(t.string) >= cfg["min_token_len"]])
                for s in sents]
    else:  # spaCy‑grc
        import spacy
        nlp = spacy.load("el_core_news_sm")
        docs = [" ".join([t.lemma_ for t in nlp(s)
                          if not (t.is_punct or t.is_space)
                          and len(t.lemma_) >= cfg["min_token_len"]])
                for s in sents]

    return docs
