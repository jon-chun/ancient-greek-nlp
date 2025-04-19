import pytest, pathlib, yaml, datetime
from utils import io_utils

@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    """Minimal config pointing to a 3‑line toy corpus for fast tests."""
    tmpdir = tmp_path_factory.mktemp("data")
    corpus_file = tmpdir / "toy.txt"
    corpus_file.write_text(
        "ΒΙΒΛΟΣ γενέσεως Ἰησοῦ.\nἈβραὰμ ἐγέννησεν Ἰσαάκ.\nἸωνᾶς ἦν προφήτης.",
        encoding="utf-8")
    cfg = {
        "raw_corpus": str(corpus_file),
        "intermediate_dir": str(tmpdir / "inter"),
        "out_dir": str(tmpdir / "out"),
        "log_dir": str(tmpdir / "log"),
        # ...override only what matters for tests...
        "remove_diacritics": False,
        "sentence_segmenter": "regex",
        "lemmatizer": "cltk",
        "min_token_len": 2,
        "topic_models": [{"name": "lda_gensim", "k_topics": 2}],
        "sentiment_method": "lexicon",
        "absa_method": "dependency",
    }
    return cfg
