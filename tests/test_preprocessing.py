from utils import preprocessing

def test_tokenisation_and_lemmatisation(tiny_cfg):
    docs = preprocessing.run(tiny_cfg)
    assert len(docs) == 3
    assert any("γεννάω" in d for d in docs)   # lemma check
