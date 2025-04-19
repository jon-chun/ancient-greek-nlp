from utils import absa, sentiment, preprocessing

def test_absa_dependency(tiny_cfg):
    docs = preprocessing.run(tiny_cfg)
    sentiments = [{"label": "NEUTRAL"}] * len(docs)
    absa_rows = absa.run(tiny_cfg, docs, sentiments)
    assert absa_rows and isinstance(absa_rows[0]["aspects"], list)
