from utils import sentiment, preprocessing

def test_dummy_sentiment(tiny_cfg):
    tiny_cfg["sentiment_method"] = "lexicon"
    docs = preprocessing.run(tiny_cfg)
    scores = sentiment.run(tiny_cfg, docs)
    assert len(scores) == len(docs)
