from utils import topic_modeling, preprocessing

def test_lda_runs(tiny_cfg):
    docs = preprocessing.run(tiny_cfg)
    res  = topic_modeling.run(tiny_cfg, docs)
    assert res and res[0]["model"] == "lda"
    assert len(res[0]["topics"]) == tiny_cfg["topic_models"][0]["k_topics"]
