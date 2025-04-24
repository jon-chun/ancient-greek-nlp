"""
Two interchangeable engines: (A) gensim LDA, (B) BERTopic
"""

from gensim import corpora, models         # Gensim 4.5.x :contentReference[oaicite:5]{index=5}
from bertopic import BERTopic              # BERTopic 0.16.x ¬ :contentReference[oaicite:6]{index=6}
from sentence_transformers import SentenceTransformer  # Apr‑2025 build :contentReference[oaicite:7]{index=7}

def _run_lda(docs, k):
    dictionary = corpora.Dictionary([d.split() for d in docs])
    corpus = [dictionary.doc2bow(d.split()) for d in docs]
    lda = models.LdaModel(corpus=corpus,
                          id2word=dictionary,
                          num_topics=k,
                          alpha="auto",
                          iterations=400)
    return {"model": "lda", "topics": lda.print_topics(num_words=10)}

def _run_bertopic(docs, embed_name):
    embedder = SentenceTransformer(embed_name)
    topic_model = BERTopic(embedding_model=embedder)
    topics, _ = topic_model.fit_transform(docs)
    return {"model": "bertopic",
            "topic_info": topic_model.get_topic_info().to_dict("records")}

def run(cfg, docs):
    results = []
    for tm in cfg["topic_models"]:
        if tm["name"] == "lda_gensim":
            results.append(_run_lda(docs, tm["k_topics"]))
        elif tm["name"] == "bertopic":
            results.append(_run_bertopic(docs, tm["embedding_model"]))
    return results
SS