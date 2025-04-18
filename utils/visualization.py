import matplotlib.pyplot as plt            # matplotlib 3.9.x :contentReference[oaicite:9]{index=9}
import seaborn as sns                      # seaborn 0.13 (2024)

def make_topic_figs(topic_results, cfg):
    for tr in topic_results:
        if tr["model"] == "lda":
            labels = [f"T{i}" for i,_ in enumerate(tr["topics"])]
            weights = [1]*len(labels)
            plt.figure()
            sns.barplot(x=labels, y=weights)
            plt.title("LDA Topic Index")
            plt.savefig(f"{cfg['out_dir']}/figs/lda_topics.png", dpi=cfg["fig_dpi"])
            plt.close()

def make_sentiment_figs(sent_results, cfg):
    polarities = [r[0]["label"] for r in sent_results]
    sns.countplot(x=polarities)
    plt.title("Sentiment distribution")
    plt.savefig(f"{cfg['out_dir']}/figs/sentiment_counts.png", dpi=cfg["fig_dpi"])
    plt.close()
