#!/usr/bin/env python3
"""
Pipeline driver for Ancient Greek NLP tasks.

Usage:
    python ancient_greek_nlp.py --config config.yml
"""

import argparse, pathlib, importlib
from ruamel.yaml import YAML  # actively maintained YAML lib 0.18.x  🠚 :contentReference[oaicite:1]{index=1}


from utils.logging_utils import init_logger
LOGGER = init_logger(__name__, "full-run", cfg)


yaml = YAML(typ="safe")

def load_config(path):
    with open(path, "r", encoding="utf‑8") as f:
        return yaml.load(f)

def main(cfg):
    # lazy imports to shorten cold‑start time
    from utils import (
        preprocessing,
        topic_modeling,
        sentiment,
        absa,
        visualization,
        io_utils,
    )

    io_utils.prepare_dirs(cfg)

    docs = preprocessing.run(cfg)                  # list[str] of cleaned verses
    topic_results = topic_modeling.run(cfg, docs)  # dict
    sent_results  = sentiment.run(cfg, docs)       # list[dict]
    absa_results  = absa.run(cfg, docs, sent_results)  # list[dict]

    # persist
    io_utils.write_jsonl(topic_results, "topics.jsonl", cfg)
    io_utils.write_jsonl(sent_results,  "sentiment.jsonl", cfg)
    io_utils.write_jsonl(absa_results,  "absa.jsonl", cfg)
    io_utils.write_txt_summary(topic_results, sent_results, absa_results, cfg)

    # optional visualisations
    if cfg["save_figs"]:
        visualization.make_topic_figs(topic_results, cfg)
        visualization.make_sentiment_figs(sent_results, cfg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="config.yml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    main(cfg)
