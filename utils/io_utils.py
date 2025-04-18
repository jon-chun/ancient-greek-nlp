from pathlib import Path, PurePath
import json, os

def prepare_dirs(cfg):
    Path(cfg["out_dir"], "figs").mkdir(parents=True, exist_ok=True)
    Path(cfg["intermediate_dir"]).mkdir(parents=True, exist_ok=True)

def write_jsonl(obj, name, cfg):
    path = Path(cfg["out_dir"], name)
    with open(path, "w", encoding="utf‑8") as f:
        for record in obj:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_txt_summary(topics, sents, absa, cfg):
    with open(Path(cfg["out_dir"], "summary.txt"), "w", encoding="utf‑8") as f:
        f.write("# Topic Models\n")
        for tm in topics:
            f.write(json.dumps(tm, indent=2, ensure_ascii=False) + "\n\n")
        f.write("# Sentiment\n")
        f.write(json.dumps(sents[:20], indent=2, ensure_ascii=False) + "\n...\n")
        f.write("# ABSA Samples\n")
        f.write(json.dumps(absa[:20], indent=2, ensure_ascii=False) + "\n")
