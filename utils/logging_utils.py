import logging, datetime, pathlib

def init_logger(name: str, cfg, tag: str = "full-run"):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = pathlib.Path(cfg["log_dir"])
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"log_{tag}_{ts}.txt"

    fmt = "%(asctime)s  %(levelname)-8s %(name)s :: %(message)s"
    handler = logging.FileHandler(log_path, encoding="utf-8")        # timestamped filename :contentReference[oaicite:10]{index=10}
    handler.setFormatter(logging.Formatter(fmt))

    logger = logging.getLogger(name)
    logger.setLevel(cfg.get("log_level", "INFO"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
