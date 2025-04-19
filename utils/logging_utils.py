import logging, datetime, pathlib

def init_logger(name: str, tag: str, cfg):
    ts   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = pathlib.Path(cfg["log_dir"])
    path.mkdir(exist_ok=True)
    file = path / f"log_{tag}_{ts}.txt"

    handler = logging.FileHandler(file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s %(name)s :: %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(cfg.get("log_level", "INFO"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
