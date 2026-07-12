import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("DataMindAI")
logger.setLevel(logging.INFO)

if not logger.handlers:

    file_handler = logging.FileHandler(
        "logs/datamind.log",
        mode="a",
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Prevent duplicate logs in terminal
    logger.propagate = False