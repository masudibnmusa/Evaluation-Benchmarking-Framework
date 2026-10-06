import logging
import os
import sys

_configured = False


def get_logger(name: str = "llm_eval") -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,  # keep stdout clean for reports
        )
        _configured = True
    return logging.getLogger(name)