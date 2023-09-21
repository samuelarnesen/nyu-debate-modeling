import logging
import os


class LoggerUtils:
    @classmethod
    def get_default_logger(cls, name: str, log_level=None):
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            log_level = log_level or LoggerUtils.get_log_level()

            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(formatter)

            logger.setLevel(log_level)
            logger.addHandler(stream_handler)
        return logger

    @classmethod
    def get_log_level(cls):
        if "LOG_LEVEL" in os.environ:
            requested = os.environ["LOG_LEVEL"]
            for level in filter(lambda x: requested == str(x), [logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR]):
                return level
        return logging.INFO
