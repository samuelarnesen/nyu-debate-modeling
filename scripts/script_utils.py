import argparse
import logging
import os
import sys


class ScriptUtils:
    # Add the parent directory to sys.path
    @classmethod
    def set_parent_as_path(cls):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    @classmethod
    def get_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--local", action="store_true", default=False)
        parser.add_argument("--num_iters", type=int, default=1_000)
        parser.add_argument("--log_level", type=str, default="INFO")
        parser.add_argument("--configuration", type=str, default="")
        args = parser.parse_args()
        ScriptUtils.set_log_level(args)
        return args

    @classmethod
    def set_log_level(cls, args) -> None:
        requested = args.log_level.lower()
        specified = None
        if requested == "debug":
            specified = logging.DEBUG
        elif requested == "info":
            specified == logging.INFO
        elif requested == "warn":
            specified == logging.WARN
        elif requested == "error":
            specified == logging.ERROR
        else:
            raise Exception(f"Request log level {requested} is not eligible")

        os.environ["LOG_LEVEL"] = str(specified)
