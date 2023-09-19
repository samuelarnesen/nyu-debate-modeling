import argparse
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
        return parser.parse_args()
