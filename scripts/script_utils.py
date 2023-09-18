import sys
import os


class ScriptUtils:
    # Add the parent directory to sys.path
    @classmethod
    def set_parent_as_path(cls):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
