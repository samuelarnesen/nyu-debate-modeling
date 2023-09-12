import sys
import os


# Add the parent directory to sys.path
def set_parent_as_path():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
