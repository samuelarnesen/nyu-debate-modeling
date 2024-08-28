import utils.constants as constants

import pandas as pd

from enum import Enum
from io import StringIO
from typing import Callable, Union
import json
import os
import re


class InputType(Enum):
    TEXT_TRANSCRIPT = ("txt", os.environ[constants.INPUT_ROOT] + "/transcripts", lambda x: x)
    JSON_TRANSCRIPT = ("json", os.environ[constants.INPUT_ROOT] + "/transcripts", lambda x: json.loads(x))
    JSON_LIST = ("jsonl", os.environ[constants.INPUT_ROOT] + "/transcripts", lambda x: [json.load(y) for y in x])
    RUN = ("csv", os.environ[constants.INPUT_ROOT] + "/runs", lambda x: pd.read_csv(StringIO(x)))

    def __init__(self, extension: str, location: str, load: Callable):
        self.extension = extension
        self.location = location
        self.load = load


def get_full_filepath(base_path: str, input_type: InputType) -> str:
    """
    Given either a full path through the prefix or just the prefix, return the full path through the prefix.
    For example, base_path='12345' -> /path/to/source/root/outputs/transcripts/123455
    """
    return base_path if "/" in base_path else f"{input_type.location}/{base_path}"


def read_file_texts(
    base_path: str | list[str],
    input_type: InputType = InputType.TEXT_TRANSCRIPT,
    include_full_file_path: bool = False,
    should_load: bool = False,
) -> list[str] | list[tuple[str, str]]:
    """
    Reads transcript generated by the run_debate script. All the files are named using the following
    convention: base_path_{round_number}_{batch_number}.txt.

    Params:
        base_path: the directory + file prefix that all the transcripts share. This can be a list if there are multiple
            sets of file prefixes that one wants to aggregate into one dataset
        extension: "txt" if the files are txt files or "json"
        include_full_file_path: if True, it returns a list of (file_text, file_path) tuples. If False, it returns a list of file_texts
        should_load: if True, the the file text gets turned into the format specific to the input type

    Returns:
        file_texts: A list of transcript contents.
    """

    def get_idxs_of_file(file_name: str) -> tuple[int, int]:
        suffix_pattern = "_(\d+)_(\d+)\." + input_type.extension
        suffix = re.search(suffix_pattern, file_name)
        if suffix:
            return suffix.group(1), suffix.group(2)
        return -1, -1

    def sort_files_by_extension(file_names: list[str]) -> list[str]:
        files_with_idxs = [(file_name, get_idxs_of_file(file_name)) for file_name in file_names]
        files_with_idxs = sorted(files_with_idxs, key=lambda x: int(x[1][1]))
        files_with_idxs = sorted(files_with_idxs, key=lambda x: int(x[1][0]))
        return [x[0] for x in files_with_idxs]

    def list_files_with_prefix(directory: str, prefix: str):
        files = os.listdir(directory)
        matching_files = [
            f"{directory}/{file}"
            for file in filter(lambda x: x.startswith(prefix) and x.endswith(input_type.extension), files)
        ]
        return sort_files_by_extension(matching_files)

    if isinstance(base_path, list):
        input_texts = []
        for path in base_path:
            input_texts += read_file_texts(
                base_path=path, input_type=input_type, include_full_file_path=include_full_file_path
            )
        return input_texts

    directory = input_type.location if "/" not in base_path else "/".join(base_path.split("/")[:-1])
    prefix = base_path if "/" not in base_path else base_path.split("/")[-1]

    eligible_files = list_files_with_prefix(directory=directory, prefix=prefix)

    file_texts = []
    for file_name in eligible_files:
        if os.path.exists(file_name):
            with open(file_name) as f:
                text = f.read()
                if should_load:
                    text = input_type.load(text)
                file_texts.append((text, file_name) if include_full_file_path else text)

    return file_texts
