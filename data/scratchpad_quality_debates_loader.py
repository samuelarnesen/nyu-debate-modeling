from data.dataset import DataRow, DatasetType, RawDataLoader, SpeechData, SplitType
from data.quality_debates_loader import QualityDebatesLoader, QualityDebatesDataset
from utils import QuoteUtils
import utils.constants as constants

from tqdm import tqdm

from typing import Any, Optional
import os
import pickle
import re


class ScratchpadQualityDebatesDataset(QualityDebatesDataset):
    MINIMUM_QUOTE_LENGTH = 1
    CONTEXT_SIZE = 0
    DEFAULT_SCRATCHPAD_TEXT = "No quotes needed"

    def __init__(self, train_data: list[str, Any], val_data: list[str, Any], test_data: list[str, Any]):
        """Dataset where each row has a question, position, debate transcript (from the human debates) and an
        automatically generated scratchpad continuation for each speech that lists out the quotes used"""
        super().__init__(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            override_type=DatasetType.SCRATCHPAD_QUALITY_DEBATES,
        )
        self._generate_scratchpads()

    def _generate_scratchpads(self) -> None:
        for split in SplitType:
            for row in self.data[split]:
                for speech in row.speeches:
                    self._generate_scratchpad(speech=speech, row=row)

    def _generate_scratchpad(self, speech: SpeechData, row: DataRow) -> Optional[str]:
        original_quotes = QuoteUtils.extract_quotes(speech.text)
        contexts = [
            QuoteUtils.extract_quote_context(
                quote_text=quote,
                background_text=row.background_text,
                context_size=ScratchpadQualityDebatesDataset.CONTEXT_SIZE,
            )
            for quote in filter(
                lambda x: len(x.split()) >= ScratchpadQualityDebatesDataset.MINIMUM_QUOTE_LENGTH, original_quotes
            )
        ]
        speech.scratchpad = (
            "\n\n".join(
                [
                    f"{(i + 1)}. {constants.QUOTE_TAG}{context}{constants.UNQUOTE_TAG}"
                    for i, context in enumerate(filter(lambda x: x, contexts))
                ]
            )
            if contexts
            else ScratchpadQualityDebatesDataset.DEFAULT_SCRATCHPAD_TEXT
        )


class ScratchpadQualityDebatesLoader(RawDataLoader):
    DEFAULT_PICKLE_PATH = (
        os.environ[constants.SRC_ROOT] + "data/datasets/scratchpad-quality-debates/scratchpad-quality-debates.p"
    )

    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        deduplicate: bool = False,
        **kwargs,
    ) -> ScratchpadQualityDebatesDataset:
        """Constructs a ScratchpadQualityDebatesDataset"""
        if os.path.exists(ScratchpadQualityDebatesLoader.DEFAULT_PICKLE_PATH):
            with open(ScratchpadQualityDebatesLoader.DEFAULT_PICKLE_PATH, "rb") as f:
                return pickle.load(f)
        full_dataset_filepath = full_dataset_filepath or QualityDebatesLoader.DEFAULT_FILE_PATH
        train, val, test = QualityDebatesLoader.get_splits(file_path=full_dataset_filepath, deduplicate=deduplicate)
        return ScratchpadQualityDebatesDataset(
            train_data=train,
            val_data=val,
            test_data=test,
        )
