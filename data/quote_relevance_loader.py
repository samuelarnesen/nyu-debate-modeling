from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SpeakerType, SpeechData, SplitType
from data.quality_loader import QualityLoader, QualityDataset
from data.scratchpad_quality_debates_loader import ScratchpadQualityDebatesLoader, ScratchpadQualityDebatesDataset
import utils.constants as constants

from typing import Any, Optional

from pydantic import BaseModel
import json
import os
import pickle


class QuoteRelevanceTopicInfo(BaseModel):
    question: str
    a_position: str
    b_position: str


class QuoteRelevanceProcessedBatchItem(BaseModel):
    a_quote_map: dict[str, int]
    b_quote_map: dict[str, int]
    question_info: QuoteRelevanceTopicInfo


class QuoteRelevanceDataset(QualityDataset):
    FILTER_THRESHOLD = 5

    def __init__(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        test_data: list[dict[str, Any]],
        quote_label_file_path: str,
        scratchpad_dataset: ScratchpadQualityDebatesDataset,
    ):
        """Dataset that builds on top of the quality dataset but there are scratchpads added that contain
        the most relevant quotes from the passage"""
        super().__init__(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            override_type=DatasetType.QUOTE_RELEVANCE,
            allow_multiple_positions_per_question=True,
        )
        self.__match_processed_quotes_to_stories(
            quote_label_file_path=quote_label_file_path, scratchpad_dataset=scratchpad_dataset
        )

    def __match_processed_quotes_to_stories(
        self, quote_label_file_path: str, scratchpad_dataset: ScratchpadQualityDebatesDataset
    ):
        def standardize_string(input_string: str):
            return input_string.strip().lower()

        with open(quote_label_file_path, "rb") as f:
            quote_labels = pickle.load(f)

        pairs = []
        for i, item in enumerate(quote_labels):
            question_info = item.question_info
            for j, row in enumerate(self.data[SplitType.TRAIN]):
                positions = [standardize_string(position) for position in row.positions]
                if (
                    standardize_string(row.question) == standardize_string(question_info.question)
                    and standardize_string(question_info.a_position) in positions
                    and standardize_string(question_info.b_position) in positions
                ):
                    pairs.append((item, row))
                    break

        rows_to_use = []
        for item, row in pairs:
            row.speeches = []

            filtered_a_quote_map = {
                quote: score
                for quote, score in filter(lambda x: x[1] > QuoteRelevanceDataset.FILTER_THRESHOLD, item.a_quote_map.items())
            }
            a_scratchpad = "\n\n".join(
                [
                    f"{(i + 1)}. {constants.QUOTE_TAG}{quote}{constants.UNQUOTE_TAG}"
                    for i, quote in enumerate(filter(lambda x: x, filtered_a_quote_map))
                ]
            ).strip()
            row.speeches.append(SpeechData(text="", position=0, speaker_type=SpeakerType.DEBATER, scratchpad=a_scratchpad))

            filtered_b_quote_map = {
                quote: score
                for quote, score in filter(lambda x: x[1] > QuoteRelevanceDataset.FILTER_THRESHOLD, item.b_quote_map.items())
            }

            b_scratchpad = "\n\n".join(
                [
                    f"{(i + 1)}. {constants.QUOTE_TAG}{quote}{constants.UNQUOTE_TAG}"
                    for i, quote in enumerate(filter(lambda x: x, filtered_b_quote_map))
                ]
            ).strip()
            row.speeches.append(SpeechData(text="", position=1, speaker_type=SpeakerType.DEBATER, scratchpad=b_scratchpad))

            if a_scratchpad or b_scratchpad:
                rows_to_use.append(row)

        rows_to_use.extend(scratchpad_dataset.get_data(split=SplitType.TRAIN))

        self.data[SplitType.TRAIN] = rows_to_use
        self.data[SplitType.VAL] = []
        self.data[SplitType.TEST] = []


class QuoteRelevanceLoader(RawDataLoader):
    DEFAULT_QUOTE_LABEL_FILE_PATH = os.environ["SRC_ROOT"] + "data/datasets/quote-relevance/quote-relevance.p"

    @classmethod
    def load(
        cls,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        supplemental_file_paths: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> QuoteRelevanceDataset:
        """Constructs a QuoteRelevanceDataset"""
        quote_label_file_path = (
            supplemental_file_paths.get("quote_label_file_path", QuoteRelevanceLoader.DEFAULT_QUOTE_LABEL_FILE_PATH)
            if supplemental_file_paths
            else QuoteRelevanceLoader.DEFAULT_QUOTE_LABEL_FILE_PATH
        )

        debate_file_path = supplemental_file_paths.get("debate_file_path", None) if supplemental_file_paths else None
        scratchpad_dataset = ScratchpadQualityDebatesLoader.load(full_dataset_filepath=debate_file_path, deduplicate=False)

        train, val, test = QualityLoader.get_splits(
            train_filepath=train_filepath, val_filepath=val_filepath, test_filepath=test_filepath
        )

        return QuoteRelevanceDataset(
            train_data=train,
            val_data=val,
            test_data=val,
            quote_label_file_path=quote_label_file_path,
            scratchpad_dataset=scratchpad_dataset,
        )
