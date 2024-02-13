from data.dataset import DataRow, DatasetType, JudgePreferenceDataRow, RawDataLoader, RawDataset, SplitType
from data.quality_loader import QualityLoader
from utils import InputType, InputUtils
import utils.constants as constants

from typing import Any, Optional
import json


class JudgePreferencesDataset(RawDataset):
    def __init__(self, train_data: list[str, Any], val_data: list[str, Any], test_data: list[str, Any]):
        """
        A dataset of judge preferences from a previous best-of-n run. Each row is a pair of speeches with one
        labelled as the chosen speech and the other as the rejected speech.
        """
        super().__init__(DatasetType.JUDGE_PREFERENCES)
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__convert_batch_to_rows(val_data),
            SplitType.TEST: self.__convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[JudgePreferenceDataRow]:
        """Returns all the data for a given split"""
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[JudgePreferenceDataRow]:
        """Returns a subset of the data for a given split"""
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return data_to_return

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> JudgePreferenceDataRow:
        """Returns an individual row in the dataset"""
        return self.data[split][idx % len(self.data[split])]

    def __convert_batch_to_rows(self, train_data: list[tuple[str, str, str]]):
        return [
            JudgePreferenceDataRow(prompt=instruction, chosen=chosen, rejected=rejected)
            for instruction, chosen, rejected in train_data
        ]


class JudgePreferencesLoader(RawDataLoader):
    MIN_GAP = 0.5

    @classmethod
    def load(
        cls, full_dataset_filepath: str | list[str], supplemental_file_paths: Optional[dict[str, str]] = None, **kwargs
    ) -> JudgePreferencesDataset:
        """
        Constructs a JudgePreferencesDataset.

        Params:
            full_dataset_filepath: This is the *prefix* of the files with all the Best-of-N generations.
            supplemental_file_paths: An optional dictionary of paths that could be used to support the creation
                of the dataset. In this case, the relevant one would be quality_file_path.

        Returns:
            A JudgePreferencesDataset where each row has a chosen and a rejected speech.
        """

        train_data = []
        input_texts = InputUtils.read_file_texts(base_path=full_dataset_filepath, input_type=InputType.JSON_TRANSCRIPT)
        for text in input_texts:
            data = json.loads(text)
            for selected in filter(
                lambda x: x["speaker"] in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME],
                data["speeches"],
            ):
                instruction = selected["supplemental"]["prompt"]
                rejected = sorted(selected["supplemental"]["rejected_responses"], key=lambda x: x["preference"])[0]
                if selected["supplemental"]["preference"] - rejected["preference"] > JudgePreferencesLoader.MIN_GAP:
                    selected_speech = (
                        selected["content"]
                        .replace(constants.INVALID_QUOTE_TAG, constants.QUOTE_TAG)
                        .replace(constants.INVALID_UNQUOTE_TAG, constants.UNQUOTE_TAG)
                    )
                    train_data.append((instruction, selected_speech, rejected["speech"]))

        return JudgePreferencesDataset(
            train_data=train_data,
            val_data=[],
            test_data=[],
        )
