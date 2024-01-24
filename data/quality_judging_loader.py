from data.dataset import DataRow, DatasetType, JudgingProbeDataRow, RawDataLoader, RawDataset, SplitType
from data.quality_loader import QualityLoader
from utils import InputUtils
import utils.constants as constants

import torch

from typing import Any, Optional
import base64
import io
import json


class QualityJudgingDataset(RawDataset):
    def __init__(self, train_data: list[str, Any], val_data: list[str, Any], test_data: list[str, Any]):
        """
        A dataset of judge internal representations, mapped to a target (whether it corresponds to the correct side).
        """
        super().__init__(DatasetType.JUDGING_PROBE)
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__convert_batch_to_rows(val_data),
            SplitType.TEST: self.__convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[JudgingProbeDataRow]:
        """Returns all the data for a given split"""
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[JudgingProbeDataRow]:
        """Returns a subset of the data for a given split"""
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return data_to_return

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> JudgingProbeDataRow:
        """Returns an individual row in the dataset"""
        return self.data[split][idx % len(self.data[split])]

    def __convert_batch_to_rows(self, train_data: list[tuple[torch.tensor, torch.tensor]]):
        return [
            JudgingProbeDataRow(internal_representation=internal_representation, target=target)
            for internal_representation, target in train_data
        ]


class QualityJudgingLoader(RawDataLoader):
    @classmethod
    def load(
        cls, full_dataset_filepath: str, supplemental_file_paths: Optional[dict[str, str]] = None, **kwargs
    ) -> QualityJudgingDataset:
        """
        Constructs a QualityJudgingDataset.

        Params:
            full_dataset_filepath: This is the *prefix* of the files with all the stored internal representations
            supplemental_file_paths: An optional dictionary of paths that could be used to support the creation
                of the dataset. In this case, the relevant one would be quality_file_path.

        Returns:
            A QualityJudgingDataset where each row has an internal representation tensor and a target winning percentage
        """

        # move this to the quality dataset
        def get_original_data_row(data: dict[Any, Any], dataset: RawDataset) -> DataRow:
            debate_identifier = data["metadata"]["debate_identifier"]
            question = data["metadata"]["question"]
            story_title = debate_identifier.replace("_" + question, "")
            for row in dataset.get_data(split=SplitType.TRAIN):
                if row.story_title == story_title and row.question == question:
                    return row
            raise Exception(f"A row with title {story_title} and question {question} could not be found in the dataset")

        quality_filepath = (supplemental_file_paths or {}).get("quality_file_path", QualityLoader.DEFAULT_TRAIN_PATH)
        quality_dataset = QualityLoader.load(full_dataset_filepath=quality_filepath)

        data = []
        input_texts = InputUtils.read_file_texts(base_path=full_dataset_filepath, extension="json")
        for text in input_texts:
            data = json.loads(text)
            row = get_original_data_row(data=data, dataset=quality_dataset)
            for speech in filter(
                lambda x: x["speaker"] == constants.DEFAULT_JUDGE_NAME and x["supplemental"]["internal_representations"],
                data["speeches"],
            ):
                internal_representations = speech["supplemental"]["internal_representations"]
                relevant_internal_representations = [internal_representations[-16], internal_representations[-1]]
                decoded_tensors = [base64.b64decode(rep) for rep in relevant_internal_representations]
                buffers = [io.BytesIO(decoded_tensor) for decoded_tensor in decoded_tensors]
                loaded_tensors = [torch.load(buffer) for buffer in buffers]
                x = torch.cat(loaded_tensors, dim=0)
                y = torch.tensor([1, 0] if row.correct_index == 0 else [0, 1]).float()
                data.append(x, y)

        return QualityJudgingDataset(
            train_data=data[0 : int(0.8 * len(data))],
            val_data=data[int(0.8 * len(data)) :],
            test_data=[],
        )
