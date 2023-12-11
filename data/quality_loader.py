from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SplitType

from typing import Any, Optional
import json
import random


class QualityDataset(RawDataset):
    def __init__(self, train_data: list[dict[str, Any]], val_data: list[dict[str, Any]], test_data: list[dict[str, Any]]):
        """Dataset where each row contains a question and positions from the Quality dataset"""
        super().__init__(DatasetType.QUALITY)
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__convert_batch_to_rows(val_data),
            SplitType.TEST: self.__convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[DataRow]:
        """Returns all the data for a given split"""
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[DataRow]:
        """Returns a subset of the data for a given split"""
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return [x for x in data_to_return]

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        """Returns an individual row in the dataset"""
        return self.data[split][idx % len(self.data[split])]

    def __convert_batch_to_rows(self, batch: list[dict[str, Any]]):
        rows = []
        for entry in batch:
            for i in range(len(entry["questions"])):
                row = self.__example_to_row(entry, i)
                if row:
                    rows.append(row)
        random.shuffle(rows)
        return rows

    def __example_to_row(self, entry: dict[str, Any], question_idx: int) -> tuple[str, Any]:
        question = entry["questions"][question_idx]
        if "gold_label" not in question or "difficult" not in question or question["difficult"] == 0:
            return None
        correct_answer = int(question["gold_label"]) - 1
        incorrect_answer = random.choice([i for i in filter(lambda x: x != correct_answer, range(len(question["options"])))])
        debater_a_correct = random.random() < 0.5
        return DataRow(
            background_text=entry["article"],
            question=question["question"],
            correct_index=0 if debater_a_correct else 1,
            positions=(
                question["options"][correct_answer if debater_a_correct else incorrect_answer],
                question["options"][incorrect_answer if debater_a_correct else correct_answer],
            ),
            story_title=entry["title"],
        )


class QualityLoader(RawDataLoader):
    @classmethod
    def load(
        cls,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        **kwargs,
    ) -> QualityDataset:
        """Constructs a QualityDataset"""
        def __load_individual_file(filepath: str) -> list[str, Any]:
            entries = []
            if filepath:
                with open(filepath) as f:
                    for line in f.readlines():
                        entries.append(json.loads(line))
            return entries

        train_split = __load_individual_file(train_filepath)
        val_split = __load_individual_file(val_filepath)
        test_split = __load_individual_file(test_filepath)
        return QualityDataset(train_data=train_split, val_data=val_split, test_data=test_split)
