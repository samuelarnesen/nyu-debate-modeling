from data.data import DataRow, RawDataLoader, RawDataset, SplitType

from typing import Any, Optional
import json


class QualityDebatesDataset(RawDataset):
    def __init__(self, train_data: list[str, Any], val_data: list[str, Any], test_data: list[str, Any]):
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__convert_batch_to_rows(val_data),
            SplitType.TEST: self.__convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[DataRow]:
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[DataRow]:
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return [x for x in self.data[split]]

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        return self.data[split][idx % len(self.data[split])]

    def __convert_batch_to_rows(self, batch: list[dict[str, Any]]):
        return [self.__example_to_row(entry) for entry in batch]

    def __example_to_row(self, entry: dict[str, Any]) -> tuple[str, Any]:
        return DataRow(
            background_text=entry["story"],
            question=entry["question"],
            positions=entry["answers"],
            speeches=[speech["text"] for speech in filter(lambda x: x["role"] == "Debater", entry["turns"])],
        )


class QualityDebatesLoader(RawDataLoader):
    @classmethod
    def load(
        cls,
        full_dataset_filepath: str,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
    ) -> QualityDebatesDataset:
        def __get_filtered_rows(file_path: str):
            rows = []
            with open(file_path) as f:
                for line in f.readlines():
                    rows.append(json.loads(line))
            return [row for row in filter(lambda x: len(x["turns"]) > 1, rows)]

        filtered_rows = __get_filtered_rows(file_path=full_dataset_filepath)
        length = len(filtered_rows)
        return QualityDebatesDataset(
            train_data=filtered_rows[0 : int(0.8 * length)],
            val_data=filtered_rows[int(0.8 * length) : int(0.9 * length)],
            test_data=filtered_rows[int(0.9 * length) :],
        )
