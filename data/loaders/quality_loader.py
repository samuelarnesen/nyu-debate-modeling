from data.data import DataLoader, Dataset, SplitType

from typing import Any
import json


class QualityDataset(Dataset):
    def __init__(self, train_data: list[str, Any], val_data: list[str, Any], test_data: list[str, Any]):
        self.data = {SplitType.TRAIN: train_data, SplitType.VAL: val_data, SplitType.TEST: test_data}
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[str]:
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[str]:
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return [x["article"] for x in data_to_return]


class QualityLoader(DataLoader):
    @classmethod
    def load(cls, train_filepath: str, val_filepath: str, test_filepath: str) -> QualityDataset:
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
