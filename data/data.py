from abc import ABC
from enum import Enum
from typing import Any, Optional


class SplitType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class DatasetType(Enum):
    QUALITY = 1
    QUALITY_DEBATES = 2


class Dataset(ABC):
    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[dict[str, Any]]:
        pass

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[str]:
        pass

    def get_example(self, split: SplitType = SplitType.TRAIN) -> dict[str]:
        return self.get_batch(split=split, batch_size=1)[0]


class DataLoader(ABC):
    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        train_filepath: Optional[str] = None,
        validation_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
    ) -> Dataset:
        pass
