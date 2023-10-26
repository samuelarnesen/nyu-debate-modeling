from abc import ABC
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class SplitType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class DatasetType(Enum):
    QUALITY = 1
    QUALITY_DEBATES = 2
    JUDGE_PREFERENCES = 3


class SpeakerType(Enum):
    DEBATER = 1
    JUDGE = 2


class SpeechData(BaseModel):
    text: str
    position: int
    speaker_type: SpeakerType


class DataRow(BaseModel):
    background_text: str
    question: Optional[str]
    positions: Optional[tuple[str, str]]
    speeches: Optional[list[SpeechData]]
    correct_index: Optional[int]


class JudgePreferenceDataRow(BaseModel):
    prompt: str
    chosen: str
    rejected: str


class RawDataset(ABC):
    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[tuple[str, Any]]:
        pass

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[tuple[str, Any]]:
        pass

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        pass


class RawDataLoader(ABC):
    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        train_filepath: Optional[str] = None,
        validation_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
    ) -> RawDataset:
        pass
