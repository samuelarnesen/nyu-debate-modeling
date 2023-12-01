from abc import ABC
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel


class SplitType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class DatasetType(Enum):
    QUALITY = 1
    QUALITY_DEBATES = 2
    JUDGE_PREFERENCES = 3
    ANNOTATED_QUALITY_DEBATES = 4


class SpeakerType(Enum):
    DEBATER = 1
    JUDGE = 2


class AnnotationTag(Enum):
    QUOTE = 0
    SUMMARY = 1
    REFUTATION = 2
    ANALYSIS = 3
    REPLY = 4
    FLOURISH = 5
    FRAMING = 6
    STATEMENT = 7
    LOGIC = 8
    Q_CONTEXT = 9
    POSITION = 10
    OOB_QUOTE = 11
    PROMISE = 12


class AnnotationBracket(Enum):
    HIGH = 1
    LOW = 2
    NEUTRAL = 3


class AnnotationData(BaseModel):
    percents: Optional[dict[Union[AnnotationTag, str], float]]
    percentiles: Optional[dict[Union[AnnotationTag, str], float]]


class SpeechData(BaseModel):
    text: str
    position: int
    speaker_type: SpeakerType
    annotation: Optional[AnnotationData]


class DataRow(BaseModel):
    background_text: str
    question: Optional[str]
    positions: Optional[tuple[str, str]]
    speeches: Optional[list[SpeechData]]
    correct_index: Optional[int]
    debate_id: Optional[str]
    story_title: Optional[str]


class JudgePreferenceDataRow(BaseModel):
    prompt: str
    chosen: str
    rejected: str


class RawDataset(ABC):
    def __init__(self, dataset_type: DatasetType):
        self.dataset_type = dataset_type

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[tuple[str, Any]]:
        pass

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[tuple[str, Any]]:
        pass

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        pass

    def get_dataset_type(self):
        return self.dataset_type


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
