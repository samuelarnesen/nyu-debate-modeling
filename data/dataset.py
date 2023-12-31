from abc import ABC
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    dataset_type: str
    full_dataset_file_path: Optional[str] = None
    train_file_path: Optional[str] = None
    val_file_path: Optional[str] = None
    test_file_path: Optional[str] = None
    supplemental_file_paths: dict[str, str] = {}
    split_type: str = "train"


class SplitType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class DatasetType(Enum):
    QUALITY = 1
    QUALITY_DEBATES = 2
    JUDGE_PREFERENCES = 3
    ANNOTATED_QUALITY_DEBATES = 4
    SCRATCHPAD_QUALITY_DEBATES = 5
    QUOTE_RELEVANCE = 6


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
    percents: Optional[dict[AnnotationTag | str, float]]
    percentiles: Optional[dict[AnnotationTag | str, float]]


class SpeechData(BaseModel):
    text: str
    position: int
    speaker_type: SpeakerType
    supplemental_file_paths: Optional[dict[str, str]]
    scratchpad: Optional[str]
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
        """Fetches all the data for a given split of the data"""
        pass

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[tuple[str, Any]]:
        """Gets a subset of the data"""
        pass

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        """Returns an individual row at the specified index"""
        pass

    def get_dataset_type(self):
        """Gets the name of the dataset"""
        return self.dataset_type


class RawDataLoader(ABC):
    @classmethod
    def load(
        cls,
        full_dataset_filepath: Optional[str] = None,
        train_filepath: Optional[str] = None,
        validation_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        supplemental_file_paths: Optional[str] = None,
    ) -> RawDataset:
        """Constructs a dataset"""
        pass
