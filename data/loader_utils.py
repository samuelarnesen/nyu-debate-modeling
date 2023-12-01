from data.dataset import RawDataLoader, DatasetType
from data.annotated_quality_debates_loader import AnnotatedQualityDebatesLoader
from data.judge_preferences_loader import JudgePreferencesLoader
from data.quality_loader import QualityLoader
from data.quality_debates_loader import QualityDebatesLoader

from enum import Enum
from typing import Type


class LoaderUtils:
    @classmethod
    def get_loader_type(cls, dataset_type: DatasetType) -> Type[RawDataLoader]:
        if dataset_type == DatasetType.QUALITY:
            return QualityLoader
        elif dataset_type == DatasetType.QUALITY_DEBATES:
            return QualityDebatesLoader
        elif dataset_type == DatasetType.JUDGE_PREFERENCES:
            return JudgePreferencesLoader
        elif dataset_type == DatasetType.ANNOTATED_QUALITY_DEBATES:
            return AnnotatedQualityDebatesLoader
        raise Exception(f"Loader {dataset_type} not found")
