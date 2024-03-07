from data.dataset import RawDataLoader, DatasetType
from data.annotated_quality_debates_loader import AnnotatedQualityDebatesLoader
from data.judge_preferences_loader import JudgePreferencesLoader
from data.scratchpad_quality_debates_loader import ScratchpadQualityDebatesLoader
from data.quality_loader import QualityLoader
from data.quality_debates_loader import QualityConsultancyLoader, QualityDebatesLoader
from data.quality_judging_loader import QualityJudgingLoader
from data.quote_relevance_loader import QuoteRelevanceLoader

from enum import Enum
from typing import Type


class LoaderUtils:
    @classmethod
    def get_loader_type(cls, dataset_type: DatasetType) -> Type[RawDataLoader]:
        """Returns the class associated with the inputted DatasetType"""
        if dataset_type == DatasetType.QUALITY:
            return QualityLoader
        elif dataset_type == DatasetType.QUALITY_DEBATES:
            return QualityDebatesLoader
        elif dataset_type == DatasetType.JUDGE_PREFERENCES:
            return JudgePreferencesLoader
        elif dataset_type == DatasetType.ANNOTATED_QUALITY_DEBATES:
            return AnnotatedQualityDebatesLoader
        elif dataset_type == DatasetType.SCRATCHPAD_QUALITY_DEBATES:
            return ScratchpadQualityDebatesLoader
        elif dataset_type == DatasetType.QUOTE_RELEVANCE:
            return QuoteRelevanceLoader
        elif dataset_type == DatasetType.JUDGING_PROBE:
            return QualityJudgingLoader
        elif dataset_type == DatasetType.QUALITY_CONSULTANCY:
            return QualityConsultancyLoader

        raise Exception(f"Loader {dataset_type} not found")
