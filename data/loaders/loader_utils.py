from data.data import DataLoader, DatasetType
from data.loaders.quality_loader import QualityLoader
from data.loaders.quality_debates_loader import QualityDebatesLoader

from enum import Enum
from typing import Type


class LoaderUtils:
    @classmethod
    def get_loader_type(cls, dataset_type: DatasetType) -> Type[DataLoader]:
        if dataset_type == DatasetType.QUALITY:
            return QualityLoader
        elif dataset_type == DatasetType.QUALITY_DEBATES:
            return QualityDebatesLoader
        raise Exception(f"Loader {dataset_type} not found")
