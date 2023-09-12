from data.data import DataLoader
from data.loaders.quality_loader import QualityLoader

from enum import Enum
from typing import Type

class DatasetType(Enum):
    QUALITY = 1

class LoaderUtils:
    @classmethod
    def get_loader_type(cls, dataset_type: DatasetType) -> Type[DataLoader]:
        if dataset_type == DatasetType.QUALITY:
            return QualityLoader
        raise Exception(f"Loader {dataset_type} not found")