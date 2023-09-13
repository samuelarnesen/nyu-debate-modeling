from agents.model import Model
from agents.models.random_model import RandomModel

from enum import Enum
from typing import Optional


class ModelType(Enum):
    RANDOM = 1


class ModelUtils:
    @classmethod
    def instantiate_model(cls, model_type: ModelType, file_path: Optional[str] = None) -> Model:
        if model_type == ModelType.RANDOM:
            return RandomModel()
        raise Exception(f"Model {model_type} not found")
