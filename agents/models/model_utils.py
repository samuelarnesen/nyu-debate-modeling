from agents.model import Model
from agents.models.llama_model import LlamaModel
from agents.models.random_model import RandomModel

from enum import Enum
from typing import Optional


class ModelType(Enum):
    RANDOM = 1
    LLAMA = 2


class ModelUtils:
    @classmethod
    def instantiate_model(cls, model_type: ModelType, file_path: Optional[str] = None, is_debater: bool = True) -> Model:
        if model_type == ModelType.RANDOM:
            return RandomModel()
        elif model_type == ModelType.LLAMA:
            return LlamaModel(file_path=file_path, is_debater=is_debater)
        raise Exception(f"Model {model_type} not found")
