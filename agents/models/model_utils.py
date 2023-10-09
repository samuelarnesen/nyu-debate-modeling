from agents.model import Model
from agents.models.deterministic_model import DeterministicModel
from agents.models.llama_model import LlamaModel
from agents.models.random_model import RandomModel

from enum import Enum
from typing import Optional


class ModelType(Enum):
    RANDOM = 1
    LLAMA = 2
    DETERMINISTIC = 3


class ModelUtils:
    @classmethod
    def instantiate_model(
        cls, alias: str, model_type: ModelType, file_path: Optional[str] = None, is_debater: bool = True
    ) -> Model:
        if model_type == ModelType.RANDOM:
            return RandomModel(alias=alias, is_debater=is_debater)
        elif model_type == ModelType.LLAMA:
            return LlamaModel(alias=alias, file_path=file_path, is_debater=is_debater)
        elif model_type == ModelType.DETERMINISTIC:
            return DeterministicModel(alias=alias, is_debater=is_debater)
        raise Exception(f"Model {model_type} not found")
