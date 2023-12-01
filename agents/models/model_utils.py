from agents.models.model import Model
from agents.models.deterministic_model import DeterministicModel
from agents.models.llama_model import LlamaModel
from agents.models.openai_model import OpenAIModel
from agents.models.random_model import RandomModel

from enum import Enum
from typing import Optional


class ModelType(Enum):
    RANDOM = 1
    LLAMA = 2
    DETERMINISTIC = 3
    OPENAI = 4
    OFFLINE = 5
    HUMAN = 6


class ModelUtils:
    @classmethod
    def instantiate_model(
        cls,
        alias: str,
        model_type: ModelType,
        file_path: Optional[str] = None,
        is_debater: bool = True,
        speeches: Optional[list[str]] = None,
        greedy: bool = False,
    ) -> Model:
        if model_type == ModelType.RANDOM:
            return RandomModel(alias=alias, is_debater=is_debater)
        elif model_type == ModelType.LLAMA:
            return LlamaModel(alias=alias, file_path=file_path, is_debater=is_debater, greedy=greedy)
        elif model_type == ModelType.DETERMINISTIC:
            return DeterministicModel(alias=alias, is_debater=is_debater)
        elif model_type == ModelType.OPENAI:
            return OpenAIModel(alias=alias, is_debater=is_debater)
        elif model_type == ModelType.OFFLINE:
            raise Exception("Offline model cannot be directly instantiated")
        elif model_type == ModelType.HUMAN:
            raise Exception("Human model cannot be directly instantiated")
        raise Exception(f"Model {model_type} not found")
