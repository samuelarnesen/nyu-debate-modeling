from agents.models.model import Model
from agents.models.deterministic_model import DeterministicModel
from agents.models.llm_model import LlamaModel, MistralModel
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
    MISTRAL = 7


class ModelUtils:
    @classmethod
    def instantiate_model(
        cls,
        alias: str,
        model_type: ModelType,
        file_path: Optional[str] = None,
        is_debater: bool = True,
        greedy: bool = False,
    ) -> Model:
        """
        Builds a model using the given inputs.

        Args:
            alias: A unique string to identify the model for metrics and deduplication
            model_type: The kind of model to be instantiated.
            file_path: If the model has to be loaded locally, this should contain the path to that file.
                This is used only for the LLModel at the moment.
            is_debater: Boolean indicating if the model is to be used as a debater or judge.
            greedy: Whether the model should decode using greedy decoding (True) or beam search (False).
                This is used only for the Llama / Mistral models at the moment.

        Returns:
            An instantiated model of the given type.

        Raises:
            Exception: Raises exception if the model type does not exist or if the model cannot be instantiated
                directly. At the moment, neither the OfflineModel nor the HumanModel can be instantiated directly.
        """
        if model_type == ModelType.RANDOM:
            return RandomModel(alias=alias, is_debater=is_debater)
        elif model_type == ModelType.LLAMA:
            return LlamaModel(alias=alias, file_path=file_path, is_debater=is_debater, greedy=greedy)
        elif model_type == ModelType.MISTRAL:
            return MistralModel(alias=alias, file_path=file_path, is_debater=is_debater, greedy=greedy)
        elif model_type == ModelType.DETERMINISTIC:
            return DeterministicModel(alias=alias, is_debater=is_debater)
        elif model_type == ModelType.OPENAI:
            return OpenAIModel(alias=alias, is_debater=is_debater)
        elif model_type == ModelType.OFFLINE:
            raise Exception("Offline model cannot be directly instantiated")
        elif model_type == ModelType.HUMAN:
            raise Exception("Human model cannot be directly instantiated")
        raise Exception(f"Model {model_type} not found")
