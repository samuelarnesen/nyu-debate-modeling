from agents.models.model import Model, ModelSettings
from agents.models.arbitrary_attribute_model import ArbitraryAttributeModel
from agents.models.deterministic_model import DeterministicModel
from agents.models.llm_model import LlamaModel, MistralModel, StubLLModel
from agents.models.openai_model import OpenAIModel
from agents.models.random_model import RandomModel
from agents.models.served_model import ServedModel

from pydantic import BaseModel

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
    STUB_LLM = 8
    ARBITRARY_ATTRIBUTE = 9


class ModelUtils:
    @classmethod
    def instantiate_model(
        cls,
        model_settings: ModelSettings,
        is_debater: bool = True,
    ) -> Optional[Model]:
        """
        Builds a model using the given inputs.

        Args:
            model_settings: the configuration object for the model
            is_debater: Boolean indicating if the model is to be used as a debater or judge.

        Returns:
            An instantiated model of the given type.

        Raises:
            Exception: Raises exception if the model type does not exist or if the model cannot be instantiated
                directly. At the moment, neither the OfflineModel nor the HumanModel can be instantiated directly.
        """
        model_type = ModelType[model_settings.model_type.upper()]
        if model_type == ModelType.RANDOM:
            model = RandomModel(alias=model_settings.alias, is_debater=is_debater)
        elif model_type == ModelType.LLAMA:
            model = LlamaModel(
                alias=model_settings.alias,
                file_path=model_settings.model_file_path,
                is_debater=is_debater,
                nucleus=model_settings.nucleus,
                probe_hyperparams=model_settings.probe_hyperparams,
            )
        elif model_type == ModelType.MISTRAL:
            model = MistralModel(
                alias=model_settings.alias,
                file_path=model_settings.model_file_path,
                is_debater=is_debater,
                nucleus=model_settings.nucleus,
                probe_hyperparams=model_settings.probe_hyperparams,
            )
        elif model_type == ModelType.STUB_LLM:
            model = StubLLModel(alias=model_settings.alias)
        elif model_type == ModelType.DETERMINISTIC:
            model = DeterministicModel(alias=model_settings.alias, is_debater=is_debater)
        elif model_type == ModelType.OPENAI:
            model = OpenAIModel(
                alias=model_settings.alias, is_debater=is_debater, tokens_of_difference=model_settings.tokens_of_difference
            )
        elif model_type == ModelType.ARBITRARY_ATTRIBUTE:
            model = ArbitraryAttributeModel(alias=model_settings.alias, is_debater=is_debater)
        elif model_type == ModelType.OFFLINE:
            model = None  # offline models aren't directly instantiated
        elif model_type == ModelType.HUMAN:
            model = None  # offline models aren't directly instantiated
        else:
            raise Exception(f"Model {model_type} not found")

        if model_settings.served:
            if model_type in [ModelType.LLAMA, ModelType.MISTRAL]:  # expand when more types allow serving
                model = ServedModel(base_model=model)
            else:
                raise Exception(f"Model type {model_type} does not support serving")

        return model
