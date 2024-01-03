from __future__ import annotations

from agents.models.model import Model, ModelInput, ModelResponse, SpeechStructure
from agents.models.llm_model import GenerationParams, LLModel
from utils import LoggerUtils, timer

from pydantic import BaseModel
import requests

from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time


class RequestParams(BaseModel):
    inputs: str
    parameters: GenerationParams


class ResponseStruct(BaseModel):
    generated_text: str


class ServedModel(Model):
    DEFAULT_GENERATION_PARAMS = GenerationParams()
    DEFAULT_SERVING_ENDPOINT = "http://127.0.0.1:8080/generate"
    MAX_PARALLEL_REQUESTS = 8
    DEFAULT_HEADER = {"Content-Type": "application/json"}

    def __init__(self, base_model: Model):
        """
        A served model calls a hosted model running on a local endpoint for inference.

        Args:
            base_model: A model of the type that is being served. This is needed so that we can
                define the input format appropriately and set the correct alias.
        """
        super().__init__(alias=base_model.alias, is_debater=base_model.is_debater)
        self.base_model = base_model
        self.logger = LoggerUtils.get_default_logger(__name__)

    def fetch(self, input_string: str) -> str:
        """Hits the default endpoint for the served model"""
        data = RequestParams(inputs=input_string, parameters=ServedModel.DEFAULT_GENERATION_PARAMS).dict()
        response = requests.post(ServedModel.DEFAULT_SERVING_ENDPOINT, headers=ServedModel.DEFAULT_HEADER, json=data)
        return ResponseStruct(**response.json()).generated_text

    @timer("served LLM inference")
    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens=300,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[str]:
        """
        Generates a list of texts in response to the given input. Note that this can only be used for
        speeches and not for judging since the log probs are not exposed.

        Args:
            inputs: A list of list of model inputs. Each ModelInput corresponds roughly to one command,
                a list of ModelInputs corresponds to a single debate (or entry in a batch), and so the
                list of lists is basically a batch of debates.
            max_new_tokens: the maximum number of new tokens to generate.
            num_return_sequences: the number of responses that the model is expected to generate. If a batch
                size of >1 is passed in, then this value will be overridden by the batch size (so you cannot
                have both num_return_sequences > 1 and len(inputs) > 1)

        Returns:
            A list of text, with one string for each entry in the batch (or for as many sequences are specified
            to be returned by num_return_sequences).

        Raises:
            Exception: Raises Exception if num_return_sequences > 1 and len(inputs) > 1
        """

        if num_return_sequences > 1 and len(inputs) > 1:
            raise Exception("You cannot have multiple return sequences and a batch size of >1")

        with ThreadPoolExecutor(max_workers=ServedModel.MAX_PARALLEL_REQUESTS) as executor:
            futures = [
                executor.submit(self.fetch, input_string)
                for input_string in self.base_model.generate_input_strs(
                    inputs=inputs, speech_structure=SpeechStructure.OPEN_ENDED
                )
            ]
            results = [ModelResponse(speech=future.result()) for future in as_completed(futures)]

        return results
