from __future__ import annotations

from agents.models.model import Model, ModelInput, ModelResponse, SpeechStructure
from prompts import RoleType
from utils import LoggerUtils, StringUtils, timer
import utils.constants as constants

from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import numpy as np
import torch

from enum import Enum
from typing import Optional, Union, Type
import copy
import math
import re


class LLMInput(BaseModel):
    instruction: str
    input: str
    extra_suffix: Optional[str]


class GenerationParams(BaseModel):
    max_new_tokens: int = 300
    temperature: float = 0.5
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    do_sample: bool = True


class LLModel(Model):
    INSTRUCTION_PREFIX = ""
    INSTRUCTION_SUFFIX = ""
    TARGET_MODULES = []
    DEFAULT_GENERATION_PARAMS = GenerationParams()

    def __init__(
        self,
        alias: str,
        file_path: Optional[str] = None,
        is_debater: bool = True,
        nucleus: bool = True,
        instruction_prefix: str = "",
        instruction_suffix: str = "",
    ):
        """
        An LLModel uses a large language model (currently Llama 2 or Mistral) to generate text.

        Args:
            alias: String that identifies the model for metrics and deduplication
            is_debater: Boolean indicating whether the model is a debater (true) or judge (false)
            nucleus: Whether nucleus sampling (true) or beam_search (false) should be used.
            instruction_prefix: the prefix to use before the instructions that get passed to the model
            instruction_suffix: the suffix to use after the instructions that get passed to the model
        """
        super().__init__(alias=alias, is_debater=is_debater)
        torch.cuda.empty_cache()
        self.logger = LoggerUtils.get_default_logger(__name__)
        self.instruction_prefix = instruction_prefix
        self.instruction_suffix = instruction_suffix
        if file_path:
            self.is_debater = is_debater
            self.tokenizer = AutoTokenizer.from_pretrained(file_path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                file_path,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
                use_flash_attention_2=True,
            )

            self.generation_config = GenerationConfig(
                max_new_tokens=LLModel.DEFAULT_GENERATION_PARAMS.max_new_tokens,
                temperature=LLModel.DEFAULT_GENERATION_PARAMS.temperature,
                top_p=LLModel.DEFAULT_GENERATION_PARAMS.top_p,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=LLModel.DEFAULT_GENERATION_PARAMS.repetition_penalty,
                do_sample=LLModel.DEFAULT_GENERATION_PARAMS.do_sample,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            if not nucleus:
                self.generation_config.num_beams = 2
                self.generation_config.do_sample = False
                self.generation_config.top_p = None
                self.generation_config.temperature = None

        else:
            self.is_debater = False
            self.tokenizer = None
            self.model = None
            self.generation_config = None

    @classmethod
    def generate_llm_input_from_model_inputs(cls, input_list: list[ModelInput], extra_suffix: str = "") -> LLMInput:
        """Converts a ModelInput into the LLMInput that's expected by the model"""
        return LLMInput(
            instruction="\n".join(
                model_input.content for model_input in filter(lambda x: x.role == RoleType.SYSTEM, input_list)
            ),
            input="\n".join(model_input.content for model_input in filter(lambda x: x.role != RoleType.SYSTEM, input_list)),
            extra_suffix=extra_suffix,
        )

    @classmethod
    def generate_input_str(cls, llm_input: LLMInput, instruction_prefix: str = "", instruction_suffix: str = "") -> str:
        """Transforms a LLMInput into a standardized format"""
        return "{} {}\n\n{} {}{}".format(
            instruction_prefix,
            llm_input.instruction,
            llm_input.input,
            instruction_suffix,
            (" " + llm_input.extra_suffix) if llm_input.extra_suffix else "",
        )

    def generate_input_strs(
        self, inputs: list[list[ModelInput]], speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED
    ) -> list[str]:
        """Converts a list of model inputs into a list of strings that can be tokenized"""

        def get_extra_suffix(speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED):
            if speech_structure == SpeechStructure.DECISION:
                return "\n\n" + constants.JUDGING_PREFIX
            return ""

        input_strs = []
        for input_list in inputs:
            input_strs.append(
                LLModel.generate_input_str(
                    llm_input=LLModel.generate_llm_input_from_model_inputs(
                        input_list=input_list, extra_suffix=get_extra_suffix(speech_structure)
                    ),
                    instruction_prefix=self.instruction_prefix,
                    instruction_suffix=self.instruction_suffix,
                )
            )

        return input_strs

    @timer("llm inference")
    @torch.inference_mode()
    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens=300,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[ModelResponse]:
        """
        Generates a list of texts in response to the given input.

        Args:
            inputs: A list of list of model inputs. Each ModelInput corresponds roughly to one command,
                a list of ModelInputs corresponds to a single debate (or entry in a batch), and so the
                list of lists is basically a batch of debates.
            max_new_tokens: the maximum number of new tokens to generate.
            speech_structure: the format that the answer is expected to be in. Option includes "open-ended"
                (which is just free text), and "decision" (which means a boolean is expected)
            num_return_sequences: the number of responses that the model is expected to generate. If a batch
                size of >1 is passed in, then this value will be overridden by the batch size (so you cannot
                have both num_return_sequences > 1 and len(inputs) > 1)

        Returns:
            A list of model responses, with one response for each entry in the batch (or for as many sequences
            are specified to be returned by num_return_sequences).

        Raises:
            Exception: Raises Exception if num_return_sequences > 1 and len(inputs) > 1
        """

        def validate():
            if num_return_sequences > 1 and len(inputs) > 1:
                raise Exception("You cannot have multiple return sequences and a batch size of >1")

        def get_string_log_prob(target_string: list[str], scores: torch.Tensor, batch_index: int) -> float:
            tokenized_target = self.tokenizer(target_string).input_ids[-1]
            return scores[0][batch_index][tokenized_target].item()

        def normalize_log_probs(a_prob: float, b_prob: float) -> tuple[float, float]:
            exponentiated = [math.exp(logprob) for logprob in [a_prob, b_prob]]
            return prob[0] / sum(exponentiated), prob[1] / sum(exponentiated)

        def create_new_generation_config():
            config_to_use = copy.deepcopy(self.generation_config)
            config_to_use.max_new_tokens = max_new_tokens
            config_to_use.num_return_sequences = num_return_sequences
            return config_to_use

        validate()
        self.model.eval()
        input_strs = self.generate_input_strs(inputs=inputs, speech_structure=speech_structure)
        inputs = self.tokenizer(input_strs, return_tensors="pt", padding=True)
        outputs = self.model.generate(**inputs, generation_config=create_new_generation_config())
        input_lengths = (inputs.input_ids != self.tokenizer.pad_token_id).sum(axis=1)

        decoded_outputs = []
        for i, row in enumerate(outputs.sequences):
            if self.is_debater or speech_structure != SpeechStructure.DECISION:
                decoded = self.tokenizer.decode(outputs.sequences[i, input_lengths[min(i, len(input_lengths) - 1)] :])
                new_tokens = decoded.split(constants.INSTRUCTION_SUFFIX)[-1]
                decoded_outputs.append(ModelResponse(speech=StringUtils.clean_string(new_tokens)))
            else:
                tokenized_debater_a = self.tokenizer(constants.DEFAULT_DEBATER_A_NAME)
                tokenized_debater_b = self.tokenizer(constants.DEFAULT_DEBATER_B_NAME)
                decoded = self.tokenizer.decode(outputs.sequences[i, input_lengths[i] :])
                a_score = get_string_log_prob(constants.DEFAULT_DEBATER_A_NAME, outputs.scores, i)
                b_score = get_string_log_prob(constants.DEFAULT_DEBATER_B_NAME, outputs.scores, i)
                normalized_a_score, normalized_b_score = normalize_log_probs(a_score, b_score)

                decoded_outputs.append(
                    ModelResponse(
                        decision=(
                            constants.DEFAULT_DEBATER_A_NAME if a_score > b_score else constants.DEFAULT_DEBATER_B_NAME
                        ),
                        probabilistic_decision={
                            constants.DEFAULT_DEBATER_A_NAME: normalized_a_score,
                            constants.DEFAULT_DEBATER_B_NAME: normalized_b_score,
                        },
                        prompt=input_strs[i],
                    )
                )

                self.logger.info(f"Scores: A {normalized_a_score} - B {normalized_b_score}")

        return decoded_outputs

    def copy(self, alias: str, is_debater: Optional[bool] = None, nucleus: bool = False) -> LLModel:
        """Generates a deepcopy of this model"""
        copy = LLModel(alias=alias, is_debater=self.is_debater if is_debater == None else is_debater, nucleus=nucleus)
        copy.is_debater = self.is_debater if is_debater == None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy


class LlamaModel(LLModel):
    INSTRUCTION_PREFIX = "instruction:"
    INSTRUCTION_SUFFIX = "output:"
    TARGET_MODULES = ["k_proj", "v_proj", "down_proj"]

    def __init__(
        self,
        alias: str,
        file_path: Optional[str] = None,
        is_debater: bool = True,
        nucleus: bool = True,
    ):
        super().__init__(
            alias=alias,
            file_path=file_path,
            is_debater=is_debater,
            nucleus=nucleus,
            instruction_prefix="instruction:",
            instruction_suffix="output:",
        )

        if self.model:
            self.model.config.max_position_embeddings = constants.MAX_LENGTH

    def copy(self, alias: str, is_debater: Optional[bool] = None, nucleus: bool = False) -> LLModel:
        """Generates a deepcopy of this model"""
        copy = LlamaModel(alias=alias, is_debater=self.is_debater if is_debater == None else is_debater, nucleus=nucleus)
        copy.is_debater = self.is_debater if is_debater == None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy


class MistralModel(LLModel):
    INSTRUCTION_PREFIX = "[INST]"
    INSTRUCTION_SUFFIX = "[/INST]"
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        alias: str,
        file_path: Optional[str] = None,
        is_debater: bool = True,
        nucleus: bool = True,
    ):
        super().__init__(
            alias=alias,
            file_path=file_path,
            is_debater=is_debater,
            nucleus=nucleus,
            instruction_prefix="[INST]",
            instruction_suffix="[/INST]",
        )

        if self.model:
            self.model.config.sliding_window = constants.MAX_LENGTH

    def copy(self, alias: str, is_debater: Optional[bool] = None, nucleus: bool = False) -> LLModel:
        """Generates a deepcopy of this model"""
        copy = MistralModel(alias=alias, is_debater=self.is_debater if is_debater == None else is_debater, nucleus=nucleus)
        copy.is_debater = self.is_debater if is_debater == None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy


class LLMType(Enum):
    LLAMA = 0
    MISTRAL = 1

    def get_llm_class(self) -> Type[LLModel]:
        if self == LLMType.LLAMA:
            return LlamaModel
        elif self == LLMType.MISTRAL:
            return MistralModel
        else:
            raise Exception(f"Model type {self} not recognized")
