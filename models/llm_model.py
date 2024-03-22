from __future__ import annotations

from models.model import Model, ModelInput, ModelResponse, ProbeHyperparams, SpeechStructure
from models.openai_model import OpenAIModel
from prompts import RoleType
from utils import logger_utils, string_utils, timer
import utils.constants as constants

from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import numpy as np
import torch.nn as nn
import torch

from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Optional, Union, Type
import base64
import copy
import io
import math
import os
import random
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
    MAX_MINI_BATCH_SIZE = 8

    def __init__(
        self,
        alias: str,
        file_path: Optional[str] = None,
        is_debater: bool = True,
        nucleus: bool = True,
        instruction_prefix: str = "",
        instruction_suffix: str = "",
        requires_file_path: bool = True,
        probe_hyperparams: Optional[ProbeHyperparams] = None,
        max_mini_batch_size: Optional[int] = None,
    ):
        """
        An LLModel uses a large language model (currently Llama 2 or Mistral) to generate text.

        Args:
            alias: String that identifies the model for metrics and deduplication
            file_path: the name of the huggingface model to load
            is_debater: Boolean indicating whether the model is a debater (true) or judge (false)
            nucleus: Whether nucleus sampling (true) or beam_search (false) should be used.
            instruction_prefix: the prefix to use before the instructions that get passed to the model
            instruction_suffix: the suffix to use after the instructions that get passed to the model
            requires_file_path: whether a file path is needed to instantiate the model
            probe_hyperparams: configuration for a linear probe judge
        """
        super().__init__(alias=alias, is_debater=is_debater)
        torch.cuda.empty_cache()
        self.logger = logger_utils.get_default_logger(__name__)
        self.instruction_prefix = instruction_prefix
        self.instruction_suffix = instruction_suffix
        self.instantiated_model = False
        self.max_mini_batch_size = max_mini_batch_size or LLModel.MAX_MINI_BATCH_SIZE
        if file_path or not requires_file_path:
            self.instantiated_model = True
            self.is_debater = is_debater

            self.tokenizer, self.model = self.instantiate_tokenizer_and_hf_model(file_path=file_path)
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
                output_hidden_states=not is_debater,
            )

            if not nucleus:
                self.generation_config.num_beams = 2
                self.generation_config.do_sample = False
                self.generation_config.top_p = None
                self.generation_config.temperature = None

            if probe_hyperparams:
                if not is_debater:
                    self.model = LLModuleWithLinearProbe(
                        base_model=self.model,
                        linear_idxs=probe_hyperparams.linear_idxs,
                        file_path=probe_hyperparams.file_path,
                    )
                else:
                    self.logger.warn("Probe hyperparameters were passed in for a debater model. This is not supported.")

        else:
            self.is_debater = False
            self.tokenizer = None
            self.model = None
            self.generation_config = None

    @classmethod
    def instantiate_tokenizer(
        self, file_path: str, requires_token: bool = False
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        tokenizer = AutoTokenizer.from_pretrained(
            file_path,
            token=os.getenv("META_ACCESS_TOKEN") if requires_token else None,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    @classmethod
    def instantiate_hf_model(
        self, file_path: str, requires_token: bool = False, use_cache: bool = True
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=file_path,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_flash_attention_2=True,
            use_cache=use_cache,
            token=os.getenv("META_ACCESS_TOKEN") if requires_token else None,
        )

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

    def instantiate_tokenizer_and_hf_model(
        self, file_path: str, requires_token: bool = False
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Constructs the tokenizer and huggingface model at the specified filepath"""
        tokenizer = LLModel.instantiate_tokenizer(file_path=file_path, requires_token=requires_token)
        hf_model = LLModel.instantiate_hf_model(file_path=file_path, requires_token=requires_token)
        return tokenizer, hf_model

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
        max_new_tokens: int = 300,
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
            return exponentiated[0] / sum(exponentiated), exponentiated[1] / sum(exponentiated)

        def create_new_generation_config():
            config_to_use = copy.deepcopy(self.generation_config)
            config_to_use.max_new_tokens = max_new_tokens
            config_to_use.num_return_sequences = num_return_sequences
            return config_to_use

        def generate_output(input_strs: list[str]):
            sequences = []
            scores = []
            input_lengths = []
            minibatches = [
                input_strs[i : i + self.max_mini_batch_size] for i in range(0, len(input_strs), self.max_mini_batch_size)
            ]
            for minibatch in minibatches:
                inputs = self.tokenizer(minibatch, return_tensors="pt", padding=True).to(device)
                outputs = self.model.generate(**inputs, generation_config=create_new_generation_config())
                mini_sequences = outputs.sequences if not isinstance(self.model, LLModuleWithLinearProbe) else outputs
                sequences += [row for row in mini_sequences]
                scores += [row for row in outputs.scores] if hasattr(outputs, "scores") else []
                input_lengths += [elem for elem in (inputs.input_ids != self.tokenizer.pad_token_id).sum(axis=1)]
            return sequences, torch.stack(input_lengths), torch.stack(scores) if scores else None

        validate()
        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_strs = self.generate_input_strs(inputs=inputs, speech_structure=speech_structure)
        sequences, input_lengths, scores = generate_output(input_strs=input_strs)

        decoded_outputs = []
        for i, row in enumerate(sequences):
            if self.is_debater or speech_structure != SpeechStructure.DECISION:
                decoded = self.tokenizer.decode(sequences[i][input_lengths[min(i, len(input_lengths) - 1)] :])
                new_tokens = decoded.split(constants.INSTRUCTION_SUFFIX)[-1]
                decoded_outputs.append(ModelResponse(speech=string_utils.clean_string(new_tokens), prompt=input_strs[i]))
            else:
                internal_representations = []
                if isinstance(self.model, LLModuleWithLinearProbe):
                    (a_score, b_score), internal_representations = outputs[i]
                else:
                    tokenized_debater_a = self.tokenizer(constants.DEFAULT_DEBATER_A_NAME)
                    tokenized_debater_b = self.tokenizer(constants.DEFAULT_DEBATER_B_NAME)
                    decoded = self.tokenizer.decode(sequences[i, input_lengths[i] :])
                    a_score = get_string_log_prob(constants.DEFAULT_DEBATER_A_NAME, scores, i)
                    b_score = get_string_log_prob(constants.DEFAULT_DEBATER_B_NAME, scores, i)

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
                        internal_representations=internal_representations,
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
        probe_hyperparams: Optional[ProbeHyperparams] = None,
    ):
        super().__init__(
            alias=alias,
            file_path=file_path,
            is_debater=is_debater,
            nucleus=nucleus,
            instruction_prefix="instruction:",
            instruction_suffix="output:",
            requires_file_path=True,
            probe_hyperparams=probe_hyperparams,
            max_mini_batch_size=1,
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
    LINEAR_IDXS = [31, 16]

    def __init__(
        self,
        alias: str,
        file_path: Optional[str] = None,
        is_debater: bool = True,
        nucleus: bool = True,
        probe_hyperparams: Optional[ProbeHyperparams] = None,
    ):
        super().__init__(
            alias=alias,
            file_path=file_path,
            is_debater=is_debater,
            nucleus=nucleus,
            instruction_prefix="[INST]",
            instruction_suffix="[/INST]",
            requires_file_path=True,
            probe_hyperparams=probe_hyperparams,
            max_mini_batch_size=1,
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


class StubLLModel(LLModel):
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
            instruction_prefix="",
            instruction_suffix="",
            requires_file_path=False,
        )

    def copy(self, alias: str, is_debater: Optional[bool] = None, nucleus: bool = False) -> LLModel:
        """Generates a deepcopy of this model"""
        return StubLLModel(alias=alias, is_debater=self.is_debater if is_debater == None else is_debater, nucleus=nucleus)

    def instantiate_tokenizer_and_hf_model(self, file_path: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Constructs the stub tokenizer and stub model"""
        return TokenizerStub(), ModelStub()


class LLModuleWithLinearProbe(nn.Module):
    def __init__(self, base_model: LLModel, linear_idxs: Optional[list[int]] = None, file_path: str = ""):
        super().__init__()
        self.linear_idxs = linear_idxs or [-1]
        self.base_model = base_model.model.to("cuda")
        self.base_model.eval()
        self.config = self.base_model.config
        self.probe = LLModuleWithLinearProbe.instantiate_probe(
            file_path=file_path, linear_idxs=self.linear_idxs, hidden_size=self.base_model.config.hidden_size
        )
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid(dim=1)

    @classmethod
    def instantiate_probe(cls, file_path: str, linear_idxs: list[int], hidden_size: int) -> nn.Module:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        probe = nn.Linear(in_features=hidden_size * len(linear_idxs), out_features=1)
        if file_path:
            probe.load_state_dict(torch.load(file_path))
        else:
            raise Exception(f"File path ({file_path}) not loaded")
        return probe.to(device)

    def encode_representation(self, representation: torch.tensor) -> str:
        buffer = io.BytesIO()
        torch.save(representation, buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def generate(self, input_ids: torch.tensor, **kwargs) -> list[tuple(tuple(float, float), torch.tensor)]:
        return self.forward(input_ids=input_ids)

    def forward(self, input_ids: Optional[torch.tensor] = None) -> list[tuple(tuple(float, float), torch.tensor)]:
        batch_size = input_ids.shape[0]

        base_model_output = self.base_model(input_ids=input_ids.to("cuda"), output_hidden_states=True)

        hidden_states = [[] for i in range(batch_size)]
        for i, layer in enumerate(base_model_output.hidden_states):
            for j in range(batch_size):
                hidden_states[j].append(layer[j, -1, :])

        input_vecs = torch.stack(
            [torch.cat([hidden_states[i][idx] for idx in self.linear_idxs], dim=0) for i in range(batch_size)]
        )

        unnormalized_outputs = self.probe(input_vecs.float())
        # outputs = self.softmax(unnormalized_outputs)
        a_odds = self.sigmoid(unnormalized_outputs)
        outputs = [a_odds, 1 - a_odds]
        reformatted_outputs = [(output[0].item(), output[1].item()) for output in outputs]
        encoded_hidden_states = [self.encode_representation(hs) for hs in hidden_states]

        return [(ro, ehs) for ro, ehs in zip(reformatted_outputs, encoded_hidden_states)]

    def parameters(self):
        return self.probe.parameters()


class LLMType(Enum):
    LLAMA = auto()
    MISTRAL = auto()
    OPENAI = auto()
    STUB_LLM = auto()

    def get_llm_class(self) -> Type[LLModel]:
        if self == LLMType.LLAMA:
            return LlamaModel
        elif self == LLMType.MISTRAL:
            return MistralModel
        elif self == LLMType.STUB_LLM:
            return StubLLModel
        elif self == LLMType.OPENAI:
            return OpenAIModel
        else:
            raise Exception(f"Model type {self} not recognized")


@dataclass
class ModelConfigStub:
    max_position_embeddings: int = 0


class TokenizerOutputStub:
    def __init__(self, input_ids: torch.tensor):
        self.input_ids = input_ids
        self.__data = {"input_ids": self.input_ids}

    def __iter__(self):
        return iter(self.__data)

    def keys(self):
        return self.__data.keys()

    def __getitem__(self, key):
        return self.__data[key]

    def to(self, device: str):
        return self


class TokenizerStub:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def __call__(self, inputs: list[str], **kwargs) -> dict[str, torch.tensor]:
        batch_size = len(inputs)
        max_sequence_length = max(len(seq) for seq in inputs)
        return TokenizerOutputStub(input_ids=torch.tensor(np.random.randint(0, 100, [batch_size, max_sequence_length])))

    def encode(self, input_string: str | list[str], **kwargs):
        if not isinstance(input_string, str) or not isinstance(input_string, list):
            return input_string

        length = len(input_string)
        if isinstance(input_string, str):
            input_string = [input_string]
        input_ids = self(input_string).input_ids
        if len(input_string) == 1:
            return input_ids[0, :]
        return input_ids

    def decode(self, tokens: torch.tensor) -> str | list[str]:
        if len(tokens.shape) == 1:
            batch_size = 1
            sequence_length = tokens.shape[0]
        else:
            batch_size, sequence_length = tokens.shape
        outputs = [
            " ".join(["".join(random.choices(self.alphabet, k=random.randrange(1, 8))) for i in range(sequence_length)])
            for _ in range(batch_size)
        ]
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


@dataclass
class ModelOutputStub:
    sequences: Any  # should be a torch tensor


class ModelStub:
    def __init__(self):
        self.config = ModelConfigStub()

    def train(self):
        pass

    def eval(self):
        pass

    def generate(self, input_ids: torch.tensor, generation_config: GenerationConfig, **kwargs):
        return self(input_ids=input_ids, generation_config=generation_config, **kwargs)

    def __call__(self, input_ids: torch.tensor, generation_config: GenerationConfig, **kwargs):
        batch_size, sequence_length = input_ids.shape
        output_sequence_length = sequence_length + generation_config.max_new_tokens
        return ModelOutputStub(sequences=torch.tensor(np.random.randint(0, 100, [batch_size, output_sequence_length])))
