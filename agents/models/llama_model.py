from __future__ import annotations

from agents.models.model import Model, ModelInput, SpeechStructure
from prompts import RoleType
from utils import LoggerUtils, StringUtils, timer
import utils.constants as constants

from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import numpy as np
import torch

from typing import Optional, Union
import copy
import re


class LlamaInput(BaseModel):
    instruction: str
    input: str
    extra_suffix: Optional[str]


class LlamaModel(Model):
    def __init__(self, alias: str, file_path: Optional[str] = None, is_debater: bool = True, greedy: bool = True):
        """
        A Llama model uses Llama2 to generate text.

        Args:
            alias: String that identifies the model for metrics and deduplication
            is_debater: Boolean indicating whether the model is a debater (true) or judge (false)
            greedy: Whether greedy decoding (true) or beam_search (false) should be used.
        """
        super().__init__(alias=alias, is_debater=is_debater)
        torch.cuda.empty_cache()
        self.logger = LoggerUtils.get_default_logger(__name__)
        if file_path:
            self.is_debater = is_debater
            self.tokenizer = AutoTokenizer.from_pretrained(
                file_path, additional_special_tokens=[constants.QUOTE_TAG, constants.UNQUOTE_TAG]
            )
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
                max_new_tokens=300,
                temperature=0.5,
                top_p=0.9,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.2,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            if not greedy:
                self.generation_config.num_beams = 2
                self.generation_config.do_sample = False
                self.generation_config.top_p = None
                self.generation_config.temperature = None

        else:
            self.is_debater = False
            self.tokenizer = None
            self.model = None
            self.generation_config = None

        if self.model:
            self.model.config.max_position_embeddings = constants.MAX_LENGTH
            self.model.config.transformers_version = "4.34.0"
            self.model.generation_config.transformers_version = "4.34.0"

    @classmethod
    def generate_input_str(cls, llama_input: LlamaInput) -> str:
        """Transforms a LlamaInput into a standardized format"""
        return "{}{}\n\n{}\n\n {} {}".format(
            constants.INSTRUCTION_PREFIX,
            llama_input.instruction,
            llama_input.input,
            constants.INSTRUCTION_SUFFIX,
            llama_input.extra_suffix,
        )

    @classmethod
    def generate_llama_input_from_model_inputs(cls, input_list: list[ModelInput], extra_suffix: str = "") -> LlamaInput:
        """Converts a ModelInput into the LlamaInput that's expected by the model"""
        return LlamaInput(
            instruction="\n".join(
                model_input.content for model_input in filter(lambda x: x.role == RoleType.SYSTEM, input_list)
            ),
            input="\n".join(model_input.content for model_input in filter(lambda x: x.role != RoleType.SYSTEM, input_list)),
            extra_suffix=extra_suffix,
        )

    @classmethod
    def generate_input_str_from_model_inputs(
        cls,
        input_list: list[ModelInput],
        is_debater: bool = True,
        alias: str = "",
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
    ) -> LlamaInput:
        """Creates the string that can be passed into Llama for it to generate responses"""

        def get_extra_suffix():
            if speech_structure == SpeechStructure.DECISION:
                return "\n\n" + constants.JUDGING_PREFIX
            elif speech_structure == SpeechStructure.PREFERENCE:
                return "\n\n" + constants.PREFERENCE_PREFIX
            return ""

        return LlamaModel.generate_input_str(
            LlamaModel.generate_llama_input_from_model_inputs(input_list=input_list, extra_suffix=get_extra_suffix())
        )

    @timer("llama inference")
    @torch.inference_mode()
    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens=300,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[str]:
        """
        Generates a list of texts in response to the given input.

        Args:
            inputs: A list of list of model inputs. Each ModelInput corresponds roughly to one command,
                a list of ModelInputs corresponds to a single debate (or entry in a batch), and so the
                list of lists is basically a batch of debates.
            max_new_tokens: the maximum number of new tokens to generate.
            speech_structure: the format that the answer is expected to be in. Option includes "open-ended"
                (which is just free text), "preference" (which means a number is expected), and "decision"
                (which means a boolean is expected)
            num_return_sequences: the number of responses that the model is expected to generate. If a batch
                size of >1 is passed in, then this value will be overridden by the batch size (so you cannot
                have both num_return_sequences > 1 and len(inputs) > 1)

        Returns:
            A list of text, with one string for each entry in the batch (or for as many sequences are specified
            to be returned by num_return_sequences).

        Raises:
            Exception: Raises Exception if num_return_sequences > 1 and len(inputs) > 1
        """

        def validate():
            if num_return_sequences > 1 and len(inputs) > 1:
                raise Exception("You cannot have multiple return sequences and a batch size of >1")

        def get_string_log_prob(target_string: list[str], scores: torch.Tensor, batch_index: int) -> float:
            tokenized_target = self.tokenizer(target_string).input_ids[-1]
            return scores[0][batch_index][tokenized_target].item()

        def create_new_generation_config():
            config_to_use = copy.deepcopy(self.generation_config)
            config_to_use.max_new_tokens = max_new_tokens
            config_to_use.num_return_sequences = num_return_sequences
            return config_to_use

        validate()
        self.model.eval()
        input_strs = [
            LlamaModel.generate_input_str_from_model_inputs(input_list, self.is_debater, self.alias, speech_structure)
            for input_list in inputs
        ]

        inputs = self.tokenizer(input_strs, return_tensors="pt", padding=True).to("cuda")
        outputs = self.model.generate(**inputs, generation_config=create_new_generation_config())
        input_lengths = (inputs.input_ids != self.tokenizer.pad_token_id).sum(axis=1)

        decoded_outputs = []
        for i, row in enumerate(outputs.sequences):
            if self.is_debater or speech_structure != SpeechStructure.DECISION:
                decoded = self.tokenizer.decode(outputs.sequences[i, input_lengths[min(i, len(input_lengths) - 1)] :])
                new_tokens = decoded.split(constants.INSTRUCTION_SUFFIX)[-1]
                if speech_structure == SpeechStructure.PREFERENCE and re.search("\\d+(\\.\\d+)?", new_tokens.strip()):
                    decoded_outputs.append(re.search("\\d+(\\.\\d+)?", new_tokens.strip()).group())
                else:
                    decoded_outputs.append(StringUtils.clean_string(new_tokens))
            else:
                tokenized_debater_a = self.tokenizer(constants.DEFAULT_DEBATER_A_NAME)
                tokenized_debater_b = self.tokenizer(constants.DEFAULT_DEBATER_B_NAME)
                decoded = self.tokenizer.decode(outputs.sequences[i, input_lengths[i] :])
                a_score = get_string_log_prob(constants.DEFAULT_DEBATER_A_NAME, outputs.scores, i)
                b_score = get_string_log_prob(constants.DEFAULT_DEBATER_B_NAME, outputs.scores, i)

                decoded_outputs.append(
                    constants.DEFAULT_DEBATER_A_NAME if a_score > b_score else constants.DEFAULT_DEBATER_B_NAME
                )
                self.logger.info(f"Scores: A {a_score} - B {b_score}")

        return decoded_outputs

    def copy(self, alias: str, is_debater: Optional[bool] = None, greedy: bool = False) -> LlamaModel:
        """Generates a deepcopy of this model"""
        copy = LlamaModel(alias=alias, is_debater=self.is_debater if is_debater == None else is_debater, greedy=greedy)
        copy.is_debater = self.is_debater if is_debater == None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy
