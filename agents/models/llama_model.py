from __future__ import annotations

from agents.model import Model, ModelInput, RoleType, SpeechStructure
from utils.logger_utils import LoggerUtils
from utils.string_utils import StringUtils
from utils.timer_utils import timer
import utils.constants as constants

from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import numpy as np
import torch

from typing import Optional, Union
import copy
import re

try:
    from utils.flash_attn_utils import replace_with_flash_decoding

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class LlamaInput(BaseModel):
    instruction: str
    input: str
    extra_suffix: Optional[str]


class LlamaModel(Model):
    def __init__(self, alias: str, file_path: Optional[str] = None, is_debater: bool = True, beam_search: bool = True):
        super().__init__(alias=alias, is_debater=is_debater)
        torch.cuda.empty_cache()
        self.logger = LoggerUtils.get_default_logger(__name__)
        if file_path:
            if FLASH_ATTENTION_AVAILABLE:
                # replace_with_flash_decoding()
                pass
            self.is_debater = is_debater
            self.tokenizer = AutoTokenizer.from_pretrained(file_path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # for open-ended generation

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

            if beam_search:
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
            self.model.config.max_position_embeddings = 32768
            self.model.config.transformers_version = "4.34.0"
            self.model.generation_config.transformers_version = "4.34.0"
            self.logger.debug(self.model.config)
            self.logger.debug(f"Flash attention 2 enabled? {self.model.config._flash_attn_2_enabled}")

    @classmethod
    def generate_input_str(cls, llama_input: LlamaInput) -> str:
        return "{}{}\n\n{}\n\n {} {}".format(
            constants.INSTRUCTION_PREFIX,
            llama_input.instruction,
            llama_input.input,
            constants.INSTRUCTION_SUFFIX,
            llama_input.extra_suffix,
        )

    @classmethod
    def generate_llama_input_from_model_inputs(cls, input_list: list[ModelInput], extra_suffix: str = "") -> LlamaInput:
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
        # TODO: remove this -- I think the base model is not responding to commands because its instruction format
        # is non-standard so this is just a patch until I figure that out and train the debating model accordingly
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

    def copy(self, alias: str, is_debater: Optional[bool] = None) -> LlamaModel:
        copy = LlamaModel(alias=alias, is_debater=self.is_debater if is_debater == None else is_debater)
        copy.is_debater = self.is_debater if is_debater == None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy
