from __future__ import annotations

from agents.model import Model, ModelInput, RoleType
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, pipeline
import numpy as np
import torch

from typing import Optional, Union
import copy
import math
import time


class LlamaModel(Model):
    def __init__(self, alias: str, file_path: Optional[str] = None, is_debater: bool = True):
        super().__init__(alias=alias, is_debater=is_debater)
        torch.cuda.empty_cache()
        if file_path:
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
                file_path, device_map="auto", quantization_config=bnb_config, trust_remote_code=False, local_files_only=True
            )

            self.generation_config = GenerationConfig(
                max_new_tokens=450,
                temperature=0.5,
                top_p=0.9,
                num_return_sequence=1,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1,
                do_sample=True,
            )
        else:
            self.is_debater = False
            self.tokenizer = None
            self.model = None
            self.generation_config = None

        self.logger = LoggerUtils.get_default_logger(__name__)

    def predict(self, inputs: list[list[ModelInput]], max_new_tokens=450, decide: bool = False, **kwargs) -> list[str]:
        # TODO: remove this -- I think the base model is not responding to commands because its instruction format
        # is non-standard so this is just a patch until I figure that out and train the debating model accordingly
        def get_judging_suffix():
            if self.is_debater:
                return ""
            if decide:
                return "\n\n" + constants.JUDGING_PREFIX
            return "\n\nThanks for the great debate. Here is my current thought process."

        def generate_input_str(input_list: list[ModelInput]) -> str:
            return "{}\n\n{}\n\n{}\n\n{}\n\n{}{}".format(
                constants.INSTRUCTION_PREFIX,
                "\n".join(model_input.content for model_input in filter(lambda x: x.role == RoleType.SYSTEM, input_list)),
                constants.INPUT_PREFIX,
                "\n".join(model_input.content for model_input in filter(lambda x: x.role != RoleType.SYSTEM, input_list)),
                constants.INSTRUCTION_SUFFIX,
                get_judging_suffix(),
            )

        def get_string_log_prob(target_string: list[str], scores: torch.Tensor, batch_index: int) -> float:
            tokenized_target = self.tokenizer(target_string).input_ids[-1]
            self.logger.debug(f"Tokenized target is {tokenized_target}")
            return scores[0][batch_index][tokenized_target].item()

        self.model.eval()
        with torch.no_grad():
            start = time.time()
            config_to_use = copy.deepcopy(self.generation_config)
            config_to_use.max_new_tokens = max_new_tokens
            input_strs = [generate_input_str(input_list) for input_list in inputs]
            inputs = self.tokenizer(input_strs, return_tensors="pt", padding=True).to("cuda")
            outputs = self.model.generate(**inputs, generation_config=config_to_use)
            input_lengths = (inputs.input_ids != self.tokenizer.pad_token_id).sum(axis=1)

            decoded_outputs = []
            for i, row in enumerate(outputs.sequences):
                if self.is_debater or not decide:
                    decoded = self.tokenizer.decode(outputs.sequences[i, input_lengths[i] :])
                    new_tokens = decoded.split(constants.INSTRUCTION_SUFFIX)[-1]
                    decoded_outputs.append(new_tokens.rstrip())
                else:
                    tokenized_debater_a = self.tokenizer(constants.DEFAULT_DEBATER_A_NAME)
                    tokenized_debater_b = self.tokenizer(constants.DEFAULT_DEBATER_B_NAME)
                    decoded = self.tokenizer.decode(outputs.sequences[i, input_lengths[i] :])
                    self.logger.debug(
                        f"It wanted to decode the following: {decoded.split(constants.INSTRUCTION_SUFFIX)[-1].rstrip()}"
                    )

                    a_score = get_string_log_prob("Debater_A", outputs.scores, i)
                    b_score = get_string_log_prob("Debater_B", outputs.scores, i)

                    decoded_outputs.append(
                        constants.DEFAULT_DEBATER_A_NAME if a_score > b_score else constants.DEFAULT_DEBATER_B_NAME
                    )
                    self.logger.info(f"Scores: A {a_score} - B {b_score}")

            self.logger.debug(f"inference in {str(round(time.time() - start, 2))}")

        return decoded_outputs

    def copy(self, alias: str, is_debater: Optional[bool] = None) -> LlamaModel:
        copy = LlamaModel(alias=alias, is_debater=self.is_debater if is_debater == None else is_debater)
        copy.is_debater = self.is_debater if is_debater == None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy
