from agents.model import Model, ModelInput, RoleType
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

from typing import Union
import os
import time


class LlamaModel(Model):
    def __init__(self, file_path: str, is_debater: bool = True):
        torch.cuda.empty_cache()
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
        self.generator_pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, trust_remote_code=False, device_map="auto"
        )
        self.logger = LoggerUtils.get_default_logger(__name__)

    def predict(self, inputs: list[ModelInput], max_new_tokens=450) -> Union[str, list[str]]:
        input_str = "{}\n\n{}\n\n{}\n\n{}\n\n{}{}".format(
            constants.INSTRUCTION_PREFIX,
            "\n".join(model_input.content for model_input in filter(lambda x: x.role == RoleType.SYSTEM, inputs)),
            constants.INPUT_PREFIX,
            "\n".join(model_input.content for model_input in filter(lambda x: x.role != RoleType.SYSTEM, inputs)),
            constants.INSTRUCTION_SUFFIX,
            ("\n\n" + constants.JUDGING_PREFIX) if not self.is_debater else "",
        )

        self.model.eval()
        with torch.no_grad():
            start = time.time()
            sequences = self.generator_pipeline(
                input_str,
                max_new_tokens=max_new_tokens if self.is_debater else 5,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.1,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                num_return_sequences=1 if self.is_debater else 8,
            )

            self.logger.debug(f"inference in {str(round(time.time() - start, 2))}")

        if not self.is_debater:
            return [sequence["generated_text"] for sequence in sequences]
        return sequences[0]["generated_text"]
