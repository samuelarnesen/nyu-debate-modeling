from agents.model import Model, ModelInput, RoleType
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

import os
import time


class LlamaModel(Model):
    def __init__(self, file_path: str):
        torch.cuda.empty_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(file_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # for open-ended generation

        self.model = AutoModelForCausalLM.from_pretrained(
            file_path, device_map="auto", trust_remote_code=False, local_files_only=True, torch_dtype=torch.float16
        )
        self.generator_pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, trust_remote_code=False, device_map="auto"
        )
        self.logger = LoggerUtils.get_default_logger(__name__)

    def predict(self, inputs: list[ModelInput], max_new_tokens=1_500) -> str:
        input_str = "{}\n\n{}\n\n{}\n\n{}\n\n{}".format(
            constants.INSTRUCTION_PREFIX,
            "\n".join(model_input.content for model_input in filter(lambda x: x.role == RoleType.SYSTEM, inputs)),
            constants.INPUT_PREFIX,
            "\n".join(model_input.content for model_input in filter(lambda x: x.role != RoleType.SYSTEM, inputs)),
            constants.INSTRUCTION_SUFFIX,
        )

        self.model.eval()
        with torch.no_grad():
            start = time.time()
            sequences = self.generator_pipeline(
                input_str,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=2,
                temperature=0.4,
                top_p=0.9,
            )

            self.logger.debug(f"inference in {str(round(time.time() - start, 2))}")

        return sequences[0]["generated_text"].split(constants.INSTRUCTION_SUFFIX)[-1].strip()
