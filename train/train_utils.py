from agents.models.llama_model import LlamaInput, LlamaModel
from agents.prompt import Prompt, PromptParser, PromptTag
from data.data import DataRow, RawDataset, SplitType
from train.row_converter import RowConverter

from utils.flash_attn_utils import replace_attn_with_flash_attn, upcast_layer_for_flash_attention
import utils.constants as constants

from pydantic import BaseModel
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from tqdm import tqdm
import pandas as pd
import torch
import yaml

from enum import Enum
from typing import Optional, Union
import os


class TrainingTarget(Enum):
    DEBATER = 1
    JUDGE = 2


class PromptConfig(BaseModel):
    prompts_file_path: str
    prompt_name: str


class LoggingAndSavingConfig(BaseModel):
    logging_steps: int
    output_dir: str


class TrainingHyperParameterConfig(BaseModel):
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    optim: str
    learning_rate: float
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str


class TrainingConfig(BaseModel):
    model_name: str
    prompt_config: PromptConfig
    logging_and_saving_config: Optional[LoggingAndSavingConfig]
    training_hyperparameters: Optional[TrainingHyperParameterConfig]
    target: Union[str, TrainingTarget]
    deepspeed: Optional[str]


class TrainUtils:
    @classmethod
    def parse_config(cls, config_name: str, config_filepath: str) -> TrainingConfig:
        with open(config_filepath) as f:
            loaded_yaml = yaml.safe_load(f)
        return TrainingConfig(**loaded_yaml[config_name])

    @classmethod
    def convert_row(
        cls, row: DataRow, prompts_file_path: str, prompt_name: str, target: TrainingTarget
    ) -> list[dict[str, str]]:
        if target == TrainingTarget.DEBATER:
            return RowConverter.convert_all_speeches_for_debater(
                row=row, prompts_file_path=prompts_file_path, prompt_name=prompt_name
            )
        elif target == TrainingTarget.JUDGE:
            return RowConverter.convert_all_speeches_for_judge(
                row=row, prompts_file_path=prompts_file_path, prompt_name=prompt_name
            )
        else:
            raise Exception(f"Tried to train on an ineligible training target of {target}. This line should not be reached.")

    @classmethod
    def format_instruction(cls, llama_dictionary: dict[str, list[str]]) -> str:
        instructions = []
        for instruction_val, input_val, extra_suffix in zip(
            llama_dictionary.get("instruction"), llama_dictionary.get("input"), llama_dictionary.get("extra_suffix")
        ):
            instructions.append(
                LlamaModel.generate_input_str(
                    LlamaInput(instruction=instruction_val, input=input_val, extra_suffix=extra_suffix)
                )
            )
        return instructions

    @classmethod
    def convert_dataset(
        cls, raw_dataset: RawDataset, prompts_file_path: str, prompt_name: str, target: TrainingTarget
    ) -> Dataset:
        llama_input_lists = [
            TrainUtils.convert_row(row=row, prompts_file_path=prompts_file_path, prompt_name=prompt_name, target=target)
            for row in raw_dataset.get_data(split=SplitType.TRAIN)
        ]
        llama_inputs = [item for llama_input_list in llama_input_lists for item in llama_input_list]
        df = pd.DataFrame(data=llama_inputs)

        if target == TrainingTarget.JUDGE:
            df = df.iloc[: int(len(df) / 4)]  # TODO: should make configurable
        return Dataset.from_pandas(df)

    @classmethod
    def load_model(cls, config: TrainingConfig, is_local: bool = False):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        if not is_local:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            return AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=config.model_name,
                quantization_config=bnb_config,
                use_cache=False,
                device_map=device_map,
                trust_remote_code=True,
                use_flash_attention_2=True,
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                revision="main",
            )

    @classmethod
    def get_tokenizer(cls, config: TrainingConfig) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # change this
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    @classmethod
    def get_trainer(cls, config: TrainingConfig, raw_dataset: RawDataset, is_local: bool = False) -> SFTTrainer:
        replace_attn_with_flash_attn()
        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(config=config, is_local=is_local)

        training_args = TrainingArguments(
            output_dir=config.logging_and_saving_config.output_dir,
            num_train_epochs=config.training_hyperparameters.num_train_epochs,
            per_device_train_batch_size=config.training_hyperparameters.per_device_train_batch_size,
            gradient_accumulation_steps=config.training_hyperparameters.gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim=config.training_hyperparameters.optim,
            logging_steps=config.logging_and_saving_config.logging_steps,
            save_strategy="epoch",
            learning_rate=config.training_hyperparameters.learning_rate,
            max_grad_norm=config.training_hyperparameters.max_grad_norm,
            warmup_ratio=config.training_hyperparameters.warmup_ratio,
            lr_scheduler_type=config.training_hyperparameters.lr_scheduler_type,
            disable_tqdm=False,
            ddp_find_unused_parameters=False,
            use_cpu=is_local,
            deepspeed=config.deepspeed if not is_local else None,  # change this
        )

        collator = DataCollatorForCompletionOnlyLM(
            response_template=tokenizer.encode("\n " + constants.INSTRUCTION_SUFFIX, add_special_tokens=False)[2:],
            tokenizer=tokenizer,
        )

        target = TrainingTarget[config.target.upper()]
        train_dataset = TrainUtils.convert_dataset(
            raw_dataset=raw_dataset,
            prompts_file_path=config.prompt_config.prompts_file_path,
            prompt_name=config.prompt_config.prompt_name,
            target=target,
        )

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
        model = upcast_layer_for_flash_attention(model, torch.bfloat16)

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config if not is_local else None,
            tokenizer=tokenizer,
            data_collator=collator,
            formatting_func=TrainUtils.format_instruction,
            max_seq_length=32_768,
            args=training_args,
        )

        torch.cuda.empty_cache()

        return trainer
