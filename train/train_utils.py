from data.data import DatasetType, RawDataset
from data.loaders.loader_utils import LoaderUtils

from peft import LoraConfig, PeftConfig, PeftType, PromptTuningInit, PromptTuningConfig, TaskType
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
import torch
import yaml

from enum import Enum
from typing import Optional, Union
import os

try:
    from utils.flash_attn_utils import replace_attn_with_flash_attn, upcast_layer_for_flash_attention
except ImportError as e:
    print("Running without flash attention")


class TrainingTarget(Enum):
    DEBATER = 1
    JUDGE = 2


class PromptConfig(BaseModel):
    prompts_file_path: str
    prompt_name: str
    dynamic_prompts_file_path: Optional[str]
    dynamic_prompt_name: Optional[str]
    annotations_file_path: Optional[str]


class LoggingAndSavingConfig(BaseModel):
    logging_steps: int
    output_dir: str
    merge_output_dir: Optional[str]


class TrainingHyperParameterConfig(BaseModel):
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    optim: str
    learning_rate: float
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str
    peft_type: Union[PeftType, str]
    steps: Optional[int]


class DatasetConfig(BaseModel):
    dataset_type: str
    full_dataset_file_path: Optional[str]
    train_file_path: Optional[str]
    val_file_path: Optional[str]
    test_file_path: Optional[str]
    annotations_file_path: Optional[str]
    split_type: Optional[str]


class TrainingConfig(BaseModel):
    model_name: str
    reference_model_name: Optional[str]
    prompt_config: PromptConfig
    logging_and_saving_config: Optional[LoggingAndSavingConfig]
    training_hyperparameters: Optional[TrainingHyperParameterConfig]
    target: Optional[Union[str, TrainingTarget]]
    dataset: Optional[DatasetConfig]
    deepspeed: Optional[str]
    opening_speeches_only: Optional[bool]


class TrainUtils:
    @classmethod
    def create_dataset(cls, config: TrainingConfig) -> RawDataset:
        dataset_config = config.dataset
        dataset_type = DatasetType[dataset_config.dataset_type.upper()]
        loader_cls = LoaderUtils.get_loader_type(dataset_type)
        return loader_cls.load(
            full_dataset_filepath=dataset_config.full_dataset_file_path,
            train_filepath=dataset_config.train_file_path,
            val_filepath=dataset_config.val_file_path,
            test_filepath=dataset_config.test_file_path,
            annotations_file_path=dataset_config.annotations_file_path,
        )

    @classmethod
    def parse_config(cls, config_name: str, config_filepath: str) -> TrainingConfig:
        with open(config_filepath) as f:
            loaded_yaml = yaml.safe_load(f)
        return TrainingConfig(**loaded_yaml[config_name])

    @classmethod
    def get_peft_config(cls, config: TrainingConfig) -> Optional[PeftConfig]:
        if not config.training_hyperparameters.peft_type.upper():
            return None
        peft_type = PeftType[config.training_hyperparameters.peft_type.upper()]
        if peft_type == PeftType.LORA:
            return LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        elif peft_type == PeftType.PROMPT_TUNING:
            return PromptTuningConfig(
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=16,
                prompt_tuning_init_text="Now give your speech:",
                tokenizer_name_or_path=config.model_name,
                task_type=TaskType.CAUSAL_LM,
            )

    @classmethod
    def load_model(cls, config: TrainingConfig, is_local: bool = False, requires_value_head: bool = False):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        if not is_local:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=config.model_name,
                quantization_config=bnb_config,
                use_cache=False,
                device_map=device_map,
                trust_remote_code=True,
                use_flash_attention_2=True,
            )

            if requires_value_head:
                peft_config = TrainUtils.get_peft_config(config=config)
                return AutoModelForCausalLMWithValueHead.from_pretrained(
                    pretrained_model_name_or_path=config.model_name,
                    quantization_config=bnb_config,
                    use_cache=False,
                    device_map=device_map,
                    trust_remote_code=True,
                    use_flash_attention_2=True,
                    peft_config=peft_config,
                )
            else:
                return model
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
