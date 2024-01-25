from agents import LLMType, LLModel, ModelStub, ScratchpadConfig, TokenizerStub
from data import DatasetConfig, DatasetType, LoaderUtils, RawDataset
from prompts import PromptLoadingConfig
import utils.constants as constants

from peft import LoraConfig, PeftConfig, PeftType, PromptTuningInit, PromptTuningConfig, TaskType
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
import torch
import yaml

from enum import Enum
from typing import Optional, Type, Union
import os


class TrainingTarget(Enum):
    DEBATER = 1
    JUDGE = 2


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
    peft_type: PeftType | str
    steps: Optional[int]


class TrainingConfig(BaseModel):
    model_name: str
    reference_model_name: Optional[str]
    llm_type: str = "llama"
    prompt_config: PromptLoadingConfig = PromptLoadingConfig()
    logging_and_saving_config: Optional[LoggingAndSavingConfig]
    training_hyperparameters: Optional[TrainingHyperParameterConfig]
    target: Optional[str | TrainingTarget] = None
    dataset: Optional[DatasetConfig]
    opening_speeches_only: bool = False
    requires_token: bool = False
    max_length: int = constants.MAX_LENGTH
    scratchpad_config: ScratchpadConfig = ScratchpadConfig()


class TrainUtils:
    @classmethod
    def create_dataset(cls, config: TrainingConfig, deduplicate: bool = False) -> RawDataset:
        """
        Constructs a dataset that will later be converted into a training dataset.

        Params:
            config: the configuration containing the prompt text and training hyperparameters
            deduplicate: whether only one example from each prompt should be used

        Returns:
            dataset: a dataset object that can later be used as a training dataset
        """
        dataset_config = config.dataset
        dataset_type = DatasetType[dataset_config.dataset_type.upper()]
        loader_cls = LoaderUtils.get_loader_type(dataset_type)
        return loader_cls.load(
            full_dataset_filepath=dataset_config.full_dataset_file_path,
            train_filepath=dataset_config.train_file_path,
            val_filepath=dataset_config.val_file_path,
            test_filepath=dataset_config.test_file_path,
            supplemental_file_paths=dataset_config.supplemental_file_paths,
            deduplicate=deduplicate,
        )

    @classmethod
    def parse_config(cls, config_name: Optional[str], config_filepath: str) -> TrainingConfig:
        """Loads a yaml file and converts it into a training configuration"""
        with open(config_filepath) as f:
            loaded_yaml = yaml.safe_load(f)
        config_name = config_name or [key for key in config_name][0]
        return TrainingConfig(**loaded_yaml[config_name])

    @classmethod
    def get_peft_config(cls, config: TrainingConfig) -> Optional[PeftConfig]:
        """Gets the configuration from parameter efficient fine tuning"""
        if not config.training_hyperparameters.peft_type.upper():
            return None
        peft_type = PeftType[config.training_hyperparameters.peft_type.upper()]
        llm_class = TrainUtils.get_llm_class(config=config)
        if peft_type == PeftType.LORA:
            return LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=llm_class.TARGET_MODULES,
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
    def load_model(
        cls, config: TrainingConfig, is_local: bool = False, requires_value_head: bool = False
    ) -> AutoModelForCausalLM:
        """
        Loads a model using the specified configuration.

        Params:
            config: the configuration covering the training hyperparameters
            is_local: whether it's being run on a cpu
            requires_value_head: whether we need to wrap the model with a layer that generates scalar values.
                (Only used for PPO training for now)

        Returns:
            model: a model loaded from huggingface
        """

        if not is_local:
            model = LLModel.instantiate_hf_model(
                file_path=config.model_name, requires_token=config.requires_token, use_cache=False
            )
            model.config.max_position_embeddings = config.max_length
            model.config.transformers_version = "4.34.0"
            model.generation_config.transformers_version = "4.34.0"
            if requires_value_head:
                peft_config = TrainUtils.get_peft_config(config=config)
                return AutoModelForCausalLMWithValueHead.from_pretrained(
                    pretrained_model_name_or_path=model,
                    quantization_config=bnb_config,
                    use_cache=False,
                    device_map=device_map,
                    trust_remote_code=True,
                    use_flash_attention_2=True,
                    peft_config=peft_config,
                )
            return model
        else:
            return ModelStub()

    @classmethod
    def get_tokenizer(cls, config: TrainingConfig, is_local: bool = False) -> AutoTokenizer:
        """Gets the tokenizer associated with the specified model"""
        if not is_local:
            return LLModel.instantiate_tokenizer(file_path=config.model_name, requires_token=config.requires_token)
        else:
            return TokenizerStub()

    @classmethod
    def get_llm_class(cls, config: TrainingConfig):
        return LLMType[config.llm_type.upper()].get_llm_class()
