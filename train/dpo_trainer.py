from data import DataRow, RawDataset, SplitType
from train.row_converter import RowConverter
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils import LoggingCallback, LoggerUtils
import utils.constants as constants

from datasets import Dataset
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
import pandas as pd
import torch

try:
    from utils.flash_attn_utils import (
        replace_attn_with_flash_attn,
        upcast_layer_for_flash_attention,
    )

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class DirectPreferenceTrainer:
    """Class for training a model using Direct Preference Optimization"""

    @classmethod
    def convert_dataset(cls, raw_dataset: RawDataset) -> Dataset:
        """Converts a dataset (abstraction used in this codebase) into a Dataset object (abstraction
        used by huggingface's trainer objects)"""
        rows = [row.dict() for row in raw_dataset.get_data(split=SplitType.TRAIN)]
        df = pd.DataFrame(data=rows)
        return Dataset.from_pandas(df)

    @classmethod
    def get_trainer(
        cls,
        config: TrainingConfig,
        raw_dataset: RawDataset,
        is_local: bool = False,
    ) -> DPOTrainer:
        """
        Generates a Trainer object.

        Params:
            config: configuration specifying the prompt setup and hyperparameters for the training run.
            raw_dataset: dataset to use for training
            is_local: whether this is being run on a cpu

        Returns:
            dpo_trainer: One can call dpo_trainer.train() to then run the training loop.
        """

        if FLASH_ATTENTION_AVAILABLE:
            replace_attn_with_flash_attn()
        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(config=config, is_local=is_local)

        training_args = TrainingArguments(
            output_dir=config.logging_and_saving_config.output_dir,
            num_train_epochs=config.training_hyperparameters.num_train_epochs,
            per_device_train_batch_size=config.training_hyperparameters.per_device_train_batch_size,
            gradient_accumulation_steps=config.training_hyperparameters.gradient_accumulation_steps,
            gradient_checkpointing=True,
            logging_steps=config.logging_and_saving_config.logging_steps,
            save_strategy="epoch",
            learning_rate=config.training_hyperparameters.learning_rate,
            disable_tqdm=False,
            ddp_find_unused_parameters=False,
            optim=config.training_hyperparameters.optim,
            lr_scheduler_type=config.training_hyperparameters.lr_scheduler_type,
            use_cpu=is_local,
        )

        train_dataset = DirectPreferenceTrainer.convert_dataset(raw_dataset=raw_dataset)

        peft_config = TrainUtils.get_peft_config(config=config)

        if FLASH_ATTENTION_AVAILABLE:
            model = upcast_layer_for_flash_attention(model, torch.bfloat16)

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            max_length=16384,
            max_prompt_length=16384,
            beta=0.1,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[LoggingCallback],
            loss_type="ipo",
        )

        torch.cuda.empty_cache()

        return trainer
