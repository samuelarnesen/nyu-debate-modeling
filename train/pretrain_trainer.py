from data import RawDataset, SplitType
from train.train_utils import TrainUtils, TrainingConfig
from utils import LoggingCallback, logger_utils

from datasets import Dataset
from peft import prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import pandas as pd
import torch

from typing import Optional
import os

try:
    from utils.flash_attn_utils import replace_attn_with_flash_attn, upcast_layer_for_flash_attention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class PretrainTrainer:
    """Class for pretraining a model using a causal language modeling objective"""

    MAX_LENGTH = 4096
    CONTENT_FIELD = "content"

    @classmethod
    def convert_dataset(cls, raw_dataset: RawDataset, tokenizer: AutoTokenizer) -> Dataset:
        """Converts a dataset (abstraction used in this codebase) into a Dataset object (abstraction
        used by huggingface's trainer objects)"""

        def tokenize(example: str):
            outputs = tokenizer(
                example[PretrainTrainer.CONTENT_FIELD],
                truncation=True,
                max_length=PretrainTrainer.MAX_LENGTH,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == PretrainTrainer.MAX_LENGTH:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        dataset = Dataset.from_pandas(
            pd.DataFrame(
                data=[
                    {PretrainTrainer.CONTENT_FIELD: example.background_text}
                    for example in raw_dataset.get_data(split=SplitType.TRAIN)
                ]
            )
        ).shuffle()

        return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    @classmethod
    def get_trainer(
        cls, config: TrainingConfig, raw_dataset: Optional[RawDataset] = None, is_local: bool = False
    ) -> Trainer:
        """
        Generates a Trainer object.

        Params:
            config: configuration specifying the prompt setup and hyperparameters for the training run.
            raw_dataset: dataset to use for training
            is_local: whether this is being run on a cpu

        Returns:
            trainer: One can call trainer.train() to then run the training loop.
        """
        logger = logger_utils.get_default_logger(__name__)
        if not raw_dataset:
            raw_dataset = TrainUtils.create_dataset(config=config)

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
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        train_dataset = PretrainTrainer.convert_dataset(raw_dataset=raw_dataset, tokenizer=tokenizer)

        peft_config = TrainUtils.get_peft_config(config)
        model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
        if FLASH_ATTENTION_AVAILABLE:
            model = upcast_layer_for_flash_attention(model, torch.bfloat16)

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[LoggingCallback],
            args=training_args,
        )

        torch.cuda.empty_cache()

        return trainer
