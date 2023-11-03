from agents.models.llama_model import LlamaInput, LlamaModel
from data.data import DataRow, RawDataset, SplitType
from train.row_converter import RowConverter
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
import utils.constants as constants
from utils.logger_utils import LoggerUtils

from datasets import Dataset
from pydantic import BaseModel
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
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
    @classmethod
    def convert_dataset(cls, raw_dataset: RawDataset) -> Dataset:
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
        logger = LoggerUtils.get_default_logger(__name__)

        if FLASH_ATTENTION_AVAILABLE:
            replace_attn_with_flash_attn()
        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(model_name=config.model_name, is_local=is_local)

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
            use_cpu=is_local,
        )

        train_dataset = DirectPreferenceTrainer.convert_dataset(raw_dataset=raw_dataset)

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
        if FLASH_ATTENTION_AVAILABLE:
            model = upcast_layer_for_flash_attention(model, torch.bfloat16)

        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            max_length=1024,
            max_prompt_length=16384,
            beta=0.1,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

        torch.cuda.empty_cache()

        return trainer
