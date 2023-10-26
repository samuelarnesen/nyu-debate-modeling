from agents.models.llama_model import LlamaInput, LlamaModel
from data.data import DataRow, RawDataset, SplitType
from train.row_converter import RowConverter
from train.train_utils import TrainingConfig, TrainingTarget
import utils.constants as constants

from datasets import Dataset
from pydantic import BaseModel
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer
import pandas as pd
import torch

try:
    from utils.flash_attn_utils import replace_attn_with_flash_attn, upcast_layer_for_flash_attention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class DPOTrainer:
    @classmethod
    def convert_dataset(cls, raw_dataset: RawDataset) -> Dataset:
        rows = [row.dict() for row in raw_dataset.get_data(split_type=SplitType.TRAIN)]
        df = pd.DataFrame(data=rows)
        return Dataset.from_pandas(df)

    @classmethod
    def get_trainer(
        cls,
        config: TrainingConfig,
        raw_dataset: RawDataset,
        is_local: bool = False,
    ) -> DPOTrainer:
        if FLASH_ATTENTION_AVAILABLE:
            replace_attn_with_flash_attn()
        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(config=config.model_name, is_local=is_local)
        reference_model = TrainUtils.load_model(config=config.reference_model_name, is_local=is_local)

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

        collator = DataCollatorForCompletionOnlyLM(
            response_template=tokenizer.encode("\n " + constants.INSTRUCTION_SUFFIX, add_special_tokens=False)[2:],
            tokenizer=tokenizer,
        )

        train_dataset = DPOTrainer.convert_dataset(
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
        if FLASH_ATTENTION_AVAILABLE:
            model = upcast_layer_for_flash_attention(model, torch.bfloat16)

        trainer = DPOTrainer(
            model=model,
            ref_model=reference_model,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

        torch.cuda.empty_cache()

        return trainer
