from train.train_utils import TrainUtils, TrainingConfig
from utils.flash_attn_utils import replace_attn_with_flash_attn, upcast_layer_for_flash_attention

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer, SFTTrainer
import torch

from enum import Enum


class TrainerType(Enum):
    SFT = 1
    DPO = 2


class Trainer:
    @classmethod
    def get_sft_trainer(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        raw_dataset: RawDataset,
        is_local: bool = False,
    ) -> SFTTrainer:
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

    @classmethod
    def get_dpo_trainer(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        raw_dataset: RawDataset,
        is_local: bool = False,
    ) -> DPOTrainer:
        pass

    @classmethod
    def get_trainer(self, trainer_type: TrainerType, training_config: TrainingConfig):
        replace_attn_with_flash_attn()

        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(config=config, is_local=is_local)

        model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
        model = upcast_layer_for_flash_attention(model, torch.bfloat16)
        pass
