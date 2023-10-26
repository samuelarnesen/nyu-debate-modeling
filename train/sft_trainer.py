from agents.models.llama_model import LlamaInput, LlamaModel
from data.data import DataRow, RawDataset, SplitType
from train.row_converter import RowConverter
from train.train_utils import TrainingConfig, TrainingTarget
import utils.constants as constants

from datasets import Dataset
import pandas as pd
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import torch

try:
    from utils.flash_attn_utils import replace_attn_with_flash_attn, upcast_layer_for_flash_attention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class SFTTrainer:
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
            SFTTrainer.convert_row(row=row, prompts_file_path=prompts_file_path, prompt_name=prompt_name, target=target)
            for row in raw_dataset.get_data(split=SplitType.TRAIN)
        ]
        llama_inputs = [item for llama_input_list in llama_input_lists for item in llama_input_list]
        df = pd.DataFrame(data=llama_inputs)

        return Dataset.from_pandas(df)

    @classmethod
    def get_trainer(
        cls,
        config: TrainingConfig,
        raw_dataset: RawDataset,
        is_local: bool = False,
    ) -> SFTTrainer:
        if FLASH_ATTENTION_AVAILABLE:
            replace_attn_with_flash_attn()
        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(config=config.model_name, is_local=is_local)

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

        target = TrainingTarget[config.target.upper()]
        train_dataset = SFTTrainer.convert_dataset(
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

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config if not is_local else None,
            tokenizer=tokenizer,
            data_collator=collator,
            formatting_func=SFTTrainer.format_instruction,
            max_seq_length=32_768,
            args=training_args,
        )

        torch.cuda.empty_cache()

        return trainer
