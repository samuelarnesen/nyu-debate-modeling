from agents import LLMInput, LLModel, LLMType
from data import DataRow, RawDataset, SplitType
from train.row_converter import RowConverter
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils import LoggingCallback
import utils.constants as constants

from datasets import Dataset
from peft import prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import pandas as pd
import torch

from typing import Optional, Type
import json

try:
    from utils.flash_attn_utils import replace_attn_with_flash_attn, upcast_layer_for_flash_attention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class SupervisedTrainer:
    """Class for training a model using Supervised Fine Tuning"""

    @classmethod
    def format_instruction(cls, llm_dictionary: dict[str, list[str]], llm_class: Type[LLModel]) -> str:
        """Converts a dataset row into a prompt string"""
        instructions = []
        for instruction_val, input_val, extra_suffix in zip(
            llm_dictionary.get("instruction"), llm_dictionary.get("input"), llm_dictionary.get("extra_suffix")
        ):
            instructions.append(
                llm_class.generate_input_str(
                    LLMInput(instruction=instruction_val, input=input_val, extra_suffix=extra_suffix),
                    llm_class.INSTRUCTION_PREFIX,
                    llm_class.INSTRUCTION_SUFFIX,
                )
            )
        return instructions

    @classmethod
    def convert_dataset(cls, raw_datasets: list[RawDataset], config: TrainingConfig) -> Dataset:
        """Converts a dataset (abstraction used in this codebase) into a Dataset object (abstraction
        used by huggingface's trainer objects)"""

        llm_inputs = []

        for idx, raw_dataset in enumerate(raw_datasets):
            speech_structure = config.speech_structure[idx % len(config.speech_structure)]
            llm_input_lists = [
                RowConverter.convert_row(row=row, config=config, dataset=raw_dataset, speech_structure=speech_structure)
                for i, row in enumerate(raw_dataset.get_data(split=config.dataset[idx].split_type))
            ]

            if llm_input_lists and isinstance(llm_input_lists[0], list) and isinstance(llm_input_lists[0][0], dict):
                llm_inputs += [
                    item
                    for llm_input_list in llm_input_lists
                    for item in filter(lambda x: x["extra_suffix"], llm_input_list)
                ]
            elif (
                llm_input_lists
                and isinstance(llm_input_lists[0], list)
                and isinstance(llm_input_lists[0][0], list)
                and isinstance(llm_input_lists[0][0][0], dict)
            ):
                llm_inputs += [
                    item
                    for llm_input_list in llm_input_lists
                    for item in filter(lambda x: x[-1]["role"] == "assistant", llm_input_list)
                ]
            else:
                raise Exception("Data format was invalid")

        df = pd.DataFrame(data=llm_inputs)

        str_versions = [json.dumps({"messages": llm_input}) for llm_input in llm_inputs]
        with open("/Users/samarnesen/nyu/scratch/debates-and-consultancies-combined.jsonl", "w") as f:
            str_version = "\n".join(str_versions)
            f.write(str_version)

        return Dataset.from_pandas(df).shuffle()

    @classmethod
    def get_trainer(
        cls,
        config: TrainingConfig,
        raw_datasets: Optional[list[RawDataset]] = None,
        is_local: bool = False,
        is_test: bool = False,
    ) -> Optional[SFTTrainer]:
        """
        Generates a Trainer object.

        Params:
            config: configuration specifying the prompt setup and hyperparameters for the training run.
            raw_dataset: dataset to use for training
            is_local: whether this is being run on a cpu
            is_test: whether to actually instantiate the trainer (if true, do not instantiate)

        Returns:
            sft_trainer: One can call dpo_trainer.train() to then run the training loop.
        """
        if FLASH_ATTENTION_AVAILABLE:
            replace_attn_with_flash_attn()

        if not raw_datasets:
            raw_datasets = TrainUtils.create_datasets(config=config)

        tokenizer = TrainUtils.get_tokenizer(config=config, is_local=is_local)
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

        llm_class = TrainUtils.get_llm_class(config=config)

        collator = DataCollatorForCompletionOnlyLM(
            response_template=tokenizer.encode("\n " + llm_class.INSTRUCTION_SUFFIX, add_special_tokens=False)[2:],
            tokenizer=tokenizer,
        )

        train_dataset = SupervisedTrainer.convert_dataset(
            raw_datasets=raw_datasets,
            config=config,
        )

        peft_config = TrainUtils.get_peft_config(config) if not is_local else None
        if peft_config:
            model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
            if FLASH_ATTENTION_AVAILABLE:
                model = upcast_layer_for_flash_attention(model, torch.bfloat16).to("cuda")

        if not is_test:
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                peft_config=peft_config,
                tokenizer=tokenizer,
                data_collator=collator,
                formatting_func=lambda x: SupervisedTrainer.format_instruction(x, llm_class),
                max_seq_length=config.max_length,
                callbacks=[LoggingCallback],
                args=training_args,
            )

            torch.cuda.empty_cache()

            return trainer
        return None
