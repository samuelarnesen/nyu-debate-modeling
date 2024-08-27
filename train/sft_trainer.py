from data import DatasetConfig, DataRow, RawDataset, SplitType
from models import LLMInput, LLModel, LLMType, ModelInput, SpeechStructure
from prompts import RoleType
from train.row_converter import RowConverter
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils import LoggingCallback, logger_utils  # TODO: REMOVE
import utils.constants as constants

from peft import prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import pandas as pd
import datasets
import torch

from typing import Any, Optional, Type
import json
import random
import sys

try:
    from utils.flash_attn_utils import replace_attn_with_flash_attn, upcast_layer_for_flash_attention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class SupervisedTrainer:
    """Class for training a model using Supervised Fine Tuning"""

    @classmethod
    def convert_dataset(
        cls, raw_datasets: list[RawDataset], config: TrainingConfig, tokenizer: AutoTokenizer
    ) -> datasets.Dataset:
        """Converts a dataset (abstraction used in this codebase) into a Dataset object (abstraction
        used by huggingface's trainer objects)"""

        def validate_structure(val: Any) -> bool:
            if not isinstance(val, list):
                print("fail 1")
                return False
            for item in val:
                if not isinstance(item, list):
                    print("fail 2")
                    return False
                for elem in item:
                    if not isinstance(elem, tuple):
                        print("fail 3")
                        print(type(elem[0]))
                        return False
                    if not isinstance(elem[1], str):
                        print("fail 4")
                        return False
                    if not isinstance(elem[0], list):
                        print("fail 5")
                        return False
                    for x in elem[0]:
                        if not isinstance(x, ModelInput):
                            print("fail 6")
                            return False
            return True

        dataset_configs = config.dataset
        if isinstance(dataset_configs, DatasetConfig):
            dataset_configs = [dataset_configs]

        output_structure = (
            SpeechStructure.OPEN_ENDED if config.target == TrainingTarget.DEBATER else SpeechStructure.DECISION
        )

        llm_inputs = []
        for idx, raw_dataset in enumerate(raw_datasets):
            speech_structure = config.speech_structure[idx % len(config.speech_structure)]
            transcript_lists = [
                RowConverter.convert_row(
                    row=row,
                    config=config,
                    dataset=raw_dataset,
                    speech_structure=speech_structure,
                    use_gold_labels=config.training_hyperparameters.supplemental.get("gold_labels", False),
                    use_minimal_output_format=config.training_hyperparameters.supplemental.get(
                        "use_minimal_output_format", False
                    ),
                )
                for i, row in enumerate(raw_dataset.get_data(split=config.dataset[idx].split_type))
            ]

            if validate_structure(val=transcript_lists):
                for transcript_list in transcript_lists:
                    for model_inputs, speech in transcript_list:
                        llm_inputs.append(
                            {
                                "instruction": LLModel.convert_to_input_string(
                                    input_list=model_inputs,
                                    tokenizer=tokenizer,
                                    speech_structure=output_structure,
                                ),
                                "output": speech,
                            }
                        )
            else:
                raise Exception("Data format was invalid")

        max_instruction_length = int((2 / 3) * len(llm_inputs))
        instruction_count = 0
        for dataset_config in filter(lambda x: not x.dataset_type.is_instantiable, dataset_configs):
            external_dataset = datasets.load_dataset(path=dataset_config.full_dataset_file_path, split="train")
            external_df = pd.DataFrame(external_dataset)
            for i, row in external_df.iterrows():
                if instruction_count < max_instruction_length and (row["instruction"] or row["input"]):
                    instruction_count += 1
                    llm_inputs.append(
                        {
                            "instruction": LLModel.convert_to_input_string(
                                input_list=[
                                    ModelInput(role=RoleType.SYSTEM, content=row["instruction"]),
                                    ModelInput(role=RoleType.USER, content=row["input"]),
                                ],
                                tokenizer=tokenizer,
                                speech_structure=output_structure,
                            ),
                            "output": row["output"],
                        }
                    )

        df = pd.DataFrame(data=llm_inputs)
        dataset = datasets.Dataset.from_pandas(df).shuffle()
        return dataset

    @classmethod
    def formatting_func(cls, llm_dictionary: dict[str, list[str]]) -> str:
        formatted = []
        for instruction, output in zip(llm_dictionary["instruction"], llm_dictionary["output"]):
            formatted.append(instruction + output.strip())
        return formatted

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
            tokenizer=tokenizer,
            config=config,
        )

        peft_config = TrainUtils.get_peft_config(config) if not is_local else None
        if peft_config:
            # model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
            if FLASH_ATTENTION_AVAILABLE:
                model = upcast_layer_for_flash_attention(model, torch.bfloat16).to("cuda")

        if not is_test:
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                peft_config=peft_config,
                tokenizer=tokenizer,
                data_collator=collator,
                formatting_func=SupervisedTrainer.formatting_func,
                max_seq_length=config.max_length,
                callbacks=[LoggingCallback],
                args=training_args,
            )

            torch.cuda.empty_cache()

            return trainer
        return None
