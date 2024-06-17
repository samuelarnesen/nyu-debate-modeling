from data import DataRow, JudgePreferencesLoader, JudgePreferencesDataset, RawDataset, SplitType
from debate import BestOfNDebater, Debater, DebateRound, Judge, QuestionMetadata
from models import BestOfNConfig, OpenAIModel, RandomModel
from prompts import PromptConfig, PromptParser
from train.impl import SmoothedDPOTrainer
from train.dpo_trainer import DirectPreferenceTrainer
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils import LoggingCallback, logger_utils
import utils.constants as constants

from datasets import Dataset
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import DPOTrainer
import pandas as pd
import torch

from typing import Optional
import copy
import json

try:
    from utils.flash_attn_utils import (
        replace_attn_with_flash_attn,
        upcast_layer_for_flash_attention,
    )

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class IterativeDirectPreferenceTrainer(DirectPreferenceTrainer):
    """Class for iteratively training a model using Direct Preference Optimization"""

    DEFAULT_DEBATER_ALIAS = "default-debater"
    DEFAULT_JUDGE_ALIAS = "default-judge"

    def __init__(self, config: TrainingConfig, smooth: bool = True, is_local: bool = False):
        self.is_local = is_local
        self.tokenizer = TrainUtils.get_tokenizer(config=config, is_local=is_local)
        self.model = TrainUtils.load_model(config=config, is_local=is_local, requires_value_head=False)

        if FLASH_ATTENTION_AVAILABLE:
            self.model = upcast_layer_for_flash_attention(self.model, torch.bfloat16)

        self.peft_config = TrainUtils.get_peft_config(config=config)

        self.judge_model = OpenAIModel(
            alias=IterativeDirectPreferenceTrainer.DEFAULT_JUDGE_ALIAS,
            is_debater=False,
            endpoint="ft:gpt-4-0613:nyu-arg::90NW3Tbx",
        )  # make configurable
        if is_local:
            self.judge_model = RandomModel(alias=IterativeDirectPreferenceTrainer.DEFAULT_JUDGE_ALIAS, is_debater=False)

        self.random_judge_model = RandomModel(alias=IterativeDirectPreferenceTrainer.DEFAULT_JUDGE_ALIAS, is_debater=False)

        self.dataset = TrainUtils.create_datasets(config=config)[0]

        self.config = config
        self.trainer_cls = SmoothedDPOTrainer if smooth else DPOTrainer
        self.logger = logger_utils.get_default_logger(__name__)

    def train(self, epoch_size: int = 128):
        for epoch in range(self.config.training_hyperparameters.steps):
            self.step(step_count=epoch, epoch_size=epoch_size)

    def step(self, step_count: int, epoch_size: int):
        output_suffix = f"/checkpoint-{step_count}" if step_count < self.config.training_hyperparameters.steps - 1 else ""
        output_name = f"{self.config.logging_and_saving_config.output_dir}{output_suffix}"
        training_args = TrainingArguments(
            output_dir=output_name,
            num_train_epochs=self.config.training_hyperparameters.num_train_epochs,
            per_device_train_batch_size=self.config.training_hyperparameters.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training_hyperparameters.gradient_accumulation_steps,
            gradient_checkpointing=True,
            logging_steps=self.config.logging_and_saving_config.logging_steps,
            save_strategy="no"
            if self.config.training_hyperparameters.steps > 1 or self.config.training_hyperparameters.num_train_epochs == 1
            else "epoch",
            learning_rate=self.config.training_hyperparameters.learning_rate * (0.8**step_count),
            disable_tqdm=False,
            ddp_find_unused_parameters=False,
            optim=self.config.training_hyperparameters.optim,
            lr_scheduler_type=self.config.training_hyperparameters.lr_scheduler_type,
            use_cpu=self.is_local,
        )
        self.logger.warn(f"Generating samples for epoch {step_count}")
        train_dataset = self.get_samples(start_idx=step_count * epoch_size, epoch_size=epoch_size)
        self.logger.warn(f"Training for epoch {step_count}")

        trainer = self.trainer_cls(
            model=self.model,
            ref_model=None,
            loss_type="bon",
            max_length=16384,
            max_prompt_length=16384,
            beta=self.config.training_hyperparameters.kl_penalty,
            alpha=self.config.training_hyperparameters.supplemental.get("alpha", 0.005),
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
            callbacks=[LoggingCallback],
        )
        trainer.train()
        trainer.save_model()

        self.model = trainer.model

    def get_samples(self, start_idx: int, epoch_size: int) -> Dataset:
        if isinstance(self.dataset, JudgePreferencesDataset):
            return DirectPreferenceTrainer.convert_dataset([self.dataset])

        samples = []
        for i in range(epoch_size):
            new_samples = self.generate_one_round_samples(idx=start_idx + i)
            samples.extend(new_samples)

        return DirectPreferenceTrainer.convert_dataset(
            [JudgePreferencesDataset(train_data=samples, val_data=[], test_data=[])]
        )

    def generate_one_round_samples(self, idx: int):
        self.logger.warn(f"Starting round {idx}")
        example = self.dataset.get_example(idx=idx, split=SplitType.TRAIN)

        llm_class = TrainUtils.get_llm_class(self.config)
        internal_model = llm_class(
            alias=IterativeDirectPreferenceTrainer.DEFAULT_DEBATER_ALIAS,
            file_path=None,
            is_debater=True,
        )
        internal_model.model = self.model
        internal_model.tokenizer = self.tokenizer
        internal_model.generation_config = internal_model.create_default_generation_config(
            is_debater=True, do_sample=True, add_penalties=False
        )
        internal_model.instantiated_model = True
        internal_model.is_debater = True

        topic = example.question
        position = example.positions[0]
        opponent_position = example.positions[1]
        background_text = example.background_text
        title = example.story_title
        correct_index = example.correct_index
        speeches = example.speeches

        debate_identifier = f"{title}_{topic}"

        config_a = PromptConfig(
            name=constants.DEFAULT_DEBATER_A_NAME,
            opponent_name=constants.DEFAULT_DEBATER_B_NAME,
            position=position,
            opponent_position=opponent_position,
            topic=topic,
            background_text=background_text,
        )

        config_b = PromptConfig(
            name=constants.DEFAULT_DEBATER_B_NAME,
            opponent_name=constants.DEFAULT_DEBATER_A_NAME,
            position=opponent_position,
            opponent_position=position,
            topic=topic,
            background_text=background_text,
        )

        prompt_a = PromptParser.parse(
            prompt_config=config_a,
            prompts_file_path=self.config.prompt_config.file_path,
            name=self.config.speech_structure[0].default_prompt_name or self.config.prompt_config.default_prompt_name,
        )

        prompt_b = PromptParser.parse(
            prompt_config=config_b,
            prompts_file_path=self.config.prompt_config.file_path,
            name=self.config.speech_structure[0].default_prompt_name or self.config.prompt_config.default_prompt_name,
        )

        prompt_judge = PromptParser.parse(
            prompt_config=config_a,
            prompts_file_path=self.config.prompt_config.file_path,
            name=self.config.speech_structure[0].default_prompt_name or self.config.prompt_config.default_prompt_name,
        )

        question_metadata = QuestionMetadata(
            first_debater_correct=correct_index == 0,
            question_idx=idx,
            background_text=background_text,
            question=topic,
            first_debater_answer=position,
            second_debater_answer=opponent_position,
            debate_identifier=debate_identifier,
        )

        num_speeches = 1  # change this later
        original_debater_a = Debater(
            name=constants.DEFAULT_DEBATER_A_NAME,
            prompt=prompt_a,
            model=internal_model,
            num_speeches=num_speeches,
            speech_format=self.config.speech_structure[0].debater_format.get_speech_format(
                name=constants.DEFAULT_DEBATER_A_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
            ),
            quotes_require_validation=True,
        )

        original_debater_b = Debater(
            name=constants.DEFAULT_DEBATER_B_NAME,
            prompt=prompt_b,
            model=internal_model,
            num_speeches=num_speeches,
            speech_format=self.config.speech_structure[0].debater_format.get_speech_format(
                name=constants.DEFAULT_DEBATER_B_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
            ),
            quotes_require_validation=True,
        )

        best_of_n_judge = Judge(
            name=constants.DEFAULT_JUDGE_NAME,
            prompt=prompt_judge,
            model=self.judge_model,
            speech_format=self.config.speech_structure[0].judge_format.get_speech_format(
                name=constants.DEFAULT_JUDGE_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
                flipped=False,
            ),
            num_speeches=num_speeches,
        )

        random_judge = Judge(
            name=constants.DEFAULT_JUDGE_NAME,
            prompt=prompt_judge,
            model=self.random_judge_model,
            speech_format=self.config.speech_structure[0].judge_format.get_speech_format(
                name=constants.DEFAULT_JUDGE_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
                flipped=False,
            ),
            num_speeches=num_speeches,
        )

        non_random_judge = Judge(
            name=constants.DEFAULT_JUDGE_NAME,
            prompt=prompt_judge,
            model=self.judge_model,
            speech_format=self.config.speech_structure[0].judge_format.get_speech_format(
                name=constants.DEFAULT_JUDGE_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
                flipped=False,
            ),
            num_speeches=num_speeches,
        )

        debater_a = BestOfNDebater(
            debater=original_debater_a,
            opposing_debater=original_debater_b,
            judge=non_random_judge,
            best_of_n_config=BestOfNConfig(
                n=2,
                opponent_n=1,
                maxmin=False,
            ),
            background_text=background_text,
        )

        debater_b = BestOfNDebater(
            debater=original_debater_b,
            opposing_debater=original_debater_a,
            judge=non_random_judge,
            best_of_n_config=BestOfNConfig(
                n=2,
                opponent_n=1,
                maxmin=False,
            ),
            background_text=background_text,
        )

        debate_round = DebateRound(
            first_debater=debater_a,
            second_debater=debater_b,
            judge=random_judge,
            metadata=[question_metadata],
        )

        summary = debate_round()[0]
        transcript_json = random_judge.transcripts[0].json_value()
        return JudgePreferencesLoader.process_row(transcript_json)