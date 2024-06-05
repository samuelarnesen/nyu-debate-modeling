from data import DataRow, JudgePreferencesLoader, JudgePreferencesDataset, RawDataset, SplitType
from models import BestOfNConfig
from train.impl import SmoothedDPOTrainer
from train.dpo_trainer import DirectPreferenceTrainer
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils import LoggingCallback, logger_utils
import utils.constants as constants

from datasets import Dataset
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
import pandas as pd
import torch

from typing import Optional

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

    def __init__(self, config: TrainingConfig, smooth: bool = True):
        self.tokenizer = TrainUtils.get_tokenizer(config=config, is_local=is_local)
        self.model = TrainUtils.load_model(config=config, is_local=is_local)

        if FLASH_ATTENTION_AVAILABLE:
            model = upcast_layer_for_flash_attention(model, torch.bfloat16)

        self.peft_config = TrainUtils.get_peft_config(config=config)

        self.judge_model = OpenAIModel(
            alias=PPOTrainerWrapper.DEFAULT_JUDGE_ALIAS, is_debater=False, endpoint="ft:gpt-4-0613:nyu-arg::90NW3Tbx"
        )  # make configurable

        if not raw_datasets:
            dataset = TrainUtils.create_datasets(config=config)
        self.dataset = dataset[0]

        self.config = config
        self.trainer_cls = SmoothedDPOTrainer if smooth else DPOTrainer
        self.logger = logger_utils.get_default_logger(__name__)

    def train(self, epoch_size: int):
        for epoch in range(config.training_hyperparameters.num_train_epochs):
            self.step(step_count=epoch, epoch_size=epoch_size)

    def step(self, step_count: int, epoch_size: int):
        training_args = TrainingArguments(
            output_dir=f"{config.logging_and_saving_config.output_dir}/checkpoint-{step_count}",
            num_train_epochs=config.training_hyperparameters.num_train_epochs,
            per_device_train_batch_size=config.training_hyperparameters.per_device_train_batch_size,
            gradient_accumulation_steps=config.training_hyperparameters.gradient_accumulation_steps,
            gradient_checkpointing=True,
            logging_steps=config.logging_and_saving_config.logging_steps,
            save_strategy="NO",
            learning_rate=config.training_hyperparameters.learning_rate,
            disable_tqdm=False,
            ddp_find_unused_parameters=False,
            optim=config.training_hyperparameters.optim,
            lr_scheduler_type=config.training_hyperparameters.lr_scheduler_type,
            use_cpu=is_local,
        )
        train_dataset = self.get_samples(start_idx=step_count * epoch_size, epoch_size=epoch_size)
        trainer = self.trainer_cls(
            model=self.model,
            ref_model=None,
            max_length=16384,
            max_prompt_length=16384,
            beta=0.1,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
            callbacks=[LoggingCallback],
        )
        trainer.train()
        trainer.save_model()

    def get_samples(self, start_idx: int, epoch_size: int) -> Dataset:
        samples = []
        for i in range(epoch_size):
            new_samples = self.generate_one_round_samples(idx=start_idx + i)
            samples.extend(new_samples)

        return DirectPreferenceTrainer.convert_dataset([JudgePreferencesDataset(train_data=samples)])

    def generate_one_round_samples(self, idx: int):
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
            prompt_config=self.config_a,
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
        debater_a = Debater(
            name=constants.DEFAULT_DEBATER_A_NAME,
            prompt=prompt_a,
            model=internal_model,
            num_speeches=num_speeches,
            speech_format=config.speech_structure[0].debater_format.get_speech_format(
                name=constants.DEFAULT_DEBATER_A_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
            ),
            quotes_require_validation=True,
            best_of_n_config=BestOfNConfig(
                n=2,
                opponent_n=1,
                maxmin=False,
            ),
        )

        debater_b = Debater(
            name=constants.DEFAULT_DEBATER_B_NAME,
            prompt=prompt_b,
            model=internal_model,
            num_speeches=num_speeches,
            speech_format=config.speech_structure[0].debater_format.get_speech_format(
                name=constants.DEFAULT_DEBATER_B_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
            ),
            quotes_require_validation=True,
            best_of_n_config=BestOfNConfig(
                n=2,
                opponent_n=1,
                maxmin=False,
            ),
        )

        judge = Judge(
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

        debate_round = DebateRound(
            first_debater=debater_a,
            second_debater=debater_b,
            judge=judge,
            metadata=[question_metadata],
        )

        summary = debate_round()
        return JudgePreferencesLoader.process_row(summary[0].judge.transcripts[0].dict())
