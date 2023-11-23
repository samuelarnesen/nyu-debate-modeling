from __future__ import annotations

from agents.model import SpeechStructure
from agents.models.openai_model import OpenAIModel
from data.data import DataRow, RawDataset, SplitType
from train.row_converter import RowConverter
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils.logger_utils import LoggingCallback, LoggerUtils
from utils.string_utils import StringUtils
import utils.constants as constants

from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model  # TODO: check if we need this
from transformers import GenerationConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import pandas as pd
import torch
import tqdm

try:
    from utils.flash_attn_utils import (
        replace_attn_with_flash_attn,
        upcast_layer_for_flash_attention,
    )

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


class PPOTrainerWrapper:
    INSTRUCTION_COLUMN = "instruction"
    INPUT_COLUMN = "input"
    QUERY_COLUMN = "query"
    SCORE_COLUMN = "score"
    RESPONSE_COLUMN = "response"
    OPPONENT_QUERY_COLUMN = "opponent_query"
    JUDGE_QUERY_COLUMN = "judge_query"
    INPUT_IDS_COLUMN = "input_ids"
    EXTRA_SUFFIX_COLUMN = "extra_suffix"
    BATCH_SIZE = 8
    DEFAULT_JUDGE_ALIAS = "openai-judge"

    def __init__(self, ppo_trainer: PPOTrainer, config: TrainingConfig):
        self.ppo_trainer = ppo_trainer
        self.reward_model = OpenAIModel(alias=DEFAULT_JUDGE_ALIAS, is_debater=False)
        self.config = config

    def train(self):
        # NOTE: TODO: This only optimizes Debater_A
        for row in self.ppo_trainer.dataset:
            query_tensors = self.ppo_trainer.tokenizer([row[PPOTrainerWrapper.QUERY_COLUMN]], return_tensors="pt").input_ids
            opponent_query_tensors = self.ppo_trainer.tokenizer(
                [row[PPOTrainerWrapper.OPPONENT_QUERY_COLUMN]], return_tensors="pt"
            ).input_ids

            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            with torch.no_grad():
                opponent_response_tensors = ppo_trainer.model.generate(opponent_query_tensors, **generation_kwargs)

            input_lengths = (query_tensors.input_ids != self.tokenizer.pad_token_id).sum(axis=1)
            opponent_input_lengths = (opponent_query_tensors.input_ids != self.tokenizer.pad_token_id).sum(axis=1)

            reward_tensors = []
            for i in range(len(response_tensors)):
                decoded = StringUtils.clean_string(
                    self.tokenizer.decode(response_tensors.sequences[i, input_lengths[min(i, len(input_lengths) - 1)] :])
                )
                opponent_decoded = StringUtils.clean_string(
                    self.tokenizer.decode(
                        opponent_response_tensors.sequences[
                            i, opponent_input_lengths[min(i, len(opponent_input_lengths) - 1)] :
                        ]
                    )
                )
                texts.append((decoded, opponent_decoded))

                judge_input_text = (
                    row[PPOTrainerWrapper.JUDGE_QUERY_COLUMN]
                    .replace(RowConverter.get_dummy_name_for_speaker(constants.DEFAULT_DEBATER_A_NAME), decoded)
                    .replace(RowConverter.get_dummy_name_for_speaker(constants.DEFAULT_DEBATER_A_NAME), opponent_decoded)
                )

                reasoning = reward_model.predict(inputs=judge_input_text, max_new_tokens=450)
                reward = reward_model.predict(
                    inputs=[f"{judge_input_text}\n{reasoning}"],
                    max_new_tokens=15,
                    speech_structure=SpeechStructure.PREFERENCE,
                )
                reward_tensors.append(torch.tensor(reward))

            stats = ppo_trainer.step(queries=query_tensors, responses=response_tensors, scores=reward_tensors)
            ppo_trainer.log_stats(stats=stats, batch=batch, rewards=rewards_tensor)

    def save_model(self):
        self.ppo_trainer.save_model(config.logging_and_saving_config.output_dir)

    @classmethod
    def convert_dataset(cls, raw_dataset: RawDataset, config: TrainingConfig) -> Dataset:
        def construct_dataframe(inputs: list[dict[str, str]]):
            df = pd.DataFrame(data=inputs)
            df[PPOTrainerWrapper.QUERY_COLUMN] = (
                df[PPOTrainerWrapper.INSTRUCTION_COLUMN] + "\n" + df[PPOTrainerWrapper.INPUT_COLUMN]
            )
            df = df.drop(
                columns=[
                    PPOTrainerWrapper.INSTRUCTION_COLUMN,
                    PPOTrainerWrapper.INPUT_COLUMN,
                    PPOTrainerWrapper.EXTRA_SUFFIX_COLUMN,
                ]
            )
            return df

        debater_input_lists = [
            RowConverter.convert_row(row=row, config=config, target=TrainingTarget.DEBATER, dataset=raw_dataset, index=i)
            for i, row in enumerate(raw_dataset.get_data(split=SplitType.TRAIN))
        ]
        debater_inputs = [item for debater_input_list in debater_input_lists for item in debater_input_list]
        debater_a_inputs = [debater_inputs[i] for i in filter(lambda x: x % 2 == 0, range(len(debater_inputs)))]
        debater_b_inputs = [debater_inputs[i] for i in filter(lambda x: x % 2 == 1, range(len(debater_inputs)))]

        judge_input_lists = [
            RowConverter.convert_row(
                row=row, config=config, target=TrainingTarget.JUDGE, dataset=raw_dataset, index=i, use_dummy=True
            )
            for i, row in enumerate(raw_dataset.get_data(split=SplitType.TRAIN))
        ]
        judge_inputs = [item for judge_input_list in judge_input_lists for item in judge_input_list]

        df = construct_dataframe(inputs=debater_a_inputs)
        opponent_df = construct_dataframe(inputs=debater_b_inputs)
        judge_df = construct_dataframe(inputs=judge_inputs)

        df[PPOTrainerWrapper.OPPONENT_QUERY_COLUMN] = opponent_df[PPOTrainerWrapper.QUERY_COLUMN]
        df[PPOTrainerWrapper.JUDGE_QUERY_COLUMN] = judge_df[PPOTrainerWrapper.QUERY_COLUMN]

        return Dataset.from_pandas(df).shuffle()

    @classmethod
    def get_trainer(
        cls,
        config: TrainingConfig,
        raw_dataset: RawDataset,
        is_local: bool = False,
    ) -> PPOTrainerWrapper:
        logger = LoggerUtils.get_default_logger(__name__)

        if FLASH_ATTENTION_AVAILABLE:
            replace_attn_with_flash_attn()

        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(model_name=config.model_name, is_local=is_local, requires_value_head=True)

        peft_config = TrainUtils.get_peft_config(config)

        model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
        if FLASH_ATTENTION_AVAILABLE:
            model = upcast_layer_for_flash_attention(model, torch.bfloat16)

        ppo_config = PPOConfig(
            steps=config.training_hyperparameters.steps,  # TODO: add this HP
            learning_rate=config.training_hyperparameters.learning_rate,
            batch_size=config.training_hyperparameters.mini_batch_size,
            gradient_accumulation_steps=config.training_hyperparameters.gradient_accumulation_steps,
            ppo_epochs=config.training_hyperparameters.ppo_epochs,  # TODO: add this HP
        )

        generation_config = GenerationConfig(
            max_new_tokens=300,
            temperature=0.5,
            top_p=0.9,
            num_return_sequences=1,
            repetition_penalty=1.2,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        train_dataset = PPOTrainerWrapper.convert_dataset(
            raw_dataset=raw_dataset,
            config=config,
        )

        ppo_trainer = PPOTrainerWrapper(
            ppo_trainer=PPOTrainer(
                model=model,
                config=ppo_config,
                dataset=train_dataset,
                tokenizer=tokenizer,
            ),
            config=config,
        )

        return trainer
