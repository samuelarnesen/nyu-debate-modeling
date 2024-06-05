from __future__ import annotations

from data import DataRow, RawDataset, SplitType
from debate import Debater, DebateRound, Judge, QuestionMetadata
from models import ArbitraryAttributeModel, Llama3Model, OpenAIModel, OfflineModel, Model, RandomModel, SpeechStructure
from prompts import Prompt, PromptConfig, PromptLoadingConfig, PromptParser
from train.impl import LlamaModelWithGradientCheckpointing, VerbosePPOTrainer
from train.row_converter import RowConverter
from train.train_utils import TrainUtils, TrainingConfig, TrainingTarget
from utils import LoggingCallback, logger_utils, string_utils, timer
import utils.constants as constants

from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, GenerationConfig, LlamaModel
from trl import PPOConfig, PPOTrainer
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch

from typing import Optional
import math
import sys
import traceback

try:
    import bitsandbytes as bnb
except:
    print("Unable to import bitsandbytes")

try:
    from utils.flash_attn_utils import (
        replace_attn_with_flash_attn,
        upcast_layer_for_flash_attention,
    )

    FLASH_ATTENTION_AVAILABLE = True
except ImportError as e:
    print("Running without flash attention")
    FLASH_ATTENTION_AVAILABLE = False


def print_available_memory(logger):
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated(0)
    logger.warn(f"Total GPU Memory: {total_memory / (1024 ** 3):.2f} GB")
    logger.warn(f"Free GPU Memory: {free_memory / (1024 ** 3):.2f} GB")


# Extended monkeypatch script to fix a bug in PPOTrainer
def logprobs_from_logits(logits, labels, gather=True):
    logp = F.log_softmax(logits, dim=-1)
    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


class PPOStats:
    def __init__(self):
        self.correct_scores = []
        self.incorrect_scores = []

    def __get_scores(self, scores: list[float], window: int = -1):
        if window <= 0:
            return sum(scores) / len(scores) if scores else -1
        elif window > len(scores):
            return sum(scores) / len(scores) if scores else -1
        else:
            return sum(scores[-window:]) / window if window else -1

    def get_correct_scores(self, window: int = -1):
        return self.__get_scores(self.correct_scores, window)

    def get_incorrect_scores(self, window: int = -1):
        return self.__get_scores(self.incorrect_scores, window)

    def get_scores(self, window: int = -1):
        return (1 / 2) * (self.get_correct_scores(window=window) + self.get_incorrect_scores(window=window))

LlamaModel.forward = LlamaModelWithGradientCheckpointing.forward

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
    BATCH_SIZE = 1
    DEFAULT_DEBATER_ALIAS = "mixtral-debater"
    DEFAULT_JUDGE_ALIAS = "openai-judge"

    def __init__(
        self, ppo_trainer: PPOTrainer, config: TrainingConfig, dataset: RawDataset, ref_model: AutoModelForCausalLM
    ):
        """
        Class for training a model using Proximate Policy Optimization. In order to keep the
        interface the same as the one used by the DPOTrainer and SFTTrainer, we construct a
        wrapper here so that one can just call the train() function to run the training loop.

        Params:
            ppo_trainer: a huggingface-object that handles the actual ppo algorithm
            config: configuration specifying the prompt setup and hyperparameters for the training run.
            dataset: the dataset to train on
            ref_model: a model to be debated against during data generation
        """

        self.ppo_trainer = ppo_trainer

        self.reward_model = OpenAIModel(
            alias=PPOTrainerWrapper.DEFAULT_JUDGE_ALIAS, is_debater=False, endpoint="ft:gpt-4-0613:nyu-arg::90NW3Tbx"
        )  # make configurable

        """
        self.reward_model = Llama3Model(
            alias=PPOTrainerWrapper.DEFAULT_JUDGE_ALIAS, file_path="/vast/spa9663/models/base_models/llama3-8b-262k", is_debater=False,
        )  # make configurable
        """

        """
        self.reward_model = ArbitraryAttributeModel(
            alias=PPOTrainerWrapper.DEFAULT_JUDGE_ALIAS, is_debater=False, feature="quote"
        )
        """
        self.ref_model = ref_model
        self.config = config
        self.dataset = dataset

        self.stats_to_log = [
            "objective/kl",
            "objective/kl_dist",
            "objective/entropy",
            "ppo/mean_scores",
            "ppo/mean_non_score_reward",
            "ppo/policy/advantages_mean",
            "ppo/val/error",
            "ppo/loss/value",
            "ppo/policy/ratio",
            "ppo/val/var_explained",
            "ppo/val/clipfrac",
            "env/reward_dist",
            "ppo/scores_dist",
            "ppo/std_scores",
        ]

        self.logger = logger_utils.get_default_logger(__name__)

        self.alternate_speech = None # TODO: delete this

    @timer("generate batch samples")
    def get_batch_samples(self, start_idx: int, ppo_stats: PPOStats) -> tuple[list[str], list[str], list[float]]:
        samples = []
        for i in range(
            self.config.training_hyperparameters.per_device_train_batch_size
        ):  # TODO: change this when we do multi-turn
            new_samples = self.generate_one_round_samples(idx=start_idx + i, ppo_stats=ppo_stats)
            samples.extend(new_samples)

        query_texts = [sample[0] for sample in samples]
        response_texts = [sample[1] for sample in samples]
        score_texts = [sample[2] for sample in samples]
        return query_texts, response_texts, score_texts

    def train(self, save_frequency: int = 10):
        ppo_stats = PPOStats()
        for i in range(self.config.training_hyperparameters.steps):
            self.train_single_batch(
                start_idx=(i * self.config.training_hyperparameters.per_device_train_batch_size), ppo_stats=ppo_stats
            )

            if i > 0 and i % save_frequency == 0:
                self.save_model(checkpoint=i)

    @timer("train one batch")
    def train_single_batch(self, start_idx: int, ppo_stats: PPOStats):
        with torch.no_grad():
            query_texts, response_texts, score_texts = self.get_batch_samples(start_idx=start_idx, ppo_stats=ppo_stats)

        queries = [torch.tensor(qt).to("cuda") for qt in query_texts]
        responses = [torch.tensor(rt).to("cuda") for rt in response_texts]
        scores = [x.to("cuda") for x in torch.FloatTensor(score_texts)]

        window = self.config.training_hyperparameters.per_device_train_batch_size // 2
        batch_idx = start_idx // self.config.training_hyperparameters.per_device_train_batch_size
        overall = ppo_stats.get_scores(window=window)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            stats = self.ppo_trainer.step(
                queries=queries,
                responses=responses,
                scores=scores,
            )

        self.ppo_trainer.log_stats(stats=stats, batch={"query": queries, "response": responses}, rewards=scores)
        try:
            cleaned_stats = {k: v for k, v in filter(lambda x: x[0] in self.stats_to_log, stats.items())}
            cleaned_stats["batch_idx"] = batch_idx
            self.logger.warn(cleaned_stats)

            first_response = self.ppo_trainer.tokenizer.decode(responses[0])
            self.logger.warn(first_response + "\n\n")

            """
            for j in [0]: # change depending on number of samples you want printed
                logprob_length = len(stats["objective/logprobs"][j])
                query_length = len(queries[j])
                max_query_length = max([len(queries[z]) for z in range(len(queries))])
                response_length = len(responses[j])
                combined = query_length + response_length
                max_combined = max_query_length + response_length

                offset = queries[j].shape[0]
                response_length = responses[j].shape[0]
                log_prob_diffs = []
                for i, (active, ref) in enumerate(
                    zip(
                        stats["objective/logprobs"][j][offset : (offset + response_length)],
                        stats["objective/ref_logprobs"][j][offset : (offset + response_length)],
                    )
                ):
                    log_prob_diffs.append((i, active.item() - ref.item()))
                log_prob_diffs = sorted(log_prob_diffs, key=lambda x: x[1])

                length = len(log_prob_diffs)
                decoded_response = self.ppo_trainer.tokenizer.decode(responses[j])
                for idx, (i, diff) in enumerate(log_prob_diffs[0:10]):
                    token = self.ppo_trainer.tokenizer.decode([responses[j][i]])
                    self.logger.warn(f"{j}:\t{idx}. {token} ({diff}) ({i}) [{length}] [{query_length}]")
            """
        except Exception as e:
            self.logger.warn("Exception when trying to print stats")
            self.logger.error(e)
            self.logger.error(traceback.format_exc())

    def generate_one_round_samples(self, idx: int, ppo_stats: PPOStats) -> list[tuple[list[int], list[int], float]]:
        def convert_reward(summary, speech):
            correct_baseline = 0.67
            incorrect_baseline = 0.33
            if speech.speaker == constants.DEFAULT_DEBATER_A_NAME:
                baseline = correct_baseline if summary[0].metadata.first_debater_correct else incorrect_baseline
                return summary[0].first_debater_win_prob - baseline
            else:
                baseline = incorrect_baseline if summary[0].metadata.first_debater_correct else correct_baseline
                return summary[0].second_debater_win_prob - baseline

        def add_to_ppo_stats(summary, speech):
            if speech.speaker == constants.DEFAULT_DEBATER_A_NAME:
                if summary[0].metadata.first_debater_correct:
                    ppo_stats.correct_scores.append(summary[0].first_debater_win_prob)
                else:
                    ppo_stats.incorrect_scores.append(summary[0].first_debater_win_prob)
            """ TODO: fix this
            elif speech.speaker == constants.DEFAULT_DEBATER_B_NAME and idx % 2 == 1:
                if summary[0].metadata.first_debater_correct:
                    ppo_stats.incorrect_scores.append(summary[0].second_debater_win_prob)
                else:
                    ppo_stats.correct_scores.append(summary[0].second_debater_win_prob)
            """

        example = self.dataset.get_example(idx=idx, split=SplitType.TRAIN)

        llm_class = TrainUtils.get_llm_class(self.config)
        internal_model = llm_class(
            alias=PPOTrainerWrapper.DEFAULT_DEBATER_ALIAS + "_active",
            file_path=None,
            is_debater=True,
        )
        internal_model.model = self.ppo_trainer.model
        internal_model.tokenizer = self.ppo_trainer.tokenizer
        internal_model.generation_config = internal_model.create_default_generation_config(
            is_debater=True, do_sample=True, add_penalties=False
        )
        internal_model.instantiated_model = True
        internal_model.is_debater = True

        reference_model = llm_class(
            alias=PPOTrainerWrapper.DEFAULT_DEBATER_ALIAS + "_reference",
            file_path=None,
            is_debater=True,
        )

        reference_model.model = self.ref_model
        reference_model.tokenizer = self.ppo_trainer.tokenizer
        reference_model.generation_config = reference_model.create_default_generation_config(
            is_debater=True, do_sample=True, add_penalties=False
        )
        reference_model.instantiated_model = True
        reference_model.is_debater = True


        # TODO: delete this
        if self.alternate_speech:
            reference_model = OfflineModel(
                alias=PPOTrainerWrapper.DEFAULT_DEBATER_ALIAS + "_reference",
                speeches=[self.alternate_speech],
                is_debater=True,
            )

        topic = example.question
        position = example.positions[0]
        opponent_position = example.positions[1]
        background_text = example.background_text
        title = example.story_title
        correct_index = example.correct_index
        speeches = example.speeches

        if idx < 8:
            self.logger.warn(
                f"Question is {topic} from the story {title}. Options are A: {position} and B: {opponent_position}. Correct index is {correct_index}"
            )

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
        debater_a = Debater(
            name=constants.DEFAULT_DEBATER_A_NAME,
            prompt=prompt_a,
            model=internal_model,  # if idx % 2 == 0 else reference_model,
            num_speeches=num_speeches,
            speech_format=self.config.speech_structure[0].debater_format.get_speech_format(
                name=constants.DEFAULT_DEBATER_A_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
            ),
            quotes_require_validation=False,
        )

        debater_b = Debater(
            name=constants.DEFAULT_DEBATER_B_NAME,
            prompt=prompt_b,
            model=reference_model,  # if idx % 2 == 0 else internal_model,
            num_speeches=num_speeches,
            speech_format=self.config.speech_structure[0].debater_format.get_speech_format(
                name=constants.DEFAULT_DEBATER_B_NAME,
                num_speeches=num_speeches,
                use_scratchpad=False,
            ),
            quotes_require_validation=False,
        )

        judge = Judge(
            name=constants.DEFAULT_JUDGE_NAME,
            prompt=prompt_judge,
            model=self.reward_model,
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

        samples = []
        expected_speaker = constants.DEFAULT_DEBATER_A_NAME  # if idx % 2 == 0 else constants.DEFAULT_DEBATER_B_NAME
        for speech in filter(
            lambda x: x.speaker == expected_speaker,
            judge.transcripts[0].speeches,
        ):
            samples.append(
                (speech.supplemental.prompt_tokens, speech.supplemental.response_tokens, convert_reward(summary, speech))
            )
            add_to_ppo_stats(summary, speech)

        # TODO: delete this
        if not self.alternate_speech:
            for speech in filter(
                lambda x: x.speaker == constants.DEFAULT_DEBATER_B_NAME,
                judge.transcripts[0].speeches,
            ):
                self.alternate_speech = speech.content
                self.logger.warn("OPPONENT SPEECH\n")
                self.logger.warn(self.alternate_speech)
                break

        return samples

    def save_model(self, checkpoint: Optional[int] = None):
        """Saves the model to the specified location. If a checkpoint number is available, we will save with that
        checkpoint as prefix"""
        location = self.config.logging_and_saving_config.output_dir
        if checkpoint:
            location += f"/checkpoint-{checkpoint}"

        self.ppo_trainer.save_pretrained(location)


    @classmethod
    def get_trainer(
        cls,
        config: TrainingConfig,
        raw_datasets: Optional[list[RawDataset]] = None,
        is_local: bool = False,
        is_test: bool = False,
    ) -> PPOTrainerWrapper:
        """
        Generates a PPOTrainerWrapper object that should have the same interface as trl's
        SFTTrainer and DPOTrainer objects.

        Params:
            config: configuration specifying the prompt setup and hyperparameters for the training run.
            raw_dataset: dataset to use for training. Autogenerated using the config if it is missing
            is_local: whether this is being run on a cpu

        Returns:
            ppo_trainer: One can call ppo_trainer.train() to then run the training loop.
        """

        if not raw_datasets:
            raw_datasets = TrainUtils.create_datasets(config=config)
        raw_dataset = raw_datasets[0]  # we don't support multiple datasets at the moment

        ppo_config = PPOConfig(
            steps=config.training_hyperparameters.steps,
            learning_rate=config.training_hyperparameters.learning_rate,
            batch_size=config.training_hyperparameters.per_device_train_batch_size,
            gradient_accumulation_steps=config.training_hyperparameters.gradient_accumulation_steps,
            mini_batch_size=1,
            ppo_epochs=1,
            optimize_device_cache=True,
            #kl_penalty='abs',
            init_kl_coef=0.10,
        )

        tokenizer = TrainUtils.get_tokenizer(config=config)
        model = TrainUtils.load_model(config=config, is_local=is_local, requires_value_head=True)
        reference_model = TrainUtils.load_model(config=config, is_local=is_local, requires_value_head=False)

        model.gradient_checkpointing_enable()
        model.pretrained_model.gradient_checkpointing_enable()
        model.pretrained_model.enable_input_require_grads()
        model.gradient_checkpointing_enable = model.pretrained_model.gradient_checkpointing_enable

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.training_hyperparameters.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        ppo_trainer = PPOTrainerWrapper(
            ppo_trainer=VerbosePPOTrainer(model=model, config=ppo_config, tokenizer=tokenizer, optimizer=optimizer, lr_scheduler=lr_scheduler),
            config=config,
            dataset=raw_dataset,
            ref_model=reference_model,
        )

        return ppo_trainer
