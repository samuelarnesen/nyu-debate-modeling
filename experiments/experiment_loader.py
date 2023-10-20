from agents.debater import Debater, OfflineDebater
from agents.judge import BoNJudge, Judge
from agents.debate_round import DebateRound, QuestionMetadata
from agents.models.model_utils import ModelType, ModelUtils
from agents.prompt import Prompt, PromptConfig, PromptParser
from data.data import DatasetType, RawDataLoader, RawDataset, SplitType
from data.loaders.loader_utils import LoaderUtils
import utils.constants as constants

from pydantic import BaseModel
import random
import yaml

from enum import Enum
from typing import Optional


class PromptLoadingConfig(BaseModel):
    file_path: str
    prompt_name: str


class ModelConfig(BaseModel):
    model_type: str
    model_file_path: Optional[str]
    alias: str
    use_scratchpad: Optional[bool]


class ModelsConfig(BaseModel):
    debater_one: ModelConfig
    debater_two: ModelConfig
    judge: ModelConfig


class DatasetConfig(BaseModel):
    dataset_type: str
    full_dataset_file_path: Optional[str]
    train_file_path: Optional[str]
    val_file_path: Optional[str]
    test_file_path: Optional[str]
    split_type: Optional[str]


class TopicConfigType(Enum):
    HARD_CODED = 1
    FROM_DATASET = 2


class TopicConfig(BaseModel):
    topic_type: str
    topic: Optional[str]
    positions: Optional[tuple[str, str]]


class OfflineConfig(BaseModel):
    debater_one: bool
    debater_two: bool
    file_path: str


class BoNConfig(BaseModel):
    count: int


class ExperimentConfig(BaseModel):
    topic_config: TopicConfig
    word_limit: int
    batch_size: int
    num_speeches: int
    flip: Optional[bool]
    prompt_config: PromptLoadingConfig
    models: ModelsConfig
    dataset: DatasetConfig
    offline: Optional[OfflineConfig]
    best_of_n: Optional[BoNConfig]


class ExperimentLoader:
    @classmethod
    def merge_debate_rounds(cls, debate_rounds: list[DebateRound]) -> DebateRound:
        def validate() -> None:
            for debate_round in debate_rounds:
                if (
                    debate_rounds[0].first_debater.model != debate_round.first_debater.model
                    or debate_rounds[0].second_debater.model != debate_round.second_debater.model
                    or debate_rounds[0].judge.model != debate_round.judge.model
                    or debate_rounds[0].first_debater.name != debate_round.first_debater.name
                    or debate_rounds[0].second_debater.name != debate_round.second_debater.name
                    or debate_rounds[0].judge.name != debate_round.judge.name
                ):
                    raise Exception("Cannot merge rounds of across models or names")

        validate()
        first_debater_prompts = []
        second_debater_prompts = []
        judge_prompts = []
        metadata_list = []
        for debate_round in debate_rounds:
            for prompt in debate_round.first_debater.prompts:
                first_debater_prompts.append(prompt)
            for prompt in debate_round.second_debater.prompts:
                second_debater_prompts.append(prompt)
            for prompt in debate_round.judge.prompts:
                judge_prompts.append(prompt)
            for metadata in debate_round.metadata:
                metadata_list.append(metadata)

        first_debater = Debater(
            name=debate_rounds[0].first_debater.name,
            prompt=first_debater_prompts,
            model=debate_rounds[0].first_debater.model,
            num_speeches=debate_rounds[0].first_debater.num_speeches,
        )
        second_debater = Debater(
            name=debate_rounds[0].second_debater.name,
            prompt=second_debater_prompts,
            model=debate_rounds[0].second_debater.model,
            num_speeches=debate_rounds[0].first_debater.num_speeches,
        )
        judge = Judge(
            name=debate_rounds[0].judge.name,
            prompt=judge_prompts,
            model=debate_rounds[0].judge.model,
            num_speeches=debate_rounds[0].first_debater.num_speeches,
        )

        return DebateRound(first_debater=first_debater, second_debater=second_debater, judge=judge, metadata=metadata_list)

    @classmethod
    def create_dataset(cls, experiment: ExperimentConfig) -> RawDataset:
        dataset_config = experiment.dataset
        dataset_type = DatasetType[dataset_config.dataset_type.upper()]
        loader_cls = LoaderUtils.get_loader_type(dataset_type)
        return loader_cls.load(
            full_dataset_filepath=dataset_config.full_dataset_file_path,
            train_filepath=dataset_config.train_file_path,
            val_filepath=dataset_config.val_file_path,
            test_filepath=dataset_config.test_file_path,
        )

    @classmethod
    def get_split(cls, experiment: ExperimentConfig) -> SplitType:
        return SplitType[experiment.dataset.split_type.upper()] if experiment.dataset.split_type else SplitType.TRAIN

    @classmethod
    def generate_debate_rounds(
        cls, experiment_file_path: str, name: str, count: int = 1
    ) -> tuple[list[list[DebateRound]], ExperimentConfig]:
        # create experiment config
        with open(experiment_file_path) as f:
            loaded_yaml = yaml.safe_load(f)
        experiment = ExperimentConfig(**loaded_yaml[name])

        # create dataset
        dataset = ExperimentLoader.create_dataset(experiment)
        split_type = ExperimentLoader.get_split(experiment)

        # create debater models
        debater_one_model_type = ModelType[experiment.models.debater_one.model_type.upper()]
        debater_two_model_type = ModelType[experiment.models.debater_two.model_type.upper()]
        judge_model_type = ModelType[experiment.models.judge.model_type.upper()]
        debater_one_model_path = experiment.models.debater_one.model_file_path
        debater_two_model_path = experiment.models.debater_two.model_file_path
        judge_model_path = experiment.models.judge.model_file_path
        debater_one_model = ModelUtils.instantiate_model(
            model_type=debater_one_model_type,
            file_path=debater_one_model_path,
            is_debater=True,
            alias=experiment.models.debater_one.alias,
        )
        debater_two_model = (
            debater_one_model.copy(alias=experiment.models.debater_two.alias, is_debater=True)
            if debater_two_model_type == debater_one_model_type and debater_one_model_path == debater_two_model_path
            else ModelUtils.instantiate_model(
                model_type=debater_two_model_type,
                file_path=experiment.models.debater_two.model_file_path,
                is_debater=True,
                alias=experiment.models.debater_two.alias,
            )
        )
        judge_model = (
            debater_one_model.copy(alias=experiment.models.judge.alias, is_debater=False)
            if judge_model_type == debater_one_model_type and debater_one_model_path == judge_model_path
            else (
                debater_two_model.copy(is_debater=False)
                if judge_model_type == debater_two_model_type and debater_two_model_path == judge_model_path
                else ModelUtils.instantiate_model(
                    model_type=judge_model_type,
                    file_path=experiment.models.judge.model_file_path,
                    is_debater=False,
                    alias=experiment.models.judge.alias,
                )
            )
        )

        # create debate rounds
        rounds = []
        topic_config_type = TopicConfigType[experiment.topic_config.topic_type.upper()]
        for i in range(count):
            if topic_config_type == TopicConfigType.FROM_DATASET:
                example = dataset.get_example(idx=i, split=split_type)
                topic = example.question
                position = example.positions[0]
                opponent_position = example.positions[1]
                background_text = example.background_text
                correct_index = example.correct_index
            elif topic_config_type == TopicConfigType.HARD_CODED:
                topic = experiment.topic_config.topic
                position = experiment.topic_config.positions[0]
                opponent_position = experiment.topic_config.positions[1]
                background_text = constants.DEFAULT_BACKGROUND_TEXT
                correct_index = None
            else:
                raise Exception(f"Topic config type {topic_config_type} is not recognized")

            config_a = PromptConfig(
                name=constants.DEFAULT_DEBATER_A_NAME,
                opponent_name=constants.DEFAULT_DEBATER_B_NAME,
                word_limit=experiment.word_limit,
                position=position,
                opponent_position=opponent_position,
                topic=topic,
                background_text=background_text,
            )

            config_b = PromptParser.generate_opponent_config(config_a)

            prompt_a = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_a,
                name=experiment.prompt_config.prompt_name,
            )

            prompt_b = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_b,
                name=experiment.prompt_config.prompt_name,
            )

            prompt_judge = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_a,
                name=experiment.prompt_config.prompt_name,
            )

            debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=prompt_a,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=experiment.models.debater_one.use_scratchpad or False,
            )

            debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=prompt_b,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=experiment.models.debater_two.use_scratchpad or False,
            )

            judge = Judge(
                name=constants.DEFAULT_JUDGE_NAME,
                prompt=prompt_judge,
                model=judge_model,
                num_speeches=experiment.num_speeches,
            )

            debate_round = DebateRound(
                first_debater=debater_a,
                second_debater=debater_b,
                judge=judge,
                metadata=[QuestionMetadata(first_debater_correct=correct_index == 0, question_idx=i, split=split_type)],
            )

            flipped_debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=prompt_a,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=experiment.models.debater_two.use_scratchpad or False,
            )

            flipped_debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=prompt_b,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=experiment.models.debater_one.use_scratchpad or False,
            )

            flipped_round = DebateRound(
                first_debater=flipped_debater_a,
                second_debater=flipped_debater_b,
                judge=judge,
                metadata=[QuestionMetadata(first_debater_correct=correct_index == 0, question_idx=i, split=split_type)],
            )

            if experiment.offline:
                if experiment.offline.debater_one:
                    debate_round.first_debater = OfflineDebater(
                        debater=debate_round.first_debater,
                        file_path=experiment.offline.file_path,
                        first_debater_prompt=prompt_a,
                    )
                    flipped_round.second_debater = OfflineDebater(
                        debater=flipped_round.second_debater,
                        file_path=experiment.offline.file_path,
                        first_debater_prompt=prompt_a,
                    )
                if experiment.offline.debater_two:
                    debate_round.second_debater = OfflineDebater(
                        debater=debate_round.second_debater,
                        file_path=experiment.offline.file_path,
                        first_debater_prompt=prompt_a,
                    )
                    flipped_round.first_debater = OfflineDebater(
                        debater=flipped_round.first_debater,
                        file_path=experiment.offline.file_path,
                        first_debater_prompt=prompt_a,
                    )

            if experiment.best_of_n:
                if experiment.num_speeches > 1:
                    raise Exception("For now, there can only be 1 speech when doing BoN")
                debate_round.judge = BoNJudge(judge=debate_round.judge, n=experiment.best_of_n.count)
                flipped_round.judge = BoNJudge(judge=flipped_round.judge, n=experiment.best_of_n.count)

            rounds.append(debate_round)
            if experiment.flip:
                rounds.append(flipped_round)

        if len(rounds) <= 1 or experiment.best_of_n:
            return rounds, experiment

        # batches the debate rounds for efficient generation
        batched_rounds = []
        current_normal_batch = []
        current_flipped_batch = []
        for i, debate_round in enumerate(rounds):
            if i % 2 == 0:
                current_normal_batch.append(debate_round)
                if len(current_normal_batch) == experiment.batch_size:
                    batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_normal_batch))
                    current_normal_batch = []
            else:
                current_flipped_batch.append(debate_round)
                if len(current_flipped_batch) == experiment.batch_size:
                    batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_flipped_batch))
                    current_flipped_batch = []

        if current_normal_batch:
            batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_normal_batch))
        if current_flipped_batch:
            batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_flipped_batch))

        return batched_rounds, experiment
