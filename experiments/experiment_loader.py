from agents.agent import Debater, Judge
from agents.debate_round import DebateRound
from agents.models.model_utils import ModelType, ModelUtils
from agents.prompt import Prompt, PromptConfig, PromptParser
from data.data import DatasetType, RawDataLoader, RawDataset, SplitType
from data.loaders.loader_utils import LoaderUtils
import utils.constants as constants

from pydantic import BaseModel
import yaml

from enum import Enum
from typing import Optional


class PromptLoadingConfig(BaseModel):
    file_path: str
    prompt_name: str


class ModelConfig(BaseModel):
    model_type: str
    model_file_path: Optional[str]


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


class TopicConfigType(Enum):
    HARD_CODED = 1
    FROM_DATASET = 2


class TopicConfig(BaseModel):
    topic_type: str
    topic: Optional[str]
    positions: Optional[tuple[str, str]]


class ExperimentConfig(BaseModel):
    topic_config: TopicConfig
    word_limit: int
    prompt_config: PromptLoadingConfig
    models: ModelsConfig
    dataset: DatasetConfig


class ExperimentLoader:
    @classmethod
    def generate_debate_rounds(cls, experiment_file_path: str, name: str, count: int = 1) -> list[DebateRound]:
        # create experiment config
        with open(experiment_file_path) as f:
            loaded_yaml = yaml.safe_load(f)
        experiment = ExperimentConfig(**loaded_yaml[name])

        # create dataset
        dataset_config = experiment.dataset
        dataset_type = DatasetType[dataset_config.dataset_type.upper()]
        loader_cls = LoaderUtils.get_loader_type(dataset_type)
        dataset = loader_cls.load(
            full_dataset_filepath=dataset_config.full_dataset_file_path,
            train_filepath=dataset_config.train_file_path,
            val_filepath=dataset_config.val_file_path,
            test_filepath=dataset_config.test_file_path,
        )

        # create debater models
        debater_one_model_type = ModelType[experiment.models.debater_one.model_type.upper()]
        debater_two_model_type = ModelType[experiment.models.debater_two.model_type.upper()]
        judge_model_type = ModelType[experiment.models.judge.model_type.upper()]
        debater_one_model_path = experiment.models.debater_one.model_file_path
        debater_two_model_path = experiment.models.debater_two.model_file_path
        judge_model_path = experiment.models.judge.model_file_path
        debater_one_model = ModelUtils.instantiate_model(model_type=debater_one_model_type, file_path=debater_one_model_path)
        debater_two_model = (
            debater_one_model
            if debater_two_model_type == debater_one_model_type and debater_one_model_path == debater_two_model_path
            else ModelUtils.instantiate_model(
                model_type=debater_two_model_type, file_path=experiment.models.debater_two.model_file_path
            )
        )
        judge_model = (
            debater_one_model
            if judge_model_type == debater_one_model_type and debater_one_model_path == judge_model_path
            else (
                debater_two_model
                if judge_model_type == debater_two_model_type and debater_two_model_path == judge_model_path
                else ModelUtils.instantiate_model(
                    model_type=judge_model_type, file_path=experiment.models.judge.model_file_path, is_debater=False
                )
            )
        )

        # create debate rounds
        rounds = []
        topic_config_type = TopicConfigType[experiment.topic_config.topic_type.upper()]
        for i in range(count):
            if topic_config_type == TopicConfigType.FROM_DATASET:
                example = dataset.get_example(idx=i)
                topic = example.question
                position = example.positions[0]
                opponent_position = example.positions[1]
                background_text = example.background_text
            elif topic_config_type == TopicConfigType.HARD_CODED:
                topic = experiment.topic_config.topic
                position = experiment.topic_config.positions[0]
                opponent_position = experiment.topic_config.positions[1]
                background_text = constants.DEFAULT_BACKGROUND_TEXT
            else:
                raise Exception(f"Topic config type {topic_config_type} is not recognized")

            config_one = PromptConfig(
                name=constants.DEFAULT_DEBATER_ONE_NAME,
                opponent_name=constants.DEFAULT_DEBATER_TWO_NAME,
                word_limit=experiment.word_limit,
                position=position,
                opponent_position=opponent_position,
                topic=topic,
                background_text=background_text,
            )

            config_two = PromptParser.generate_opponent_config(config_one)

            prompt_one = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_one,
                name=experiment.prompt_config.prompt_name,
            )

            prompt_two = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_two,
                name=experiment.prompt_config.prompt_name,
            )

            prompt_judge = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_one,
                name=experiment.prompt_config.prompt_name,
            )

            debater_one = Debater(
                name=constants.DEFAULT_DEBATER_ONE_NAME,
                prompt=prompt_one,
                model=debater_one_model,
            )
            debater_two = Debater(
                name=constants.DEFAULT_DEBATER_TWO_NAME,
                prompt=prompt_two,
                model=debater_two_model,
            )

            judge = Judge(
                name=constants.DEFAULT_JUDGE_NAME,
                prompt=prompt_judge,
                model=judge_model,
            )

            debate_round = DebateRound(
                first_debater=debater_one, second_debater=debater_two, judge=judge, idx=i, split=SplitType.TRAIN
            )
            rounds.append(debate_round)

        return rounds
