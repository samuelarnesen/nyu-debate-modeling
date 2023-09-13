from agents.agent import Debater, Judge
from agents.debate_round import DebateRound
from agents.models.model_utils import ModelType, ModelUtils
from agents.prompt import Prompt, PromptConfig, PromptParser
from data.data import DataLoader, Dataset, DatasetType
from data.loaders.loader_utils import LoaderUtils

from pydantic import BaseModel
import yaml

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


class ExperimentConfig(BaseModel):
    topic: str
    word_limit: int
    positions: tuple[str, str]
    prompt_config: PromptLoadingConfig
    models: ModelsConfig
    dataset: DatasetConfig


DEFAULT_DEBATER_ONE_NAME = "Debater_One"
DEFAULT_DEBATER_TWO_NAME = "Debater_Two"
DEFAULT_JUDGE_NAME = "Judge"


class ExperimentLoader:
    @classmethod
    def generate_debate_round(cls, experiment_file_path: str, name: str) -> DebateRound:
        with open(experiment_file_path) as f:
            loaded_yaml = yaml.safe_load(f)

        experiment = ExperimentConfig(**loaded_yaml[name])

        config_one = PromptConfig(
            name=DEFAULT_DEBATER_ONE_NAME,  # TODO: change defaults?
            opponent_name=DEFAULT_DEBATER_TWO_NAME,
            judge_name=DEFAULT_JUDGE_NAME,
            word_limit=experiment.word_limit,
            position=experiment.positions[0],
            opponent_position=experiment.positions[1],
            topic=experiment.topic,
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
            name=DEFAULT_DEBATER_ONE_NAME,
            prompt=prompt_one,
            model=ModelUtils.instantiate_model(model_type=ModelType[experiment.models.debater_one.model_type.upper()]),
        )
        debater_two = Debater(
            name=DEFAULT_DEBATER_TWO_NAME,
            prompt=prompt_two,
            model=ModelUtils.instantiate_model(model_type=ModelType[experiment.models.debater_two.model_type.upper()]),
        )
        judge = Judge(
            name=DEFAULT_JUDGE_NAME,
            prompt=prompt_judge,
            model=ModelUtils.instantiate_model(model_type=ModelType[experiment.models.judge.model_type.upper()]),
        )

        dataset_config = experiment.dataset
        dataset_type = DatasetType[dataset_config.dataset_type.upper()]
        loader_cls = LoaderUtils.get_loader_type(dataset_type)
        dataset = loader_cls.load(
            full_dataset_filepath=dataset_config.full_dataset_file_path,
            train_filepath=dataset_config.train_file_path,
            val_filepath=dataset_config.val_file_path,
            test_filepath=dataset_config.test_file_path,
        )

        return DebateRound(first_debater=debater_one, second_debater=debater_two, judge=judge, dataset=dataset)
