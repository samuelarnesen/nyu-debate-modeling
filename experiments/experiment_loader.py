from agents.debater import BoNDebater, Debater, HumanDebater, OfflineDebater
from agents.judge import BoNJudge, Judge
from agents.debate_round import DebateRound, QuestionMetadata
from agents.model import Model
from agents.models.model_utils import ModelType, ModelUtils
from agents.prompt import DynamicPromptParser, Prompt, PromptConfig, PromptParser
from data.data import DatasetType, RawDataLoader, RawDataset, SplitType
from data.loaders.loader_utils import LoaderUtils
from utils.logger_utils import LoggerUtils
import utils.constants as constants

from pydantic import BaseModel
import random
import yaml

from enum import Enum
from typing import Optional
import itertools


class PromptLoadingConfig(BaseModel):
    file_path: str
    default_prompt_name: str
    dynamic_prompts_file_path: Optional[str]
    dynamic_prompt_name: Optional[str]


class AgentConfig(BaseModel):
    model_type: str
    model_file_path: Optional[str]
    alias: str
    use_scratchpad: Optional[bool]
    override_prompt: Optional[str]


class AgentsConfig(BaseModel):
    debaters: list[AgentConfig]
    judge: AgentConfig


class DatasetConfig(BaseModel):
    dataset_type: str
    full_dataset_file_path: Optional[str]
    train_file_path: Optional[str]
    val_file_path: Optional[str]
    test_file_path: Optional[str]
    annotations_file_path: Optional[str]
    split_type: Optional[str]


class TopicConfigType(Enum):
    HARD_CODED = 1
    FROM_DATASET = 2


class TopicConfig(BaseModel):
    topic_type: str
    topic: Optional[str]
    positions: Optional[tuple[str, str]]


class OfflineConfig(BaseModel):
    debaters: list[bool]
    file_path: str


class HumanConfig(BaseModel):
    debaters: list[str]


class BoNConfig(BaseModel):
    count: int
    prompts: Optional[list[str]]


class ExperimentConfig(BaseModel):
    topic_config: TopicConfig
    word_limit: int
    batch_size: int
    num_speeches: int
    flip: Optional[bool]
    prompt_config: PromptLoadingConfig
    agents: AgentsConfig
    dataset: DatasetConfig
    offline: Optional[OfflineConfig]
    best_of_n: Optional[BoNConfig]
    human: Optional[HumanConfig]


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
            annotations_file_path=dataset_config.annotations_file_path,
        )

    @classmethod
    def get_split(cls, experiment: ExperimentConfig) -> SplitType:
        return SplitType[experiment.dataset.split_type.upper()] if experiment.dataset.split_type else SplitType.TRAIN

    @classmethod
    def create_debate_rounds_for_combination(
        cls,
        experiment: ExperimentConfig,
        dataset: RawDataset,
        split_type: SplitType,
        debater_idxs: tuple[int, int],
        count: int,
        model_cache: Optional[dict[str, Model]] = None,
    ) -> list[DebateRound]:
        # create logger
        logger = LoggerUtils.get_default_logger(__name__)

        # create debater models
        debater_one_model_type = ModelType[experiment.agents.debaters[debater_idxs[0]].model_type.upper()]
        debater_two_model_type = ModelType[experiment.agents.debaters[debater_idxs[1]].model_type.upper()]
        judge_model_type = ModelType[experiment.agents.judge.model_type.upper()]
        debater_one_model_path = experiment.agents.debaters[debater_idxs[0]].model_file_path
        debater_two_model_path = experiment.agents.debaters[debater_idxs[1]].model_file_path
        judge_model_path = experiment.agents.judge.model_file_path
        logger.debug(f"Instantiating a {debater_one_model_type} from {debater_one_model_path}")

        if not model_cache:
            model_cache = {}

        debater_one_model = (
            ModelUtils.instantiate_model(
                model_type=debater_one_model_type,
                file_path=debater_one_model_path,
                is_debater=True,
                alias=experiment.agents.debaters[debater_idxs[0]].alias,
            )
            if f"{debater_one_model_type}_{debater_one_model_path}" not in model_cache
            else model_cache[f"{debater_one_model_type}_{debater_one_model_path}"].copy(
                alias=experiment.agents.debaters[debater_idxs[0]].alias, is_debater=True
            )
        )
        model_cache[f"{debater_one_model_type}_{debater_one_model_path}"] = debater_one_model

        debater_two_model = (
            ModelUtils.instantiate_model(
                model_type=debater_two_model_type,
                file_path=debater_two_model_path,
                is_debater=True,
                alias=experiment.agents.debaters[debater_idxs[1]].alias,
            )
            if f"{debater_two_model_type}_{debater_two_model_path}" not in model_cache
            else model_cache[f"{debater_two_model_type}_{debater_two_model_path}"].copy(
                alias=experiment.agents.debaters[debater_idxs[1]].alias, is_debater=True
            )
        )
        model_cache[f"{debater_two_model_type}_{debater_two_model_path}"] = debater_two_model

        judge_model = (
            ModelUtils.instantiate_model(
                model_type=judge_model_type,
                file_path=judge_model_path,
                is_debater=False,
                alias=experiment.agents.judge.alias,
            )
            if f"{judge_model_type}_{judge_model_path}" not in model_cache
            else model_cache[f"{judge_model_type}_{judge_model_path}"].copy(
                alias=experiment.agents.judge.alias, is_debater=False
            )
        )
        model_cache[f"{judge_model_type}_{judge_model_path}"] = judge_model

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
                speeches = example.speeches
            elif topic_config_type == TopicConfigType.HARD_CODED:
                topic = experiment.topic_config.topic
                position = experiment.topic_config.positions[0]
                opponent_position = experiment.topic_config.positions[1]
                background_text = constants.DEFAULT_BACKGROUND_TEXT
                correct_index = None
                speeches = []
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
                name=experiment.agents.debaters[debater_idxs[0]].override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            flipped_prompt_a = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_a,
                name=experiment.agents.debaters[debater_idxs[1]].override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            prompt_b = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_b,
                name=experiment.agents.debaters[debater_idxs[1]].override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            flipped_prompt_b = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_a,
                name=experiment.agents.debaters[debater_idxs[0]].override_prompt
                or experiment.prompt_config.default_prompt_name,
            )

            prompt_judge = PromptParser.parse(
                prompts_file_path=experiment.prompt_config.file_path,
                prompt_config=config_a,
                name=experiment.prompt_config.default_prompt_name,
            )

            if experiment.prompt_config.dynamic_prompts_file_path and experiment.prompt_config.dynamic_prompt_name:
                prompt_a = DynamicPromptParser.convert_to_dynamic_prompt(
                    dynamic_prompt_file_path=experiment.prompt_config.dynamic_prompts_file_path,
                    prompt=prompt_a,
                    prompt_config=config_a,
                    dataset=dataset,
                    index=i,
                    split=split_type,
                    row=example,
                    dynamic_prompt_name=experiment.prompt_config.dynamic_prompt_name,
                )

                prompt_b = DynamicPromptParser.convert_to_dynamic_prompt(
                    dynamic_prompt_file_path=experiment.prompt_config.dynamic_prompts_file_path,
                    prompt=prompt_b,
                    prompt_config=config_b,
                    dataset=dataset,
                    index=i,
                    split=split_type,
                    row=example,
                    dynamic_prompt_name=experiment.prompt_config.dynamic_prompt_name,
                )

            debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=prompt_a,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=False,  # TODO: change later
            )

            debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=prompt_b,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=False,  # TODO: change later
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
                metadata=[
                    QuestionMetadata(
                        first_debater_correct=correct_index == 0,
                        question_idx=i,
                        background_text=background_text,
                        split=split_type,
                    )
                ],
            )

            flipped_debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=flipped_prompt_a,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=False,  # change later
            )

            flipped_debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=flipped_prompt_b,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                use_scratchpad=False,  # change later
            )

            flipped_round = DebateRound(
                first_debater=flipped_debater_a,
                second_debater=flipped_debater_b,
                judge=judge,
                metadata=[
                    QuestionMetadata(
                        first_debater_correct=correct_index == 0,
                        question_idx=i,
                        background_text=background_text,
                        split=split_type,
                    )
                ],
            )

            if experiment.best_of_n:
                if experiment.num_speeches > 1:
                    raise Exception("For now, there can only be 1 speech when doing BoN")

                debater_a_prompts = []
                debater_b_prompts = []
                logger.info(
                    f"Using {(len(experiment.best_of_n.prompts) if experiment.best_of_n.prompts else 0)} new prompts"
                )
                for prompt_name in experiment.best_of_n.prompts or [experiment.prompt_config.default_prompt_name]:
                    for prompt_list, config in zip([debater_a_prompts, debater_b_prompts], [config_a, config_b]):
                        prompt_list.append(
                            PromptParser.parse(
                                prompts_file_path=experiment.prompt_config.file_path,
                                prompt_config=config,
                                name=prompt_name,
                            )
                        )

                debate_round.set_judge(BoNJudge(judge=debate_round.judge, n=experiment.best_of_n.count, debater_a=True))
                flipped_round.set_judge(BoNJudge(judge=flipped_round.judge, n=experiment.best_of_n.count, debater_a=False))
                debate_round.set_first_debater(
                    BoNDebater(
                        debater=debate_round.first_debater,
                        n=experiment.best_of_n.count,
                        prompts=debater_a_prompts,
                        evaluated=True,
                    )
                )
                flipped_round.set_first_debater(
                    BoNDebater(debater=flipped_round.first_debater, n=experiment.best_of_n.count, evaluated=False)
                )
                debate_round.set_second_debater(
                    BoNDebater(debater=debate_round.second_debater, n=experiment.best_of_n.count, evaluated=False)
                )
                flipped_round.set_second_debater(
                    BoNDebater(
                        debater=flipped_round.second_debater,
                        n=experiment.best_of_n.count,
                        prompts=debater_b_prompts,
                        evaluated=True,
                    )
                )

            if experiment.offline:
                if experiment.offline.debaters[0]:
                    debate_round.set_first_debater(
                        OfflineDebater(
                            debater=debate_round.first_debater,
                            file_path=experiment.offline.file_path,
                            first_debater_prompt=prompt_a,
                        )
                    )
                    flipped_round.set_second_debater(
                        OfflineDebater(
                            debater=flipped_round.second_debater,
                            file_path=experiment.offline.file_path,
                            first_debater_prompt=prompt_a,
                        )
                    )
                if experiment.offline.debaters[1]:
                    debate_round.set_second_debater(
                        OfflineDebater(
                            debater=debate_round.second_debater,
                            file_path=experiment.offline.file_path,
                            first_debater_prompt=prompt_a,
                        )
                    )
                    flipped_round.set_first_debater(
                        OfflineDebater(
                            debater=flipped_round.first_debater,
                            file_path=experiment.offline.file_path,
                            first_debater_prompt=prompt_a,
                        )
                    )
            if experiment.human:
                if debate_round.first_debater.model.alias in experiment.human.debaters:
                    debate_round.set_first_debater(HumanDebater(debater=debate_round.first_debater, speeches=speeches))
                if debate_round.second_debater.model.alias in experiment.human.debaters:
                    debate_round.set_second_debater(HumanDebater(debater=debate_round.second_debater, speeches=speeches))
                if flipped_round.first_debater.model.alias in experiment.human.debaters:
                    flipped_round.set_first_debater(HumanDebater(debater=flipped_round.first_debater, speeches=speeches))
                if flipped_round.second_debater.model.alias in experiment.human.debaters:
                    flipped_round.set_second_debater(HumanDebater(debater=flipped_round.second_debater, speeches=speeches))

            rounds.append(debate_round)
            if experiment.flip:
                rounds.append(flipped_round)

        if len(rounds) <= 1 or experiment.best_of_n:
            return rounds, model_cache

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

        return batched_rounds, model_cache

    @classmethod
    def get_debater_combinations(cls, experiment: ExperimentConfig):
        all_idxs = [i for i in range(len(experiment.agents.debaters))]
        if len(all_idxs) < 2:
            raise Exception("At least 2 debaters must be defined")

        return [elem for elem in itertools.combinations(all_idxs, r=2)]

    @classmethod
    def generate_debate_rounds(
        cls, experiment_file_path: str, name: str, count: int = 1
    ) -> tuple[list[DebateRound], ExperimentConfig]:
        # create experiment config
        with open(experiment_file_path) as f:
            loaded_yaml = yaml.safe_load(f)
        experiment = ExperimentConfig(**loaded_yaml[name])

        # create dataset
        dataset = ExperimentLoader.create_dataset(experiment)
        split_type = ExperimentLoader.get_split(experiment)

        all_rounds = []
        model_cache = {}
        for combination in ExperimentLoader.get_debater_combinations(experiment=experiment):
            rounds, model_cache = ExperimentLoader.create_debate_rounds_for_combination(
                experiment=experiment,
                dataset=dataset,
                split_type=split_type,
                debater_idxs=combination,
                count=count,
                model_cache=model_cache,
            )
            all_rounds.extend(rounds)

        return all_rounds, experiment
