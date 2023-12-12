from dotenv import load_dotenv
from pydantic import BaseModel

from enum import Enum
import argparse
import logging
import os
import sys


class DebateRoundScriptConfig(BaseModel):
    experiment_name: str
    experiment_file_path: str
    save_path_base: str


class ModelRunScriptConfig(BaseModel):
    config_filepath: str
    full_dataset_filepath: str
    config_name: str


class TrainType(Enum):
    SFT = 0
    DPO = 1
    PPO = 2
    PRETRAIN = 3


class ScriptUtils:
    @classmethod
    def setup_script(cls):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        load_dotenv()

    @classmethod
    def get_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--local", action="store_true", default=False)
        parser.add_argument("--num_iters", type=int, default=1_000)
        parser.add_argument("--log_level", type=str, default="INFO")
        parser.add_argument("--configuration", type=str, default="")
        parser.add_argument("--test", action="store_true", default=False)  # needed for local testing (optional otherwise)
        parser.add_argument("--load_only", action="store_true", default=False)
        parser.add_argument("--suppress_graphs", action="store_true", default=False)
        parser.add_argument("--local_rank", type=int, default=0)  # needed for multi-GPU training
        parser.add_argument("--bon", action="store_true", default=False)  # needed for best-of-n
        parser.add_argument("--train_type", type=str, default="")  # needed for dpo training
        parser.add_argument("--dataset", type=str, default="")  # needed for dpo training
        args = parser.parse_args()
        ScriptUtils.set_log_level(args)
        return args

    @classmethod
    def set_log_level(cls, args) -> None:
        requested = args.log_level.lower()
        specified = None
        if requested == "debug":
            specified = logging.DEBUG
        elif requested == "info":
            specified == logging.INFO
        elif requested == "warn":
            specified == logging.WARN
        elif requested == "error":
            specified == logging.ERROR
        else:
            raise Exception(f"Request log level {requested} is not eligible")

        os.environ["LOG_LEVEL"] = str(specified)

    @classmethod
    def get_debate_round_script_config(cls, args) -> DebateRoundScriptConfig:
        if args.test:
            experiment_name = args.configuration or "Single_Test"
            experiment_file_path = "experiments/configs/test_experiment.yaml"
            save_path_base = "../../debate-data/transcripts" if not args.bon else "../../debate-data/bon"
            if not args.local:
                experiment_name = "Test Experiment 2 - HPC"
                experiment_file_path = "/home/spa9663/debate/" + experiment_file_path
                save_path_base = "/home/spa9663/debate-data/transcripts" if not args.bon else "/home/spa9663/debate-data/bon"
        else:
            experiment_name = args.configuration or "Local Experiment"
            experiment_file_path = (
                "experiments/configs/standard_experiment.yaml" if not args.bon else "experiments/configs/bon_experiment.yaml"
            )
            save_path_base = "../../debate-data/transcripts" if not args.bon else "../../debate-data/bon"
            if not args.local:
                experiment_name = args.configuration or "7B-Base Experiment"
                experiment_file_path = "/home/spa9663/debate/" + experiment_file_path
                save_path_base = "/home/spa9663/debate-data/transcripts" if not args.bon else "/home/spa9663/debate-data/bon"
        return DebateRoundScriptConfig(
            experiment_name=experiment_name,
            experiment_file_path=experiment_file_path,
            save_path_base=save_path_base,
        )

    @classmethod
    def get_config_filepath(cls, train_type: TrainType) -> str:
        if train_type == TrainType.SFT:
            return "train/configs/sft_config.yaml"
        elif train_type == TrainType.DPO:
            return "train/configs/dpo_config.yaml"
        elif train_type == TrainType.PPO:
            return "train/configs/ppo_config.yaml"
        elif train_type == TrainType.PRETRAIN:
            return "train/configs/pretrain_config.yaml"
        else:
            raise Exception(f"Train type {train_type} is not recognized")

    @classmethod
    def get_training_run_script_config(cls, args, train_type: TrainType) -> ModelRunScriptConfig:
        config_filepath = ScriptUtils.get_config_filepath(train_type=train_type)
        full_dataset_filepath = args.dataset or "../../debate-data/debates-readable.jsonl"
        config_name = args.configuration or "Default - Local"
        if not args.local:
            config_filepath = "/home/spa9663/debate/" + config_filepath
            full_dataset_filepath = args.dataset or "/home/spa9663/debate-data/debates-readable.jsonl"
            config_name = args.configuration or "Extended - 7B"
        return ModelRunScriptConfig(
            config_filepath=config_filepath, full_dataset_filepath=full_dataset_filepath, config_name=config_name
        )
