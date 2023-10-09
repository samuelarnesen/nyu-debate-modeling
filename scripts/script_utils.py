from pydantic import BaseModel

import argparse
import logging
import os
import sys


class DebateRoundScriptConfig(BaseModel):
    experiment_name: str
    experiment_file_path: str
    save_path_base: str
    quotes_file_path: str


class ModelRunScriptConfig(BaseModel):
    config_filepath: str
    full_dataset_filepath: str
    config_name: str


class ScriptUtils:
    # Add the parent directory to sys.path
    @classmethod
    def set_parent_as_path(cls):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    @classmethod
    def get_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--local", action="store_true", default=False)
        parser.add_argument("--num_iters", type=int, default=1_000)
        parser.add_argument("--log_level", type=str, default="INFO")
        parser.add_argument("--configuration", type=str, default="")
        parser.add_argument("--random", action="store_true", default=False)
        parser.add_argument("--load_only", action="store_true", default=False)
        parser.add_argument("--suppress_graphs", action="store_true", default=False)
        parser.add_argument("--local_rank", type=int, default=0)
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
        if args.random:
            experiment_name = args.configuration or "Test Experiment 2"
            experiment_file_path = "experiments/configs/test_experiment.yaml"
            save_path_base = "../../debate-data/transcripts"
            quotes_file_path = "../../debate-data/quotes_dataset.p"
            if not args.local:
                experiment_name = "Test Experiment 2 - HPC"
                experiment_file_path = "/home/spa9663/debate/" + experiment_file_path
                save_path_base = "/home/spa9663/debate-data/transcripts"
                quotes_file_path = "/home/spa9663/debate-data/quotes_dataset.p"
        else:
            experiment_name = args.configuration or "Local Experiment"
            experiment_file_path = "experiments/configs/sft_experiment.yaml"
            save_path_base = "../../debate-data/transcripts"
            quotes_file_path = "../../debate-data/quotes_dataset.p"
            if not args.local:
                experiment_name = args.configuration or "7B-Base Experiment"
                experiment_file_path = "/home/spa9663/debate/" + experiment_file_path
                save_path_base = "/home/spa9663/debate-data/transcripts"
                quotes_file_path = "/home/spa9663/debate-data/quotes_dataset.p"
        return DebateRoundScriptConfig(
            experiment_name=experiment_name,
            experiment_file_path=experiment_file_path,
            save_path_base=save_path_base,
            quotes_file_path=quotes_file_path,
        )

    @classmethod
    def get_model_run_script_config(cls, args) -> ModelRunScriptConfig:
        config_filepath = "train/configs/training_config.yaml"
        full_dataset_filepath = "../../debate-data/debates-readable.jsonl"
        config_name = args.configuration or "Default - Local"
        if not args.local:
            config_filepath = "/home/spa9663/debate/" + config_filepath
            full_dataset_filepath = "/home/spa9663/debate-data/debates-readable.jsonl"
            config_name = args.configuration or "Extended - 7B"
        return ModelRunScriptConfig(
            config_filepath=config_filepath, full_dataset_filepath=full_dataset_filepath, config_name=config_name
        )
