from script_utils import ScriptUtils

ScriptUtils.set_parent_as_path()

from data.data import RawDataset
from data.loaders.quality_debates_loader import QualityDebatesLoader
from train.train_utils import TrainUtils

args = ScriptUtils.get_args()

config_filepath = "./train/training_config.yaml"
full_dataset_filepath = "../debate-data/debates-readable.jsonl"
config_name = "Default - Local"
if not args.local:
    config_filepath = "/home/spa9663/debate/" + config_filepath
    full_dataset_filepath = "/home/spa9663/debate-data/debates-readable.jsonl"
    config_name = "Default - HPC"

config = TrainUtils.parse_config(config_name=config_name, config_filepath=config_filepath)
quality_debates_dataset = QualityDebatesLoader.load(full_dataset_filepath=full_dataset_filepath)
trainer = TrainUtils.get_trainer(config=config, raw_dataset=quality_debates_dataset, is_local=args.local)
trainer.train()
trainer.save_model()
