from script_utils import ScriptUtils

ScriptUtils.set_parent_as_path()

from data.data import RawDataset
from data.loaders.quality_debates_loader import QualityDebatesLoader
from train.train_utils import TrainUtils

config = TrainUtils.parse_config(config_name="Default", config_filepath="./train/training_config.yaml")
quality_debates_dataset = QualityDebatesLoader.load(full_dataset_filepath="../debate-data/debates-readable.jsonl")
trainer = TrainUtils.get_trainer(config=config, raw_dataset=quality_debates_dataset, is_local=True)
