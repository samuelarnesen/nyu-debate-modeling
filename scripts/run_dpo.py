from script_utils import ScriptUtils

ScriptUtils.setup_script()

from data.data import RawDataset
from data.loaders.judge_preferences_loader import JudgePreferencesLoader
from train.train_utils import TrainUtils
from train.dpo_trainer import DPOTrainer

args = ScriptUtils.get_args()
script_config = ScriptUtils.get_model_run_script_config(args)

config = TrainUtils.parse_config(config_name=script_config.config_name, config_filepath=script_config.config_filepath)
judge_preferences_dataset = JudgePreferencesLoader.load(full_dataset_filepath=script_config.full_dataset_filepath)

trainer = DPOTrainer.get_trainer(config=config, raw_dataset=judge_preferences_dataset, is_local=args.local)
if not args.load_only:
    trainer.train()
trainer.save_model()
