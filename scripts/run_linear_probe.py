from script_utils import ScriptUtils, TrainType

ScriptUtils.setup_script()

from data import RawDataset
from train import ProbeTrainer, TrainUtils
from utils import SaveUtils

args = ScriptUtils.get_args()
script_config = ScriptUtils.get_training_run_script_config(args, train_type=TrainType.PROBE)

config = TrainUtils.parse_config(config_name=script_config.config_name, config_filepath=script_config.config_filepath)
dataset = TrainUtils.create_dataset(config=config, linear_idxs=config.training_hyperparameters.supplemental["linear_idxs"])

trainer = ProbeTrainer.get_trainer(config=config, raw_dataset=dataset, is_local=args.local)
trainer.train()
trainer.save_model()

if config.logging_and_saving_config.merge_output_dir:
    trainer = None
    SaveUtils.save(
        base_model_name=config.model_name,
        adapter_name=config.logging_and_saving_config.output_dir,
        merge_name=config.logging_and_saving_config.merge_output_dir,
    )
