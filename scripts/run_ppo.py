from script_utils import ScriptUtils, TrainType

ScriptUtils.setup_script()

from data import RawDataset
from train import PPOTrainerWrapper, TrainUtils
from utils import save_utils

args = ScriptUtils.get_args()
script_config = ScriptUtils.get_training_run_script_config(args, train_type=TrainType.PPO)

config = TrainUtils.parse_config(config_name=script_config.config_name, config_filepath=script_config.config_filepath)

trainer = PPOTrainerWrapper.get_trainer(config=config, is_local=args.local, is_test=args.test)
trainer.train(num_iters=10)
trainer.save_model()

if config.logging_and_saving_config.merge_output_dir:
    trainer = None
    save_utils.save(
        base_model_name=config.model_name,
        adapter_name=config.logging_and_saving_config.output_dir,
        merge_name=config.logging_and_saving_config.merge_output_dir,
    )
