from script_utils import ScriptUtils, TrainType

ScriptUtils.setup_script()

from train import SupervisedTrainer, TrainUtils
from utils import SaveUtils

args = ScriptUtils.get_args()
script_config = ScriptUtils.get_training_run_script_config(args, train_type=TrainType.SFT)

config = TrainUtils.parse_config(config_name=script_config.config_name, config_filepath=script_config.config_filepath)
trainer = SupervisedTrainer.get_trainer(config=config, is_local=args.local, is_test=args.test)

if not args.load_only and not args.test:
    trainer.train()

if not args.test:
    trainer.save_model()

if config.logging_and_saving_config.merge_output_dir:
    trainer = None
    SaveUtils.save(
        base_model_name=config.model_name,
        adapter_name=config.logging_and_saving_config.output_dir,
        merge_name=config.logging_and_saving_config.merge_output_dir,
    )
