from script_utils import ScriptUtils, TrainType

ScriptUtils.setup_script()

from data import RawDataset
from train import IterativeDirectPreferenceTrainer, TrainUtils
from utils import save_utils

args = ScriptUtils.get_args()
script_config = ScriptUtils.get_training_run_script_config(args, train_type=TrainType.DPO)

config = TrainUtils.parse_config(config_name=script_config.config_name, config_filepath=script_config.config_filepath)
trainer = IterativeDirectPreferenceTrainer(config=config, smooth=True, is_local=args.test)
epoch_size = config.training_hyperparameters.supplemental["epoch_size"]

if not args.test:
    trainer.train(epoch_size=epoch_size)
else:
    samples = trainer.get_samples(start_idx=0, epoch_size=epoch_size)
