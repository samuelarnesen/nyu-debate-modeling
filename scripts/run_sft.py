from script_utils import ScriptUtils

ScriptUtils.setup_script()

from data.data import RawDataset
from data.loaders.annotated_quality_debates_loader import AnnotatedQualityDebatesLoader
from data.loaders.quality_debates_loader import QualityDebatesLoader
from train.train_utils import TrainUtils
from train.sft_trainer import SupervisedTrainer
from utils.save_utils import SaveUtils

args = ScriptUtils.get_args()
script_config = ScriptUtils.get_model_run_script_config(args)

config = TrainUtils.parse_config(config_name=script_config.config_name, config_filepath=script_config.config_filepath)
quality_debates_dataset = (
    QualityDebatesLoader.load(full_dataset_filepath=script_config.full_dataset_filepath, deduplicate=False)
    if not config.prompt_config.dynamic_prompts_file_path or not config.prompt_config.dynamic_prompt_name
    else AnnotatedQualityDebatesLoader.load(
        full_dataset_filepath=script_config.full_dataset_filepath,
        deduplicate=False,
        annotations_file_path=config.prompt_config.annotations_file_path,
    )
)

trainer = SupervisedTrainer.get_trainer(config=config, raw_dataset=quality_debates_dataset, is_local=args.local)
if not args.load_only:
    trainer.train()
trainer.save_model()

if config.logging_and_saving_config.merge_output_dir:
    trainer = None
    SaveUtils.save(
        base_model_name=config.model_name,
        adapter_name=config.logging_and_saving_config.output_dir,
        merge_name=config.logging_and_saving_config.merge_output_dir,
    )
