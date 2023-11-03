from script_utils import ScriptUtils

ScriptUtils.setup_script()

from data.data import RawDataset
from data.loaders.judge_preferences_loader import JudgePreferencesLoader
from train.train_utils import TrainUtils
from train.dpo_trainer import DirectPreferenceTrainer

from transformers import AutoTokenizer

args = ScriptUtils.get_args()
script_config = ScriptUtils.get_model_run_script_config(args)

tokenizer = AutoTokenizer.from_pretrained("Yukang/LongAlpaca-13B")
config = TrainUtils.parse_config(config_name=script_config.config_name, config_filepath=script_config.config_filepath)
judge_preferences_dataset = JudgePreferencesLoader.load(full_dataset_filepath=script_config.full_dataset_filepath)

for row in judge_preferences_dataset.get_data():
    prompt = tokenizer(row.prompt).input_ids
    chosen = tokenizer(row.chosen).input_ids
    rejected = tokenizer(row.rejected).input_ids
    print(len(prompt), max(len(chosen), len(rejected)))
# converted = DirectPreferenceTrainer.convert_dataset(judge_preferences_dataset)
# print(converted)
