from script_utils import ScriptUtils

ScriptUtils.setup_script()

from agents.debate_round import SplittableDebateRound, SplittingRule
from experiments.experiment_loader import ExperimentLoader

from datetime import datetime

args = ScriptUtils.get_args()
config = ScriptUtils.get_debate_round_script_config(args)
start_time = str(datetime.now()).replace(" ", "_")

debate_rounds, experiment = ExperimentLoader.generate_debate_rounds(
    experiment_file_path=config.experiment_file_path, name=config.experiment_name, count=args.num_iters
)

for i, debate_round in enumerate(debate_rounds):
    save_file_path_prefix = f"{config.save_path_base}/{start_time}_{i}" if not args.local else None
    first_summary, second_summary = SplittableDebateRound.run_split_round(
        debate_round=debate_round, splitting_rule=SplittingRule.OPENING_ONLY, save_file_path_prefix=save_file_path_prefix
    )
