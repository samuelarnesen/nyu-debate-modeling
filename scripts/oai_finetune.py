from script_utils import ScriptUtils, TrainType

ScriptUtils.setup_script()

from openai import OpenAI

client = OpenAI()

"""
client.files.create(
  file=open("/Users/samarnesen/nyu/scratch/nyu-blind-rounds.jsonl", "rb"),
  purpose="fine-tune"
)
"""
"""
client.fine_tuning.jobs.create(
  training_file="file-n7fvnScZBWodZ7F8nLGk4eQG", 
  model="gpt-4",
  hyperparameters={"n_epochs": 2}
)
"""
