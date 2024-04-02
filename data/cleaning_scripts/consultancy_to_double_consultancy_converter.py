import os, sys, json

src_root = '/Users/samarnesen/nyu/debate/nyu-debate-modeling/'
os.environ["SRC_ROOT"] = src_root
sys.path.insert(0, src_root)

from utils import input_utils, InputType

output_prefix = "/Users/samarnesen/nyu/scratch/runs/double-consultancy-val-sft/"
file_prefix = "2024-03-25_14:22:26.620890"

texts = [json.loads(x) for x in input_utils.read_file_texts(f"{src_root}outputs/transcripts/{file_prefix}", input_type=InputType.JSON_TRANSCRIPT)]

idx = 0
new_files = []
while idx < len(texts):
	first = texts[idx]
	second = texts[idx + 1]
	assert first["metadata"]["question"] == second["metadata"]["question"]
	assert first["metadata"]["first_debater_answer"] != second["metadata"]["first_debater_answer"]
	assert first["metadata"]["first_debater_answer"] == second["metadata"]["second_debater_answer"]


	first_speakers = set([speech["speaker"] for speech in first["speeches"]])
	second_speakers = set([speech["speaker"] for speech in second["speeches"]])

	first_speeches = [speech for speech in filter(lambda x: x["speaker"] == "Debater_A", first["speeches"])]
	second_speeches = [speech for speech in filter(lambda x: x["speaker"] == "Debater_B", second["speeches"])]

	new = {"metadata": first["metadata"], "speeches": first_speeches + second_speeches}

	with open(f"{output_prefix}{file_prefix}_{idx // 2}_0.json", "w") as f:
		json.dump(new, f)

	idx += 2
