from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default="current")
parser.add_argument("--adapter", type=str)
parser.add_argument("--target", type=str, default="current")
args = parser.parse_args()

base_model = AutoModelForCausalLM.from_pretrained(
    f"/vast/spa9663/models/trained_models/Llama-2-13B-32K-dpo-{args.base}",
    return_dict=True,
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, f"/vast/spa9663/models/trained_models/Llama-2-13B-32K-dpo-{args.adapter}")
model = model.merge_and_unload()
model.save_pretrained(f"/vast/spa9663/models/trained_models/Llama-2-13B-32K-dpo-{args.target}")
