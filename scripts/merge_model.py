from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str)
parser.add_argument("--adapter", type=str)
parser.add_argument("--target", type=str)
args = parser.parse_args()

base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    return_dict=True,
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(args.target)
