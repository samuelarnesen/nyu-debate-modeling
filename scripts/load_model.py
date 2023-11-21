from transformers import AutoModelForCausalLM
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--save_name", type=str)
args = parser.parse_args()

base_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    return_dict=True,
    torch_dtype=torch.float16,
)

base_model.save(args.save_name)
