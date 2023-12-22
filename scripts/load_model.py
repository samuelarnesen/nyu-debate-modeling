from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--save_name", type=str)
parser.add_argument("--requires_token", action="store_true", default=False)
args = parser.parse_args()

load_dotenv()

base_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    return_dict=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    token=os.getenv("META_ACCESS_TOKEN") if args.requires_token else None,
)

base_model.save_pretrained(args.save_name)

base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
base_tokenizer.save_pretrained(args.save_name)
