from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "/vast/spa9663/models/trained_models/Llama-2-13B-32K-Merged",
    return_dict=True,
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, "/vast/spa9663/models/trained_models/Llama-2-13B-32K-dpo")
model = model.merge_and_unload()
model.save_pretrained("/vast/spa9663/models/trained_models/Llama-2-13B-32K-dpo")
