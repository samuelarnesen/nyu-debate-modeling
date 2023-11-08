from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch


class SaveUtils:
    @classmethod
    def save(cls, base_model_name: str, adapter_name: str, merge_name: str):
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            return_dict=True,
            torch_dtype=torch.float16,
        )

        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()
        model.save_pretrained(merge_name)
