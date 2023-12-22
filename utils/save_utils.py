from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch


class SaveUtils:
    @classmethod
    def save(cls, base_model_name: str, adapter_name: str, merge_name: str):
        """
        Loads a model and its adapter and saves it to the specified location.

        Params:
            base_model_name: the name (or file path) of the model to load
            adapter_name: the name (or file path) of the trained adapter
            merge_name: the file_path one wants to save the merged model to
        """
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            return_dict=True,
            torch_dtype=torch.float16,
        )

        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()
        model.save_pretrained(merge_name)
