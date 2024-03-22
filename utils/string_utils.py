def clean_string(input_string) -> str:
    """Removes pad tokens from a model output"""
    return input_string.replace("<s>", "").replace("</s>", "").rstrip()
