class StringUtils:
    @classmethod
    def clean_string(cls, input_string) -> str:
        """Removes pad tokens from a model output"""
        return input_string.replace("<s>", "").replace("</s>", "").rstrip()
