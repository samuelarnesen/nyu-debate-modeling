class StringUtils:
    @classmethod
    def clean_string(cls, input_string) -> str:
        return input_string.replace("<s>", "").replace("</s>", "").rstrip()
