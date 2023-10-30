import utils.constants as constants

import re


class QuoteUtils:
    @classmethod
    def simplify_text(cls, text: str):
        replaced_text = (
            text.replace(",", "")
            .replace(".", "")
            .replace('"', "")
            .replace("'", "")
            .replace("\n", " ")
            .replace(constants.QUOTE_TAG, "")
            .replace(constants.UNQUOTE_TAG, "")
            .lower()
        )
        return re.sub("\s+", " ", replaced_text)

    @classmethod
    def validate_quote(cls, quote: str, background_text: str) -> bool:
        return quote in background_text or QuoteUtils.simplify_text(quote) in QuoteUtils.simplify_text(background_text)

    @classmethod
    def extract_quotes(cls, speech_content: str):
        return re.findall(f"{constants.QUOTE_TAG}(.*?){constants.UNQUOTE_TAG}", speech_content) + re.findall(
            f"{constants.INVALID_QUOTE_TAG}(.*?){constants.INVALID_UNQUOTE_TAG}", speech_content
        )

    @classmethod
    def validate_and_replace_quotes(cls, speech_content: str, background_text: str) -> str:
        updated_speech_content = speech_content
        for quote in QuoteUtils.extract_quotes(speech_content=speech_content):
            if not QuoteUtils.validate_quote(quote=quote, background_text=background_text):
                updated_speech_content = re.sub(
                    f"{re.escape(constants.QUOTE_TAG)}{re.escape(quote)}{re.escape(constants.UNQUOTE_TAG)}",
                    f"{constants.INVALID_QUOTE_TAG}{quote}{constants.INVALID_UNQUOTE_TAG}",
                    updated_speech_content,
                    flags=re.DOTALL,
                )
        return updated_speech_content
