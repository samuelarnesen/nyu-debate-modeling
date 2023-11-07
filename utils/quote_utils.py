from utils.logger_utils import LoggerUtils
import utils.constants as constants

from difflib import SequenceMatcher
from typing import Optional
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
    def replace_invalid_quote(
        cls,
        speech_content: str,
        quote: str,
        background_text: str,
        early_stopping_threshold: int = 0.9,
        min_threshold: int = 0.8,
    ) -> str:
        def split_into_words(text) -> list[str]:
            return re.findall(r"\w+", text)

        def find_best_match() -> Optional[str]:
            split_background_text = split_into_words(background_text)
            quote_words = split_into_words(quote)

            max_ratio = 0
            best_match = None
            for i in range(len(split_background_text) - len(quote_words)):
                substring = " ".join(split_background_text[i : i + len(quote_words)])
                ratio = SequenceMatcher(None, substring, quote).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_match = substring
                if max_ratio >= early_stopping_threshold:
                    return best_match
            if max_ratio >= min_threshold:
                return best_match

            return None

        logger = LoggerUtils.get_default_logger(__name__)
        best_replacement = find_best_match()
        if best_replacement:
            logger.debug(f'Replacing "{quote}" with "{best_replacement}"')
            return re.sub(re.escape(quote), best_replacement, speech_content, flags=re.DOTALL)
        else:
            return re.sub(
                f"{re.escape(constants.QUOTE_TAG)}{re.escape(quote)}{re.escape(constants.UNQUOTE_TAG)}",
                f"{constants.INVALID_QUOTE_TAG}{quote}{constants.INVALID_UNQUOTE_TAG}",
                speech_content,
                flags=re.DOTALL,
            )

    @classmethod
    def validate_and_replace_quotes(cls, speech_content: str, background_text: str) -> str:
        updated_speech_content = speech_content
        for quote in QuoteUtils.extract_quotes(speech_content=speech_content):
            if not QuoteUtils.validate_quote(quote=quote, background_text=background_text):
                updated_speech_content = QuoteUtils.replace_invalid_quote(
                    speech_content=updated_speech_content,
                    quote=quote,
                    background_text=background_text,
                    early_stopping_threshold=constants.QUOTE_FUZZY_MATCH_EARLY_STOPPING_THRESHOLD,
                    min_threshold=constants.QUOTE_FUZZY_MATCH_MIN_THRESHOLD,
                )
        return updated_speech_content
