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
    def split_text(cls, text: str):
        return text.split()

    @classmethod
    def validate_quote(cls, quote: str, background_text: str, prevalidated_speech: Optional[str] = None) -> bool:
        if prevalidated_speech:
            if re.search(f"{constants.QUOTE_TAG}\s*{re.escape(quote)}\s*{constants.UNQUOTE_TAG}", prevalidated_speech):
                return True
            elif re.search(
                f"{constants.INVALID_QUOTE_TAG}\s*{re.escape(quote)}\s*{constants.INVALID_UNQUOTE_TAG}", prevalidated_speech
            ):
                return False
        return quote in background_text or QuoteUtils.simplify_text(quote) in QuoteUtils.simplify_text(background_text)

    @classmethod
    def extract_quotes(cls, speech_content: str):
        return re.findall(
            f"{constants.QUOTE_TAG}(.*?){constants.UNQUOTE_TAG}", speech_content, flags=re.DOTALL
        ) + re.findall(f"{constants.INVALID_QUOTE_TAG}(.*?){constants.INVALID_UNQUOTE_TAG}", speech_content, flags=re.DOTALL)

    @classmethod
    def find_best_match(
        cls, quote: str, background_text: str, early_stopping_threshold: int = 0.9, min_threshold: int = 0.8
    ) -> Optional[str]:
        split_background_text = QuoteUtils.split_text(text=background_text)
        quote_words = re.findall(r"\w+", quote)

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

    @classmethod
    def replace_invalid_quote(
        cls,
        speech_content: str,
        quote: str,
        background_text: str,
        early_stopping_threshold: int = 0.9,
        min_threshold: int = 0.8,
    ) -> str:
        logger = LoggerUtils.get_default_logger(__name__)
        best_replacement = QuoteUtils.find_best_match(
            quote=quote,
            background_text=background_text,
            early_stopping_threshold=early_stopping_threshold,
            min_threshold=min_threshold,
        )
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

    @classmethod
    def extract_quote_context(
        cls, quote_text: str, background_text: str, context_size: int = 10, retried: bool = False
    ) -> Optional[str]:
        def get_match(quote_text: str, background_text: str, context_size: int) -> Optional[str]:
            if not quote_text or not background_text:
                return None

            pattern = r"((?:\A|\b)\s*(?:\w+\W+){0,%d})%s((?:\W+\w+){0,%d}\s*(?:\Z|\b))" % (
                context_size,
                re.escape(quote_text),
                context_size,
            )
            match = re.search(pattern, background_text, flags=re.DOTALL)
            if match:
                return "{}{}{}".format(match.group(1), quote_text, match.group(2))

        if not quote_text:
            return None

        # normal matching
        matched_text = get_match(quote_text=quote_text, background_text=background_text, context_size=context_size)
        if matched_text:
            return matched_text

        # quotation-mark-less matching
        matched_text = get_match(
            quote_text=quote_text.replace('"', ""),
            background_text=background_text.replace('"', ""),
            context_size=context_size,
        )
        if matched_text:
            return matched_text

        # replacement matching
        matched_text = get_match(
            quote_text=QuoteUtils.find_best_match(quote=quote_text, background_text=background_text),
            background_text=" ".join(QuoteUtils.split_text(text=background_text)),
            context_size=context_size,
        )
        if matched_text:
            return matched_text

        return None
