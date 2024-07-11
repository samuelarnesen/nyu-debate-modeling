import utils.logger_utils as logger_utils
import utils.constants as constants

from difflib import SequenceMatcher
from typing import Optional
import re

import fuzzysearch  #  potentially delete


def simplify_text(text: str):
    """Strips out characters that are often butchered by different tokenizers"""
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


def split_text(text: str):
    """Splits texts by word -- is its own function in case we want to change to re.split(r"\b", text)"""
    return text.split()


def validate_quote(quote: str, background_text: str, prevalidated_speech: Optional[str] = None) -> bool:
    """
    Returns whether a given quote is truly present in the background_text.

    Params:
        quote: the quote that is being verified
        background_text: the source where the quote allegedly came from
        prevalidated_speech: if a speech has already been validated (e.g. if we're re-judging based on
            previously-generated transcripts), then we just check whether the quote is wrapped in a quote tag
            or an invalid-quote tag.

    Returns:
        Returns true if the quote is actually in the background text.
    """
    if prevalidated_speech:
        if re.search(f"{constants.QUOTE_TAG}\s*{re.escape(quote)}\s*{constants.UNQUOTE_TAG}", prevalidated_speech):
            return True
        elif re.search(
            f"{constants.INVALID_QUOTE_TAG}\s*{re.escape(quote)}\s*{constants.INVALID_UNQUOTE_TAG}", prevalidated_speech
        ):
            return False
    return quote in background_text or simplify_text(quote) in simplify_text(background_text)


def extract_quotes(speech_content: str) -> list[str]:
    """Pulls out all the quotes (valid or invalid) from the speech"""
    return re.findall(f"{constants.QUOTE_TAG}(.*?){constants.UNQUOTE_TAG}", speech_content, flags=re.DOTALL) + re.findall(
        f"{constants.INVALID_QUOTE_TAG}(.*?){constants.INVALID_UNQUOTE_TAG}", speech_content, flags=re.DOTALL
    )


def clean_up_quotes(speech_content: str) -> list[str]:
    """Cleans up minorly butchered quotes that interfere with human comprehension and/or stats
    (empty quotes, unterminated quotes, unpaired quotes)"""

    # cleans up duplicate tags and empty quotes
    previous = speech_content
    replaced_text = speech_content
    replaced_text = re.sub(f"({constants.QUOTE_TAG}{constants.UNQUOTE_TAG})+", "", replaced_text, flags=re.DOTALL)
    replaced_text = re.sub(f"{constants.QUOTE_TAG}{2,}", constants.QUOTE_TAG, replaced_text, flags=re.DOTALL)
    replaced_text = re.sub(f"{constants.UNQUOTE_TAG}{2,}", constants.UNQUOTE_TAG, replaced_text, flags=re.DOTALL)

    # adds an unquote tag to the end of the text if there's an unterminated quote at the end
    all_quote_tags = re.findall(rf"{constants.QUOTE_TAG}|{constants.UNQUOTE_TAG}", replaced_text, flags=re.DOTALL)
    if all_quote_tags and all_quote_tags[-1] == constants.QUOTE_TAG:
        replaced_text += constants.UNQUOTE_TAG

    # removes end tags that aren't matched with a start tag
    tag_pattern = re.compile(rf"{constants.QUOTE_TAG}|{constants.UNQUOTE_TAG}")
    all_tags = tag_pattern.finditer(replaced_text)
    opening_tag_count = 0
    superfluous_tags = []
    for match in all_tags:
        tag = match.group()
        if tag == constants.QUOTE_TAG:
            opening_tag_count += 1
        elif opening_tag_count > 0:
            opening_tag_count -= 1
        else:
            superfluous_tags.append((match.start(), match.end()))
    for start, end in reversed(superfluous_tags):
        replaced_text = replaced_text[:start] + replaced_text[end:]

    return replaced_text


def find_best_match(
    quote: str, background_text: str, early_stopping_threshold: int = 0.9, min_threshold: int = 0.8
) -> Optional[str]:
    """
    Identifies whether there is a close match to a quote inside the background text.

    Params:
        quote: the quote that we're trying to match for
        background_text: the source where the quote allegedly comes from
        early_stopping_threshold: if a potential match exceeds this value, it will stop searching for more matches
        min_threshold: the minimum value needed for a match to be considered a candidate match.

    Returns:
        The best matching string that exceeds the min_threshold
    """

    quote_words = re.findall(r"\w+", quote)
    if not quote_words:
        return None

    matches = fuzzysearch.find_near_matches(quote, background_text, max_l_dist=max(len(quote) // 10, 3))
    if matches:
        return background_text[matches[0].start : matches[0].end]

    split_background_text = split_text(text=background_text)
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


def replace_quotes_containing_ellipses(
    speech_content: str, quote: str, background_text: str, early_stopping_threshold: int = 0.9, min_threshold: int = 0.8
) -> Optional[str]:
    """
    Some quotes may contain ellipses. These quotes may be marked as invalid because they contain words from
    multiple disparate sections of the text. Here, we will split up the quote based on its ellipses pattern
    and check if each subcomponent is present in the text. If the quote is partially fixed up, then we will
    return a quote containing quote tags. If not, then we will return None.
    """
    ELLIPSES = "..."
    if ELLIPSES not in quote:
        return None

    partially_validated = False
    components = []
    for subquote in filter(lambda x: len(x.strip()) > 1, quote.split(ELLIPSES)):
        if validate_quote(quote=subquote.strip(), background_text=background_text):
            partially_validated = True
            components.append(f"{constants.QUOTE_TAG}{subquote}{constants.UNQUOTE_TAG}")
        else:
            components.append(f"{constants.INVALID_QUOTE_TAG}{subquote}{constants.INVALID_UNQUOTE_TAG}")
    if partially_validated:
        return " ".join(components)
    return None


def replace_invalid_quote(
    speech_content: str,
    quote: str,
    background_text: str,
    early_stopping_threshold: int = 0.9,
    min_threshold: int = 0.8,
) -> str:
    """Tries to find the best match for a quote in the background text. This assumes that
    there is not already a direct match. If a match is found, the input quote is replaced
    with the correct text. If no match is found, the original quote is wrapped in
    invalid quote tags. See find_best_match() for explanations of the
    remaining parameters."""
    logger = logger_utils.get_default_logger(__name__)
    best_replacement = find_best_match(
        quote=quote,
        background_text=background_text,
        early_stopping_threshold=early_stopping_threshold,
        min_threshold=min_threshold,
    )
    if best_replacement:
        logger.debug(f'Replacing "{quote}" with "{best_replacement}"')
        return re.sub(re.escape(quote), best_replacement, speech_content, flags=re.DOTALL)
    else:
        potentially_fixed_quote = replace_quotes_containing_ellipses(
            speech_content=speech_content,
            quote=quote,
            background_text=background_text,
            early_stopping_threshold=early_stopping_threshold,
            min_threshold=min_threshold,
        )

        if potentially_fixed_quote:
            logger.debug(f'Replacing "{quote}" with "{potentially_fixed_quote}"')

        return re.sub(
            f"{re.escape(constants.QUOTE_TAG)}{re.escape(quote)}{re.escape(constants.UNQUOTE_TAG)}|{re.escape(constants.INVALID_QUOTE_TAG)}{re.escape(quote)}{re.escape(constants.INVALID_UNQUOTE_TAG)}",
            potentially_fixed_quote or f"{constants.INVALID_QUOTE_TAG}{quote}{constants.INVALID_UNQUOTE_TAG}",
            speech_content,
            flags=re.DOTALL,
        )


def validate_and_replace_quotes(speech_content: str, background_text: str) -> str:
    """Tries to find the best match for a quote in the background text. If the quote can be
    found directly in the underlying text, nothing happens. If a non-identical match is found,
    the input quote is replaced with the correct text. If no match is found, the original
    quote is wrapped in invalid quote tags. See find_best_match() for
    explanations of the remaining parameters."""
    updated_speech_content = clean_up_quotes(speech_content)
    for quote in extract_quotes(speech_content=updated_speech_content):
        if not validate_quote(quote=quote, background_text=background_text):
            updated_speech_content = replace_invalid_quote(
                speech_content=updated_speech_content,
                quote=quote,
                background_text=background_text,
                early_stopping_threshold=constants.QUOTE_FUZZY_MATCH_EARLY_STOPPING_THRESHOLD,
                min_threshold=constants.QUOTE_FUZZY_MATCH_MIN_THRESHOLD,
            )
    return updated_speech_content


def extract_quote_context(quote_text: str, background_text: str, context_size: int = 10) -> Optional[str]:
    """Returns the words surrounding a specified quote"""

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
        quote_text=find_best_match(quote=quote_text, background_text=background_text),
        background_text=" ".join(split_text(text=background_text)),
        context_size=context_size,
    )
    if matched_text:
        return matched_text

    return None
