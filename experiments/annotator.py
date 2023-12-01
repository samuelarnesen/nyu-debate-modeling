from agents import DebateRoundSummary
import utils.constants as constants

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
import torch.nn as nn
import torch

from typing import Optional
import copy
import re


class PredictedAnnotation(BaseModel):
    statement: float
    summary: float
    analysis: float
    q_context: float
    framing: float
    refutation: float
    flourish: float
    position: float
    promise: float
    logic: float
    oob_quote: float
    reply: float
    none: float
    quote: float


class SentenceClassification(BaseModel):
    sentence: str
    annotation: PredictedAnnotation


class ParagraphClassification(BaseModel):
    paragraph: str
    annotation: PredictedAnnotation
    sentences: list[SentenceClassification]


class ClassificationConfig(BaseModel):
    top_k: Optional[int]
    min_threshold: Optional[float]
    special_quotes_handling: Optional[bool]
    combine_commentary: Optional[bool]


class Annotator:
    TAGS = [
        "statement",
        "summary",
        "analysis",
        "q_context",
        "framing",
        "refutation",
        "flourish",
        "position",
        "promise",
        "logic",
        "oob_quote",
        "reply",
        "none",
        "quote",
    ]

    DEFAULT_CONFIG = ClassificationConfig(top_k=3, min_threshold=0.2, special_quotes_handling=True, combine_commentary=False)

    def __init__(self, model_path: str):
        self.base = SentenceTransformer("all-MiniLM-L6-v2")
        self.linear = torch.load(model_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.softmax = nn.Softmax(dim=0)
        self.results = {}

    def classify(self, paragraph: str, config: ClassificationConfig = DEFAULT_CONFIG) -> ParagraphClassification:
        def clean_string(original: str, removals: list[str] = []):
            removals = removals + [
                constants.INVALID_QUOTE_TAG,
                constants.INVALID_UNQUOTE_TAG,
                constants.QUOTE_TAG,
                constants.UNQUOTE_TAG,
                ".",
                ",",
                '"',
            ]
            new_version = copy.deepcopy(original)
            for string_to_remove in removals:
                new_version = new_version.replace(string_to_remove, "")
            return new_version

        doc = self.nlp(paragraph)
        sentences = [sentence.text for sentence in doc.sents]
        sentence_lengths = [len(re.findall(r"\w+", clean_string(sentence))) for sentence in sentences]
        paragraph_length = sum(sentence_lengths)
        embeddings = self.base.encode(sentences)
        results = self.linear(torch.tensor(embeddings))

        classification_results = [
            PredictedAnnotation(**{Annotator.TAGS[i]: prob.item() for i, prob in enumerate(self.softmax(result))})
            for result in results
        ]

        # remove quotes predictions first if we're going to precisely calculate later
        if config.special_quotes_handling:
            temp_classification_results = []
            for result in classification_results:
                result_dict = result.dict()
                result_dict["quote"] = 0.0
                new_sum = sum(result_dict.values())
                renormalized_result_dict = {key: value / new_sum for key, value in result_dict.items()}
                temp_classification_results.append(PredictedAnnotation(**renormalized_result_dict))
            classification_results = temp_classification_results

        if config.top_k:
            min_threshold = config.min_threshold or 0.0
            temp_classification_results = []
            for i, result in enumerate(classification_results):
                new_entry = {tag: 0.0 for tag in Annotator.TAGS}
                probs_list = [(key, item) for key, item in result.dict().items()]
                sorted_probs_list = sorted(probs_list, key=lambda x: x[1], reverse=True)
                eligible_probs = [(tag, prob) for tag, prob in filter(lambda x: x[1] > min_threshold, sorted_probs_list)]
                if len(eligible_probs) > 0:
                    new_sum = sum([prob for tag, prob in eligible_probs])
                    for tag, prob in eligible_probs:
                        new_entry[tag] = prob / new_sum
                temp_classification_results.append(PredictedAnnotation(**new_entry))
            classification_results = temp_classification_results

        if config.special_quotes_handling:
            temp_classification_results = []
            for i, (sentence_length, sentence, result) in enumerate(
                zip(sentence_lengths, sentences, classification_results)
            ):
                quote_matches = re.findall(
                    f"{constants.QUOTE_TAG}(.*?){constants.UNQUOTE_TAG}|{constants.QUOTE_TAG}(.*)|{constants.INVALID_QUOTE_TAG}(.*?){constants.INVALID_UNQUOTE_TAG}|{constants.INVALID_UNQUOTE_TAG}(.*)",
                    sentence,
                )
                full_quotes = [match for tup in quote_matches for match in tup] if quote_matches else []

                quote_length = 0
                for quote in filter(lambda x: x, full_quotes):
                    quoteless_sentence = clean_string(sentence, [quote])
                    quoteless_length = len(re.findall(r"\w+", quoteless_sentence))
                    quote_length += sentence_length - quoteless_length
                if quote_length > 0:
                    total_results = {tag: (prob * sentence_length) for tag, prob in result.dict().items()}
                    total_results["quote"] = quote_length
                    overall_total = max(sum(total_results.values()), 1)
                    temp_classification_results.append(
                        PredictedAnnotation(**{tag: total_results[tag] / overall_total for tag in total_results})
                    )
                else:
                    temp_classification_results.append(classification_results[i])
            classification_results = temp_classification_results

        sentence_classifications = [
            SentenceClassification(sentence=sentence, annotation=annotation)
            for sentence, annotation in zip(sentences, classification_results)
        ]

        cumulative_annotation = {tag: 0.0 for tag in Annotator.TAGS}
        for i, sentence_classification in enumerate(sentence_classifications):
            for tag, prob in sentence_classification.annotation.dict().items():
                cumulative_annotation[tag] += prob * (sentence_lengths[i] / paragraph_length)

        paragraph_annotation = PredictedAnnotation(**cumulative_annotation)
        paragraph_classification = ParagraphClassification(
            paragraph=paragraph, annotation=paragraph_annotation, sentences=sentence_classifications
        )

        return paragraph_classification

    def classify_debate_round(
        self, summary: DebateRoundSummary, config: ClassificationConfig = DEFAULT_CONFIG
    ) -> dict[str, list[ParagraphClassification]]:
        speaker_to_classification = {}
        for speech in filter(
            lambda x: x.speaker in [constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME],
            summary.transcript.speeches,
        ):
            classification = self.classify(paragraph=speech.content)
            speaker_alias = (
                summary.first_debater_alias
                if speech.speaker == constants.DEFAULT_DEBATER_A_NAME
                else summary.second_debater_alias
            )
            speaker_to_classification.setdefault(speaker_alias, [])
            speaker_to_classification[speaker_alias].append(classification)
        for speaker, classifications in speaker_to_classification.items():
            self.results.setdefault(speaker, [])
            self.results[speaker] += classifications
        return speaker_to_classification

    def get_results(
        self, percentile: float = 0.8
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        average_speaker_results = {}
        lower_speaker_results = {}
        upper_speaker_results = {}
        for speaker in self.results:
            average_speaker_results[speaker] = {}
            lower_speaker_results[speaker] = {}
            upper_speaker_results[speaker] = {}
            all_results = {tag: [] for tag in Annotator.TAGS}
            for classification in self.results[speaker]:
                for tag in all_results:
                    all_results[tag].append(classification.annotation.dict()[tag])
            for tag in all_results:
                average_speaker_results[speaker][tag] = sum(all_results[tag]) / len(all_results[tag])
                lower_speaker_results[speaker][tag] = sorted(all_results[tag])[int((1 - percentile) * len(all_results[tag]))]
                upper_speaker_results[speaker][tag] = sorted(all_results[tag])[int(percentile * len(all_results[tag]))]
        return average_speaker_results, lower_speaker_results, upper_speaker_results
