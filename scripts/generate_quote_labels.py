from script_utils import ScriptUtils

ScriptUtils.setup_script()

from agents import OpenAIModel, ModelInput
from data import QuoteRelevanceLoader, QuoteRelevanceDataset, QuoteRelevanceProcessedBatchItem, QuoteRelevanceTopicInfo
from prompts import RoleType
from utils import InputUtils
import utils.constants as constants

from pydantic import BaseModel
from tqdm import tqdm

import argparse
import pickle
import random
import re
import os

DEFAULT_INSTRUCTIONS = """
You will be presented with a topic that is going to be debated by two debaters as part of an educational exercise. 
They will be debating the answer to a reading comprehension question. 
They have read the story the question is about and their goal is to convince a judge as to the correct answer, 
despite the judge not having read the story themselves. As part of the preparations for this exercise, 
the debaters have selected quotes from a background story. I will present those quotes to you in order. 
Please respond whether you think that quote could potentially help support their side. 
Each numbered line is a separate quote. You should respond similarly, with a numeric rating 
(0 is completely irrelevant, 10 is obviously very relevant, 5 is potentially relevant) 
next to the appropriate numbered item. Do not provide any extra text. 
If the debater did not provide any quotes, we will write 'None provided'. In that case, just ignore that debater.

Example output format: 

Debater_A:
1. 3
2. 8
3. 4

Debater_B:
1. 3
2. 8
3. 4
4. 9

Here is the question they will be discussing:

<QUESTION>

Here is the side that Debater_A is defending:

<DEBATER_A_POSITION>

Here is the side Debater_B is defending:

<DEBATER_B_POSITION>

Here are the quotes that Debater_A selected:

<DEBATER_A_QUOTES>

Here are the quotes that Debater_B selected:

<DEBATER_B_QUOTES>

Now provide us the expected feedback:
"""


class QuoteRelevanceBatchItem(BaseModel):
    a_quote_list: list[str]
    b_quote_list: list[str]
    model_input: ModelInput
    question_info: QuoteRelevanceTopicInfo


class FakeOpenAIModel:
    def predict(self, a_quote_list: list[str], b_quote_list: list[str]):
        a_scores = [random.randint(0, 10) for quote in a_quote_list]
        a_text = "\n".join([f"{i + 1}. {score}" for i, score in enumerate(a_scores)])

        b_scores = [random.randint(0, 10) for quote in b_quote_list]
        b_text = "\n".join([f"{i + 1}. {score}" for i, score in enumerate(b_scores)])

        return f"{constants.DEFAULT_DEBATER_A_NAME}:\n{a_text}\n\n{constants.DEFAULT_DEBATER_B_NAME}:\n{b_text}"


def get_scratchpads(text: str):
    a_match = re.search(r"This is what Debater_A said during their speech\.(.*?)#####", text, flags=re.DOTALL)
    b_match = re.search(r"This is what Debater_B said during their speech\.(.*?)#####", text, flags=re.DOTALL)
    return a_match.group(1) if a_match else None, b_match.group(1) if b_match else None


def get_topic(text: str):
    question = None
    first_position = None
    second_position = None
    full_topic_section = re.search(
        r"Here is the topic they will be debating:(.*)This is what Debater_A said during their speech",
        text,
        re.DOTALL,
    )
    if not full_topic_section:
        return question, first_position, second_position
    matching_text = full_topic_section.group(1).strip()

    question_section = re.search(r"(.*)Debater_A will defend the position", matching_text, re.DOTALL)
    if not question_section:
        return question, first_position, second_position
    question = question_section.group(1).strip()

    first_position_section = re.search(
        r"Debater_A will defend the position that the answer is \"(.*)\s*\"\s*\.\s*Debater_B", matching_text
    )
    if not first_position_section:
        return question, first_position, second_position
    first_position = first_position_section.group(1).strip()

    second_position_section = re.search(
        r"Debater_B will defend the position that the answer is \"(.*)\s*\"\s*\.\s*", matching_text
    )
    if not second_position_section:
        return question, first_position, second_position
    second_position = second_position_section.group(1).strip()

    return question, first_position, second_position


def process_scratchpad(scratchpad_text: str):
    return re.findall(r"<quote>(.*?)</quote>", scratchpad_text)


def process_model_output(output: str, a_quote_list: list[str], b_quote_list: list[str]):
    debater_a_match = re.search(
        f"{constants.DEFAULT_DEBATER_A_NAME}:(.*?){constants.DEFAULT_DEBATER_A_NAME}:", output, flags=re.DOTALL
    )
    debater_a_text = debater_a_match.group(1) if debater_a_match else ""
    debater_b_match = re.search(f"{constants.DEFAULT_DEBATER_B_NAME}:(.*)", output, flags=re.DOTALL)
    debater_b_text = debater_b_match.group(1) if debater_b_match else ""

    a_quote_map = {}
    debater_a_score_lines = re.findall(r"\d.\s*\d+", debater_a_text, flags=re.DOTALL)
    for i, (quote, score_line) in enumerate(zip(a_quote_list, debater_a_score_lines)):
        a_quote_map[quote] = int(re.search(r"\d.\s*(\d+)", score_line, flags=re.DOTALL).group(1))

    b_quote_map = {}
    debater_b_score_lines = re.findall(r"\d.\s*\d+", debater_b_text, flags=re.DOTALL)
    for i, (quote, score_line) in enumerate(zip(b_quote_list, debater_b_score_lines)):
        b_quote_map[quote] = int(re.search(r"\d.\s*(\d+)", score_line, flags=re.DOTALL).group(1))

    return a_quote_map, b_quote_map


if __name__ == "__main__":
    root = os.environ[constants.SRC_ROOT]
    batch_size = 8

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()

    model = OpenAIModel(alias="relevance-judge", is_debater=False) if not args.test else FakeOpenAIModel()
    input_texts = InputUtils.read_file_texts(base_path=f"{root}outputs/transcripts/{args.timestamp}")

    results = []
    current_batch = []
    total_score = 0
    total_quotes = 0
    for i, text in tqdm(enumerate(input_texts)):
        a, b = get_scratchpads(text)
        if not a or not b:
            continue

        question, first_position, second_position = get_topic(text)
        if not question or not first_position or not second_position:
            continue

        a_quote_list = set(process_scratchpad(a))
        b_quote_list = set(process_scratchpad(b))
        a_quotes = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(a_quote_list)])
        b_quotes = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(b_quote_list)])

        instructions = (
            DEFAULT_INSTRUCTIONS.replace("<QUESTION>", question)
            .replace("<DEBATER_A_POSITION>", first_position)
            .replace("<DEBATER_B_POSITION>", second_position)
            .replace("<DEBATER_A_QUOTES>", a_quotes if a_quotes else "")
            .replace("<DEBATER_B_QUOTES>", b_quotes if b_quotes else "")
        )

        current_batch.append(
            QuoteRelevanceBatchItem(
                a_quote_list=a_quote_list,
                b_quote_list=b_quote_list,
                model_input=ModelInput(role=RoleType.USER, content=instructions),
                question_info=QuoteRelevanceTopicInfo(
                    question=question, a_position=first_position, b_position=second_position
                ),
            )
        )
        if len(current_batch) == batch_size or i == len(input_texts) - 1:
            model_inputs = [[item.model_input] for item in current_batch]
            predictions = (
                model.predict(model_inputs)
                if not args.test
                else [model.predict(item.a_quote_list, item.b_quote_list) for item in current_batch]
            )

            for prediction, item in zip(predictions, current_batch):
                a_quote_map, b_quote_map = process_model_output(
                    output=prediction, a_quote_list=item.a_quote_list, b_quote_list=item.b_quote_list
                )
                results.append(
                    QuoteRelevanceProcessedBatchItem(
                        a_quote_map=a_quote_map, b_quote_map=b_quote_map, question_info=item.question_info
                    )
                )
                total_score += sum(a_quote_map.values()) + sum(b_quote_map.values())
                total_quotes += len(a_quote_map.values()) + len(b_quote_map.values())

            current_batch = []

    pickle_path = f"{root}data/datasets/quote-relevance/quote-relevance.p"

    if args.save or not args.test:
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)

    dataset = QuoteRelevanceLoader.load()

    average_score = total_score / total_quotes
    print(f"Average score is {average_score}")
