from script_utils import ScriptUtils

ScriptUtils.setup_script()

from agents import OpenAIModel, ModelInput
from prompts import RoleType
from utils import InputUtils

import re
import sys

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


if __name__ == "__main__":
    batch_size = 16
    model = OpenAIModel(alias="relevance-judge", is_debater=False)
    input_texts = InputUtils.read_file_texts(
        base_path="/Users/samarnesen/nyu/debate-data/transcripts/2023-12-12_17:09:09.590983", group_by_batch=False
    )

    results = []
    current_batch = []
    for i, text in enumerate(input_texts):
        a, b = get_scratchpads(text)
        if not a or not b:
            continue

        question, first_position, second_position = get_topic(text)
        if not question or not first_position or not second_position:
            continue

        a_quotes = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(process_scratchpad(a))])
        b_quotes = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(process_scratchpad(b))])

        instructions = (
            DEFAULT_INSTRUCTIONS.replace("<QUESTION>", question)
            .replace("<DEBATER_A_POSITION", first_position)
            .replace("<DEBATER_B_POSITION>", second_position)
            .replace("<DEBATER_A_QUOTES>", a_quotes if a_quotes else "None provided")
            .replace("<DEBATER_B_QUOTES>", b_quotes if b_quotes else "None provided")
        )

        current_batch.append([ModelInput(role=RoleType.USER, content=instructions)])
        if len(current_batch) == batch_size:
            results.append(model.predict(inputs=current_batch))
            current_batch = []

    if current_batch:
        results.append(model.predict(inputs=current_batch))

    print(results)

    # TODO: match the results with the quotes and save the results somewhere





        
