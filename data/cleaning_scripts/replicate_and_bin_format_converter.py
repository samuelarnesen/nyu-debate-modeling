"""This script loads debates that John Hughes, Dan Valentine, and Akbir Khan ran with GPT-4 (https://github.com/akbir/debate), 
and reformats them to the format that QualityDebatesLoader expects.
"""

import json
import pandas as pd
import re
import sys
from typing import Optional


output_gpt_only = "/Users/samarnesen/nyu/scratch/binned_gpt_debates_and_consultancies.jsonl"
output_combined = "/Users/samarnesen/nyu/scratch/binned_human_and_gpt4_debates_and_consultancies.jsonl"

external_debate_sources = [
    "/Users/samarnesen/Downloads/llm_debate_dataset/llm_debate_human_judge_dataset.csv",
]


def get_debaters_from_file_path(file_path: str):
    filename = file_path.split("/")[-3]
    split_debaters = filename.split("_")
    debater_one = {"model_type": None, "bo": 0, "co": 0}
    debater_two = {"model_type": None, "bo": 0, "co": 0}
    current = None
    for comp in filename.split("_"):
        if re.match("Bo\d", comp):
            current["bo"] = int(re.match("Bo(\d)", comp).group(1))
        elif re.match("Co\d", comp):
            current["co"] = int(re.match("Co(\d)", comp).group(1))
        else:
            if not current:
                current = debater_one
            else:
                current = debater_two
            current["model_type"] = comp

    return debater_one, debater_two


def get_external_debates(file_path: str):
    def create_turn(
        text: str,
        correct: bool = True,
        swap: bool = False,
        role: str = "Debater",
        probs: Optional[tuple[float, float]] = None,
    ):
        return {
            "chars": -1,
            "charLimit": -1,
            "quoteChars": -1,
            "quoteCharLimit": -1,
            "index": 0 if correct != swap else 1,
            "probabilities": probs,
            "role": role,
            "text": text,
        }

    debates = []

    df = pd.read_csv(file_path)
    df = df[~df["judge_id"].isin([11, 17, 20, 22, 29, 32, 34, 35, 36, 37])]
    df = df[(df["debate_method"] == "debate") | (df["debate_method"] == "consultancy")]
    one, two = get_debaters_from_file_path(file_path)
    for i, row in df.iterrows():
        swap = i % 2 == 1
        transcript = json.loads(row["transcript"])
        if swap:
            answers = [transcript["answers"]["incorrect"], transcript["answers"]["correct"]]
        else:
            answers = [transcript["answers"]["correct"], transcript["answers"]["incorrect"]]

        rd = transcript["rounds"][0]
        correct_turn = create_turn(text=rd["correct"], correct=True, swap=swap) if rd["correct"] else None
        incorrect_turn = create_turn(text=rd["incorrect"], correct=False, swap=swap) if rd["incorrect"] else None

        turns = [correct_turn, incorrect_turn] if not swap else [incorrect_turn, correct_turn]
        turns = [turn for turn in filter(lambda x: x is not None, turns)]

        judge_probs = [0, 0]
        if not swap:
            if row["correct"]:  # then A is correct and they voted for A
                judge_probs = (row["confidence"] / 100, 1 - (row["confidence"] / 100))
            else:  # then A is correct and they voted for B
                judge_probs = (1 - (row["confidence"] / 100), row["confidence"] / 100)
        else:
            if row["correct"]:  # then B is correct and they voted for B
                judge_probs = (1 - (row["confidence"] / 100), row["confidence"] / 100)
            else:  # then B is correct and they voted for A
                judge_probs = (row["confidence"] / 100, 1 - (row["confidence"] / 100))
        turns.append(create_turn(text="", role="Judge", probs=judge_probs))

        new_debate = {
            "storyId": "-1",
            "storyTitle": row["story_title"],
            "story": row["story_title"],
            "question": row["question"],
            "answers": answers,
            "debateId": "-1",
            "judge": "-1",
            "turns": turns,
            "isJudgeCorrect": False,
            "correctAnswer": transcript["answers"]["correct"],
            "debaters": [one, two],
        }

        debates.append(new_debate)

    return debates


def deduplicate(debates: list[dict]):
    story_id_to_debate = {}
    for debate in debates:
        key = debate["storyTitle"] + "_" + debate["question"]
        if key not in story_id_to_debate:
            story_id_to_debate[key] = debate
        else:
            existing = story_id_to_debate[key]["debaters"][0]
            current = debate["debaters"][0]
            if existing["model_type"] != current["model_type"]:
                story_id_to_debate[key] = (
                    story_id_to_debate[key]
                    if existing["model_type"] == "gpt4t" and current["model_type"] == "claude2.1"
                    else debate
                )
            elif existing["bo"] != current["bo"]:
                story_id_to_debate[key] = story_id_to_debate[key] if existing["bo"] > current["bo"] else debate
            elif existing["co"] != current["co"]:
                story_id_to_debate[key] = story_id_to_debate[key] if existing["co"] > current["co"] else debate
    for debate in debates:
        del debate["debaters"]

    return list(story_id_to_debate.values())


if __name__ == "__main__":
    gpt_debates = []
    for source in external_debate_sources:
        external_debates = get_external_debates(source)
        for debate in external_debates:
            is_truncated = False
            for turn in debate["turns"]:
                if "TRUNCATED" in turn["text"]:
                    is_truncated = True
            if not is_truncated:
                gpt_debates.append(debate)

    with open("data/datasets/quality-debates/debates-readable.jsonl", "r") as human_f:
        lines = human_f.readlines()
        human_debates = [json.loads(line) for line in lines]

    with open(output_combined, "w") as f:
        all_debates = human_debates + gpt_debates
        for debate in all_debates:
            f.write(json.dumps(debate))
            f.write("\n")
