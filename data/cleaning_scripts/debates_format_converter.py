"""This script loads debates that John Hughes, Dan Valentine, and Akbir Khan ran with GPT-4 (https://github.com/akbir/debate), 
and reformats them to the format that QualityDebatesLoader expects.
"""

import json
import pandas as pd
import re
import sys


"""
external_debate_sources = [
    "/Users/samarnesen/Downloads/sp/claude2.1_Bo16_claude2.1_Bo16/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/claude2.1_Bo4_Co8_claude2.1_Bo4_Co8/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/claude2.1_Bo4_claude2.1_Bo4/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/claude2.1_Bo8_claude2.1_Bo8/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/claude2.1_Co16_claude2.1_Co16/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/claude2.1_Co2_claude2.1_Co2/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/gpt4t_Bo16_gpt4t_Bo16/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/gpt4t_Bo1_gpt4t_Bo1/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/gpt4t_Bo32_gpt4t_Bo32/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/gpt4t_Bo4_Co8_gpt4t_Bo4_Co8/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/gpt4t_Bo4_gpt4t_Bo4/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/gpt4t_Bo8_gpt4t_Bo8/debate_sim/data0.csv",
    "/Users/samarnesen/Downloads/sp/gpt4t_Co16_gpt4t_Co16/debate_sim/data0.csv",
]

"""
output_gpt_only = "/Users/samarnesen/nyu/scratch/combined_gpt_debates.jsonl"
output_combined = "/Users/samarnesen/nyu/scratch/combined_human_and_gpt4_debates.jsonl"


external_debate_sources = [
    "/Users/samarnesen/Downloads/sp/gpt4t_Bo32_gpt4t_Bo32/debate_sim/data0.csv",
]

# output_gpt_only = "/Users/samarnesen/nyu/scratch/adj_datasets/gpt3.5_bo16.jsonl"


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
    def create_turn(text: str, correct: bool = True, swap: bool = False, role: str = "Debater"):
        return {
            "chars": -1,
            "charLimit": -1,
            "quoteChars": -1,
            "quoteCharLimit": -1,
            "index": 0 if correct != swap else 1,
            "probabilities": None,
            "role": role,
            "text": text,
        }

    debates = []

    df = pd.read_csv(file_path)
    one, two = get_debaters_from_file_path(file_path)
    for i, row in df.iterrows():
        swap = i % 2 == 1
        transcript = json.loads(row["transcript"])
        if swap:
            answers = [transcript["answers"]["incorrect"], transcript["answers"]["correct"]]
        else:
            answers = [transcript["answers"]["correct"], transcript["answers"]["incorrect"]]
        turns = []

        rd = transcript["rounds"][0]
        correct_turn = create_turn(text=rd["correct"], correct=True, swap=swap)
        incorrect_turn = create_turn(rd["incorrect"], correct=False, swap=swap)
        turns.extend([correct_turn, incorrect_turn] if not swap else [incorrect_turn, correct_turn])
        turns.append(create_turn(text="", role="Judge"))

        new_debate = {
            "storyId": "-1",
            "storyTitle": row["story_title"],
            "story": row["story"],
            "question": row["question"],
            "answers": answers,
            "debateId": "-1",
            "judge": "-1",
            "turns": turns,
            "isJudgeCorrect": False,
            "correctAnswer": row["correct answer"],
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
                if not turn["text"]:
                    print("MISSING")
                if "TRUNCATED" in turn["text"]:
                    is_truncated = True
            if not is_truncated:
                gpt_debates.append(debate)

    """
    with open(output_gpt_only, "w+") as f:
        for debate in gpt_debates:
            f.write(json.dumps(debate))
            f.write("\n")
    """

    with open("data/datasets/quality-debates/debates-readable.jsonl", "r") as human_f:
        lines = human_f.readlines()
        human_debates = [json.loads(line) for line in lines]

    """
    with open(output_combined, "w") as f:
        all_debates = human_debates + gpt_debates
        for debate in all_debates:
            f.write(json.dumps(debate))
            f.write("\n")
    """

    print(len(gpt_debates))
    print(len(human_debates))
