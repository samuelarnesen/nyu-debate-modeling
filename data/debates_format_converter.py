"""This script loads debates that John Hughes, Dan Valentine, and Akbir Khan ran with GPT-4 (https://github.com/akbir/debate), 
and reformats them to the format that QualityDebatesLoader expects.
"""

import json
import pandas as pd


if __name__ == "__main__":

    def create_turn(text):
        turn = {
            "chars": -1,
            "charLimit": -1,
            "quoteChars": -1,
            "quoteCharLimit": -1,
            "index": -1,
            "probabilities": None,
            "role": "Debater",
            "text": text,
        }
        return turn

    df = pd.read_csv("data/datasets/john_dan_debates.csv")
    with open("data/datasets/debates-readable.jsonl", "r") as human_f:
        lines = human_f.readlines()
        human_debates = [json.loads(line) for line in lines]

    gpt_debates = []
    for i, row in df.iterrows():
        transcript = json.loads(row["transcript"])
        if transcript["swap"]:
            answers = [transcript["answers"]["incorrect"], transcript["answers"]["correct"]]
        else:
            answers = [transcript["answers"]["correct"], transcript["answers"]["incorrect"]]
        turns = []
        for round in transcript["rounds"]:
            correct_turn = create_turn(round["correct"])
            incorrect_turn = create_turn(round["incorrect"])
            if transcript["swap"]:
                turns.append(incorrect_turn)
                turns.append(correct_turn)
            else:
                turns.append(correct_turn)
                turns.append(incorrect_turn)

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
        }
        gpt_debates.append(new_debate)

    with open("data/datasets/gpt_debates.jsonl", "w") as f:
        for debate in gpt_debates:
            f.write(json.dumps(debate))
            f.write("\n")

    with open("data/datasets/human_and_gpt4_debates.jsonl", "w") as f:
        all_debates = human_debates + gpt_debates
        for debate in all_debates:
            f.write(json.dumps(debate))
            f.write("\n")
