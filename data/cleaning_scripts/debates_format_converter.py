"""This script loads debates that John Hughes, Dan Valentine, and Akbir Khan ran with GPT-4 (https://github.com/akbir/debate), 
and reformats them to the format that QualityDebatesLoader expects.
"""

import json
import pandas as pd


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
output_gpt_only = "/Users/samarnesen/nyu/scratch/mega_gpt_debates.jsonl"
output_combined = "/Users/samarnesen/nyu/scratch/mega_human_and_gpt4_debates.jsonl"

def get_external_debates(file_path: str):
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

    debates = []

    print(file_path)
    df = pd.read_csv(file_path)
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
                incorrect_turn["index"] = 0
                correct_turn["index"] = 1
                turns.append(incorrect_turn)
                turns.append(correct_turn)
            else:
                correct_turn["index"] = 0
                incorrect_turn["index"] = 1
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
            "correctAnswer": row["correct answer"],
        }
        debates.append(new_debate)

    return debates

if __name__ == "__main__":

    gpt_debates = []
    for source in external_debate_sources:
        gpt_debates += get_external_debates(source)

    with open(output_gpt_only, "w+") as f:
        for debate in gpt_debates:
            f.write(json.dumps(debate))
            f.write("\n")

    with open("data/datasets/quality-debates/debates-readable.jsonl", "r") as human_f:
        lines = human_f.readlines()
        human_debates = [json.loads(line) for line in lines]


    with open(output_combined, "w") as f:
        all_debates = human_debates + gpt_debates
        for debate in all_debates:
            f.write(json.dumps(debate))
            f.write("\n")
