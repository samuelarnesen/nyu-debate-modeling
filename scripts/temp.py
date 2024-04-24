from script_utils import ScriptUtils, TrainType

ScriptUtils.setup_script()

from data import QualityLoader, SplitType

import pickle

data = QualityLoader.load()


val_questions = []
for row in data.get_data(split=SplitType.VAL):
    val_questions.append(row.question)

with open("/Users/samarnesen/nyu/scratch/val_questions.p", "wb") as f:
    pickle.dump(val_questions, f)

test_questions = []
for row in data.get_data(split=SplitType.TEST):
    test_questions.append(row.question)

with open("/Users/samarnesen/nyu/scratch/test_questions.p", "wb") as f:
    pickle.dump(test_questions, f)
