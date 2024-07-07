DEFAULT_DEBATER_A_NAME = "Debater_A"
DEFAULT_DEBATER_B_NAME = "Debater_B"
DEFAULT_JUDGE_NAME = "Judge"
DEFAULT_BACKGROUND_TEXT = "None provided"
DEFAULT_WORD_LIMIT = 750
INSTRUCTION_PREFIX = "instruction:"
INPUT_PREFIX = "### User:"
INSTRUCTION_SUFFIX = "output:"
JUDGING_PREFIX = "I believe the winner of this debate is Debater_"
PREFERENCE_PREFIX = "Overall Score: "
WINNER = "Winner"
LOSER = "Loser"
CORRECT = "Correct"
INCORRECT = "Incorrect"
OVERALL = "Overall"
BASE_MODEL_PREFIX = "base"
DEBATER_A_IDENTIFICATION = "You are Debater_A"
BASE_PROMPT = "Base Prompt"
QUOTE_TAG = "<quote>"
UNQUOTE_TAG = "</quote>"
INVALID_QUOTE_TAG = "<invalid_quote>"
INVALID_UNQUOTE_TAG = "</invalid_quote>"
BEGIN_SPEECH_OPTIONS = [
    "Write out your speech:",
    "Now begin your speech.",
    "Please deliver your speech.",
    "We will now await your speech.",
]
BEGIN_JUDGING_OPTIONS = ["Here is the decision that the judge made:"]
QUOTE_FUZZY_MATCH_EARLY_STOPPING_THRESHOLD = 0.9
QUOTE_FUZZY_MATCH_MIN_THRESHOLD = 0.8
MAX_SCORE = 10
DEBATER_A_POSITION = 0
DEBATER_B_POSITION = 1
MAX_LENGTH = 32768
LINE_SEPARATOR = "\n######\n"
SRC_ROOT = "SRC_ROOT"
