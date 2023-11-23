#!/bin/bash

export HF_HOME="/vast/spa9663/models/.cache"
set -e

# generates training data
#python3 /home/spa9663/debate/scripts/run_debate.py --num_iters=207 --log_level='DEBUG' --configuration='13B Experiment - Train - DPO' --bon

# runs dpo
TRANSCRIPTS_SRC="/home/spa9663/debate-data/bon/$(ls -t ~/debate-data/bon/ | head -n 1 | cut -d '_' -f 1-2)"

echo $TRANSCRIPTS_SRC
python3 /home/spa9663/debate/scripts/run_dpo.py --configuration='13B - Alpaca - Updated' --dpo --dataset="$TRANSCRIPTS_SRC"

# validates
python3 /home/spa9663/debate/scripts/run_debate.py --num_iters=34 --log_level='DEBUG' --configuration='13B Validation - OpenAI'
