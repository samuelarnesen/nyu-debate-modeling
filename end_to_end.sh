#!/bin/bash

export HF_HOME="/vast/spa9663/models/.cache"

# generates training data
python3 /home/spa9663/debate/scripts/run_debate.py --num_iters=38 --log_level='DEBUG' --configuration='13B Experiment - Train - DPO' --bon

# runs dpo
TRANSCRIPTS_SRC="~/debate-data/bon/$(ls -t ~/debate-data/bon/ | head -n 1 | cut -d '_' -f 1-2)"
python3 /home/spa9663/debate/scripts/run_dpo.py --configuration='13B - Alpaca - Basic' --dpo --dataset="$TRANSCRIPTS_SRC"

# merge
python3 /home/spa9663/debate/scripts/merge_model.py --base='/vast/spa9663/models/trained_models/Llama-2-13B-32K-Merged' --adapter='/vast/spa9663/models/trained_models/Llama-2-13B-32K-dpo-1/' --target='/vast/spa9663/models/trained_models/Llama-2-13B-32K-dpo-current/'

# validates
python3 /home/spa9663/debate/scripts/run_debate.py --num_iters=38 --log_level='DEBUG' --configuration='13B Validation - OpenAI'
