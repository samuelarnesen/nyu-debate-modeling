#!/bin/bash

python scripts/run_debate.py --configuration=$1_Test --num_iters=75 --local --test --suppress_graphs --log_level=INFO
