#!/bin/bash

python scripts/run_debate.py --configuration=$1_Test --num_iters=2 --local --test --log_level=INFO --suppress_graphs
