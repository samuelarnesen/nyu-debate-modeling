#!/bin/bash

python scripts/run_debate.py --configuration=$1_Test --num_iters=10 --local --test --log_level=INFO --suppress_graphs
