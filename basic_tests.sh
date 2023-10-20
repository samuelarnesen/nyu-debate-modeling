#!/bin/bash

PYTHON_PROGRAM="scripts/test_debate_round.py"
ARGUMENTS=("--configuration=Single_Test --num_iters=1" "--configuration=BoN_Test --num_iters=1" "--configuration=Batched_Test --num_iters=40")
COMMON_ARGS=("--local --test --suppress_graphs --log_level=INFO")

# Loop over each argument and run the Python program
for ARG in "${ARGUMENTS[@]}"; do
    eval $(echo python "$PYTHON_PROGRAM" "${COMMON_ARGS[@]}" "$ARG")
    
    # Check if the Python script exited with an error
    if [ $? -ne 0 ]; then
        echo "$ARG failed"
        break
    fi
done
