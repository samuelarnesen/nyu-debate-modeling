#!/bin/bash

PYTHON_PROGRAM="scripts/run_debate.py"
ARGUMENTS=("--configuration=Simple_Test --num_iters=1" "--configuration=Single_Test --num_iters=1" "--configuration=Batched_Test --num_iters=40" "--configuration=Quality_Test --num_iters=10" "--configuration=Human_Test --num_iters=5" "--configuration=Labelled_Scratchpad_Test --num_iters=5" "--configuration=Dynamic_Override_Test --num_iters=5" "--configuration=Dynamic_Test --num_iters=5")
COMMON_ARGS=("--local --test --suppress_graphs --log_level=INFO")

# Loop over each argument and run the Python program
for ARG in "${ARGUMENTS[@]}"; do
    eval $(echo python "$PYTHON_PROGRAM" "${COMMON_ARGS[@]}" "$ARG")
    #echo python "$PYTHON_PROGRAM" "${COMMON_ARGS[@]}" "$ARG"
    
    # Check if the Python script exited with an error
    if [ $? -ne 0 ]; then
        echo "$ARG failed"
        echo $(echo python "$PYTHON_PROGRAM" "${COMMON_ARGS[@]}" "$ARG")
        break
    fi
done
