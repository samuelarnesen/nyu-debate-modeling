#!/bin/bash

PYTHON_PROGRAM="scripts/run_debate.py"
ARGUMENTS=("--configuration=Simple_Test --num_iters=1" "--configuration=Single_Test --num_iters=1" "--configuration=Batched_Test --num_iters=40" "--configuration=Quality_Test --num_iters=10" "--configuration=Quality_Test --num_iters=10" "--configuration=BoN_Test --num_iters=10" "--configuration=Previous_Run_To_Replicate_Test --num_iters=10" "--configuration=Stub_LLM_Test --num_iters=10" "--configuration=Consultancy_Test --num_iters=10" "--configuration=Empty_Round_Test --num_iters=10")
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
