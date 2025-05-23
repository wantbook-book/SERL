set -ex
# Start time
start_time=$(date +%s)

# ================need to modify=======================
OUTPUT_DIRS=(
    "/path/to/your/SeRL/evaluation/Math-Benchmarks/outputs/your_model_1/math_eval_sampling_n"
    "/path/to/your/SeRL/evaluation/Math-Benchmarks/outputs/your_model_2/math_eval_sampling_n"
    "/path/to/your/SeRL/evaluation/Math-Benchmarks/outputs/your_model_3/math_eval_sampling_n"
)
SUBDIRS=("math_500" "math_hard" "asdiv" "college_math" "tabmwp")
MAJ_K=16
FILE_NAME="test_pure_-1_seed0_t1.0_s0_e-1_maj_eval.jsonl"
# ================need to modify=======================

for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    # Check if the root directory exists
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Skipped $OUTPUT_DIR: Directory does not exist."
        continue
    fi
    # Iterate through subdirectories
    for SUBDIR in "${SUBDIRS[@]}"; do
        # Check if the subdirectory exists
        SUBDIR="$OUTPUT_DIR/$SUBDIR"
        if [ ! -d "$SUBDIR" ]; then
            echo "Skipped $SUBDIR: Directory does not exist."
            continue
        fi

        INPUT_JSONL="$SUBDIR/$FILE_NAME"
        if [ -f "$INPUT_JSONL" ]; then
            OUTPUT_JSONL="${INPUT_JSONL%.jsonl}_output.jsonl"
            echo "Evaluating: $INPUT_JSONL"
            python evaluate_outputs_maj.py --input_jsonl "$INPUT_JSONL" --output_jsonl "$OUTPUT_JSONL" --maj_n ${MAJ_K} --gold_is_latex 
        else
            echo "Skipped $SUBDIR: test.jsonl not found."
        fi
    done
done


# End time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"