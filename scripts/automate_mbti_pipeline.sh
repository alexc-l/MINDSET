#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# automate_mbti_pipeline.sh
# Automates the MBTI batch reasoning pipeline for all stages.

# Set variables
THEORY="mbti"
CONFIG="config/example_mbti.yml"
EXCEL_DIR="data/PEW-continue-qwen3-14b"
CACHE_DIR="cache_batch/qwen3-14b/PEW/mbti"
DATASETS_DIR="datasets/qwen3-14b/PEW/mbti"
OUTPUT_BASE_DIR="output/batched_outputs/qwen3-14b/PEW/mbti"
RESULTS_DIR="results/qwen3-14b/PEW/mbti"
MODEL_PATH="cache/qwen3-14b"
TEMPLATE="qwen3"
TEMPERATURE=0.95
TOP_P=0.7
MAX_NEW_TOKENS=32768
VLLM_CONFIG='{"enforce_eager": true, "gpu_memory_utilization": 0.8}'
BATCH_SIZE=1024
DATASET_PREFIX="PEW-qwen3-14b"

echo "Dir cleaning"
rm -rf CACHE_DIR
rm -rf DATASETS_DIR
rm -rf OUTPUT_BASE_DIR
rm -rf RESULTS_DIR

# Create directories if not exist
mkdir -p $CACHE_DIR $DATASETS_DIR $OUTPUT_BASE_DIR $RESULTS_DIR

# Stages from yml (hardcoded based on content)
STAGES=("stress_chara" "mbti_select" "get_stack" "assign_impact" "reason" "synthesis" "apply_constraints")

# LLM stages (those with prompt_paths)
LLM_STAGES=("stress_chara" "mbti_select" "assign_impact" "reason" "synthesis" "apply_constraints")

# Function to check if stage is LLM
is_llm_stage() {
  for llm in "${LLM_STAGES[@]}"; do
    if [ "$1" == "$llm" ]; then
      return 0
    fi
  done
  return 1
}

# Loop over each stage
for STAGE in "${STAGES[@]}"; do
  echo "Processing stage: $STAGE"

  PRODUCE=""
  DATASET_JSONL="data/$DATASETS_DIR/${STAGE}_all.jsonl"
  if is_llm_stage "$STAGE"; then
    PRODUCE="--produce_dataset"
  fi

  # Step 1: Build dataset or run pure python
  python build_local_infer_batch_dataset.py \
    --excel_dir "$EXCEL_DIR" \
    --theory_name "$THEORY" \
    --config_path "$CONFIG" \
    --stage_name "$STAGE" \
    --prev_stages_dir "$CACHE_DIR" \
    --output_jsonl "$DATASET_JSONL" \
    --dataset_prefix "$DATASET_PREFIX" \
    $PRODUCE

  # Move outputs to cache_dir (for pure python stages)
  OUTPUTS_FILE="${STAGE}_outputs.jsonl"
  if [ -f "$OUTPUTS_FILE" ]; then
    mv "$OUTPUTS_FILE" "$CACHE_DIR/"
    echo "Moved $OUTPUTS_FILE to $CACHE_DIR"
  fi

  # If LLM stage, proceed with inference and refinement
  if is_llm_stage "$STAGE"; then
    STAGE_OUTPUT_DIR="$OUTPUT_BASE_DIR/$STAGE"
    mkdir -p "$STAGE_OUTPUT_DIR"

    # Initial inference
    python scripts/vllm_infer.py \
      --model_name_or_path "$MODEL_PATH" \
      --dataset "${DATASET_PREFIX}_${THEORY}_${STAGE}" \
      --dataset_dir data \
      --template "$TEMPLATE" \
      --temperature $TEMPERATURE \
      --top_p $TOP_P \
      --save_name "$STAGE_OUTPUT_DIR/vllm_outputs.jsonl" \
      --max_new_tokens $MAX_NEW_TOKENS \
      --vllm_config "$VLLM_CONFIG" \
      --batch_size $BATCH_SIZE

    # Retry loop
    while true; do
      RETRY_JSONL="data/$DATASETS_DIR/${STAGE}_retry.jsonl"
      RETRY_INDICES="$STAGE_OUTPUT_DIR/retry_indices.txt"

      # Step 3: Refine
      python refine_vllm_cache.py \
        --vllm_output_dir "$STAGE_OUTPUT_DIR" \
        --dataset_path "$DATASET_JSONL" \
        --stage_name "$STAGE" \
        --main_output_path "vllm_outputs.jsonl" \
        --retry_output_path "retry_outputs.jsonl" \
        --matched_output_path "$CACHE_DIR/${STAGE}_matched_outputs.jsonl" \
        --retry_dataset_path "$RETRY_JSONL" \
        --retry_indices_path "$RETRY_INDICES" \
        --theory_name "$THEORY"

      # Check if retry needed
      if [ -s "$RETRY_JSONL" ]; then
        echo "Retry needed for $STAGE. Running inference on retry dataset..."

        # Step 4: Inference on retry
        python scripts/vllm_infer.py \
          --model_name_or_path "$MODEL_PATH" \
          --dataset "${STAGE}_${THEORY}_retries" \
          --dataset_dir data \
          --template "$TEMPLATE" \
          --temperature $TEMPERATURE \
          --top_p $TOP_P \
          --save_name "$STAGE_OUTPUT_DIR/retry_outputs.jsonl" \
          --max_new_tokens $MAX_NEW_TOKENS \
          --vllm_config "$VLLM_CONFIG" \
          --batch_size $BATCH_SIZE

        rm $RETRY_JSONL
      else
        echo "No more retries needed for $STAGE."
        break
      fi
    done
  fi
done

# Final extraction (on last stage's matched outputs)
LAST_STAGE="${STAGES[-1]}"
python extract_answer_from_batch.py \
  --matched_jsonl_path "$CACHE_DIR/${LAST_STAGE}_matched_outputs.jsonl" \
  --output_dir "$RESULTS_DIR" \
  --theory "$THEORY"

echo "Pipeline complete!"