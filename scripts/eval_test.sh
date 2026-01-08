#Step1: Extract model predictions to excel result table.
DATASET_JSONL="data/ablation/unrelated_theory" # batched input dataset
STAGE="unrelated_theory"
RESULTS_DIR="results/qwen3-14b/$STAGE"

extract_country() {
    local filepath="$1"
    local filename=$(basename "$filepath")
    local name_without_ext="${filename%.jsonl}"
    echo "${name_without_ext#output_}"
}

for file in results/qwen3-14b/unrelated_theory/*.jsonl; do # batched output folder
  [[ -f "$file" ]] || continue  # 确保是普通文件
  country=$(extract_country "$file")
  echo "$(dirname "$file")"

  python refine_vllm_cache.py \
    --vllm_output_dir "$(dirname "$file")" \
    --dataset_path "$DATASET_JSONL/$country".json \
    --stage_name "$STAGE" \
    --main_output_path output_"$country".jsonl \
    --retry_output_path "retry_outputs.jsonl" \
    --matched_output_path "$(dirname "$file")/matched/${country}_matched_outputs.jsonl" \
    --baseline_parse true \
    --theory_name "$STAGE"

  python extract_answer_from_batch.py \
  --matched_jsonl_path "$(dirname "$file")/matched/${country}_matched_outputs.jsonl" \
  --output_dir "$RESULTS_DIR" \
  --theory "$STAGE"

done

#Step2: Calculate metrics
DATA_PATH="data/wvs_representatives_final/sampled"
DATA_JSON_PATH="/"
PRED_PATH="results/qwen3-14b/unrelated_theory"
MODEL="qwen3-14b"
THEORY="unrelated_theory"
PREFIX="unrelated_theory-"
SUFFIX="cleaned"

python scripts/evaluate_cluster_question_group.py \
  --real_data_dir $DATA_PATH \
  --json_dist_dir $DATA_JSON_PATH \
  --pred_data_dir $PRED_PATH \
  --suffix $SUFFIX \
  --prefix $PREFIX \
  --output_dir "results/$MODEL/$THEORY"
