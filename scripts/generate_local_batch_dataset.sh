# stages for MBTI: stress_chara, mbti_select, get_stack, assign_impact, reason, apply_constraints
#

# step 1
python build_local_infer_batch_dataset.py \
  --excel_dir data/test \
  --theory_name big5 \
  --config_path config/example_big5.yml \
  --stage_name reason \
  --prev_stages_dir cache_batch/big5 \
  --output_jsonl datasets/reason_all.jsonl \
  --produce_dataset

# step 2

#python scripts/vllm_infer.py \
#  --model_name_or_path cache/qwen3-14b-awq \
#  --dataset test_apply_constraints \
#  --dataset_dir data \
#  --template qwen3 \
#  --temperature 0.95 \
#  --top_p 0.7 \
#  --save_name results/output.jsonl \
#  --max_new_tokens 32768 \
#  --vllm_config '{"enforce_eager": true, "gpu_memory_utilization": 0.8}' \
#  --batch_size 1024

# step 3
#python refine_vllm_cache.py \
#  --vllm_output_dir output/batched_outputs/apply_constraints \
#  --dataset_path datasets/apply_constraints_all.jsonl \
#  --stage_name apply_constraints \
#  --main_output_path test_apply_constraints_output.jsonl \
#  --matched_output_path cache_batch/apply_constraints_matched_outputs.jsonl \
#  --retry_dataset_path retry.jsonl

# if stage_name_retires.jsonl is not empty, remove stage_name_retires.jsonl once the inference is complete
#python scripts/vllm_infer.py \
#  --model_name_or_path cache/qwen3-14b-awq \
#  --dataset retries \
#  --dataset_dir data \
#  --template qwen3 \
#  --temperature 0.95 \
#  --top_p 0.7 \
#  --save_name results/stage_name_retires_output.jsonl \
#  --max_new_tokens 32768 \
#  --vllm_config '{"enforce_eager": true, "gpu_memory_utilization": 0.8}' \
#  --batch_size 1024

# When all stages are complete, and no retries for the last stage
#python extract_answer_from_batch.py \
#  --matched_jsonl_path cache_batch/apply_constraints_matched_outputs.jsonl \
#  --output_dir results \
#  --theory MBTI