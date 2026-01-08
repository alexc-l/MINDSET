#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
shopt -s nullglob  # 无匹配时返回空而不是通配符本身
export LLM_MODEL="cache/qwen3-8b-awq"
export OUTPUT_PATH="output/qwen3-8b"
export BASELINE="durmus_2023"
export BATCH_SIZE=1024


for file in data/wvs_representatives_final/sampled/*.xlsx; do
    [[ -f "$file" ]] || continue  # 确保是普通文件
    export DATA_PATH=$file
    export OUTPUT_PATH="output/$LLM_MODEL/$BASELINE"
    echo "Running simulation for $COUNTRY-$BASELINE"
    python scripts/run_baselines_batch_vllm.py \
      --data_path "$file" \
      --baseline $BASELINE \
      --model $LLM_MODEL \
      --tensor_parallel_size 4 \
      --enforce_eager \
      --max_num_seqs $BATCH_SIZE \
      --temperature 0.95 \
      --top_p 0.7 \
      --max_tokens 2048 \
      --output_dir baseline_outputs/gpt-4o-mini
done

