#!/usr/bin/env bash

export DATA_PATH="data/test_US_1.xlsx"
export EXP_NAME="test_US_2"
export THEORY_NAME="mbti"
export CONFIG_PATH="config/example_mbti.yml"
export LLM_PROVIDER="vllm"
export LLM_MODEL="cache/Qwen/Qwen3-8B"
export API_KEY="EMPTY"
export OUTPUT_PATH="output"
export BATCH_SIZE=2
export ZHIPUAI_API_KEY="e6d073982c874667aaf80ff6dd01acb0.Np7TNMDFC0fpDV8W"

python run_batch_simulation.py \
  --data_path $DATA_PATH \
  --exp_name $EXP_NAME \
  --theory_name $THEORY_NAME \
  --config_path $CONFIG_PATH \
  --llm_provider $LLM_PROVIDER \
  --llm_model $LLM_MODEL \
  --api_key $API_KEY \
  --output_path $OUTPUT_PATH \
  --max_questions_per_person 2 \
  --batch_size $BATCH_SIZE