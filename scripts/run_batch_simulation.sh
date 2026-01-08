#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_PATH="data/wvs_representatives_final/sampled/United_States_cleaned.xlsx"
export EXP_NAME="United_States_cleaned"
export THEORY_NAME="mbti"
export CONFIG_PATH="config/example_mbti.yml"
export LLM_PROVIDER="vllm"
export LLM_MODEL="cache/Apriel-1.6-15b-Thinker"
export API_KEY="EMPTY"
export OUTPUT_PATH="output"
export BATCH_SIZE=32
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
  --batch_size $BATCH_SIZE