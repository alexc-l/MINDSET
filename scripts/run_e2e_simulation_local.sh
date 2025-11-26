#!/usr/bin/env bash

export DATA_PATH="data/test_US_1.xlsx"
export EXP_NAME="test_US_1"
export THEORY_NAME="bigfive"
export CONFIG_PATH="config/example_big5.yml"
export LLM_PROVIDER="vllm"
export LLM_MODEL="cache/Qwen/Qwen3-8B"
export BASE_URL="http://0.0.0.0:8000/v1"
export API_KEY="EMPTY"
export OUTPUT_PATH="output"
export ZHIPUAI_API_KEY="e6d073982c874667aaf80ff6dd01acb0.Np7TNMDFC0fpDV8W"

python run_simulation.py \
  --data_path $DATA_PATH \
  --exp_name $EXP_NAME \
  --theory_name $THEORY_NAME \
  --config_path $CONFIG_PATH \
  --llm_provider $LLM_PROVIDER \
  --llm_model $LLM_MODEL \
  --base_url $BASE_URL \
  --api_key $API_KEY \
  --include_metadata \
  --output_path $OUTPUT_PATH\
  --max_questions_per_person 1 \
  --debug_mode


#sudo docker run -it --gpus all -p 8250:8250 -v /home/shenyan/lhj/projects/mindset:/workspace nvcr.m.daocloud.io/nvidia/vllm:25.09-py3