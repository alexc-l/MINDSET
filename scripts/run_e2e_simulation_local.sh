#!/usr/bin/env bash

export DATA_PATH="data/wvs_representatives_final/sampled/Brazil_cleaned.xlsx"
export EXP_NAME="Brazil_cleaned"
export THEORY_NAME="bigfive"
export CONFIG_PATH="config/example_big5.yml"
export LLM_PROVIDER="vllm"
export LLM_MODEL="Apriel-1.6-15b-Thinker"
export BASE_URL="http://0.0.0.0:8000/v1"
export API_KEY="EMPTY"
export OUTPUT_PATH="output/$LLM_MODEL"
export CACHE_DIR="debug_cache/$LLM_MODEL/$EXP_NAME/$THEORY_NAME"
export ZHIPUAI_API_KEY=""

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
  --debug_mode \
  --debug_cache_dir $CACHE_DIR


#sudo docker run -it --gpus all -p 8250:8250 -v /home/shenyan/lhj/projects/mindset:/workspace nvcr.m.daocloud.io/nvidia/vllm:25.09-py3