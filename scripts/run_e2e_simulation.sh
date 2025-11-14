#!/usr/bin/env bash

export DATA_PATH="data/test_US_1.xlsx"
export EXP_NAME="test_US_1"
export CONFIG_PATH="config/example_mbit.yml"
export LLM_PROVIDER="openai"
export API_KEY="sk-5kb8StDLQAuqopMs306fD601015f4fA2A7A6Ee526356763b"
export BASE_URL="https://api.laozhang.ai/v1"
export LLM_MODEL="gpt-3.5-turbo"
export OUTPUT_PATH="output"

python run_simulation.py \
  --data_path $DATA_PATH \
  --exp_name $EXP_NAME \
  --config_path $CONFIG_PATH \
  --llm_provider $LLM_PROVIDER \
  --base_url $BASE_URL \
  --api_key $API_KEY \
  --llm_model $LLM_MODEL \
  --include_metadata \
  --output_path $OUTPUT_PATH\
  --max_questions_per_person 1 \
  --debug_mode