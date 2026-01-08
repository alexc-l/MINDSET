#!/usr/bin/env bash

export DATA_PATH="data/wvs_representatives_final/processed_countries/Japan_cleaned.xlsx"
export EXP_NAME="Japan_cleaned"
#export THEORY_NAME="MBTI"
export THEORY_NAME="bigfive"
#export CONFIG_PATH="config/example_mbti.yml"
export CONFIG_PATH="config/example_big5.yml"
export LLM_PROVIDER="openai"
#export API_KEY="sk-ZEolQUWCMpDCZFeF97B1B2E07e85420eA1B6E976Fb4c69B6"
export API_KEY="sk-5kb8StDLQAuqopMs306fD601015f4fA2A7A6Ee526356763b"
export BASE_URL="https://api.laozhang.ai/v1"
export LLM_MODEL="gpt-4o-mini"
export OUTPUT_PATH="output"

python run_simulation.py \
  --data_path $DATA_PATH \
  --exp_name $EXP_NAME \
  --theory_name $THEORY_NAME \
  --config_path $CONFIG_PATH \
  --llm_provider $LLM_PROVIDER \
  --base_url $BASE_URL \
  --api_key $API_KEY \
  --llm_model $LLM_MODEL \
  --include_metadata \
  --output_path $OUTPUT_PATH\
  --debug_mode