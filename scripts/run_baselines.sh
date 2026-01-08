#!/usr/bin/env bash
shopt -s nullglob  # 无匹配时返回空而不是通配符本身
export BASE_URL="https://api.laozhang.ai/v1"
export LLM_MODEL="gpt-4o-mini"
export OUTPUT_PATH="output/$LLM_MODEL"
export API_KEY="sk-5kb8StDLQAuqopMs306fD601015f4fA2A7A6Ee526356763b"
export BASELINE="durmus_2023"

extract_country() {
    local filepath="$1"
    local filename=$(basename "$filepath")      # 提取文件名: Brazil_cleaned.xlsx
    local name_without_ext="${filename%.xlsx}"  # 去掉.xlsx: Brazil_cleaned
    local country="${name_without_ext%_*}"      # 去掉_及之后: Brazil
    echo "$country"
}


for file in data/opinion_sampled_200/sampled/*.xlsx; do
    [[ -f "$file" ]] || continue  # 确保是普通文件
    export DATA_PATH=$file
    export COUNTRY=$(extract_country "$file")
    export OUTPUT_PATH="output/$LLM_MODEL/$BASELINE"
    echo "Running simulation for $COUNTRY-$BASELINE"
    python scripts/run_baseline_zs.py \
      --data_path "$file" \
      --country "$COUNTRY" \
      --baseline $BASELINE \
      --base_url $BASE_URL \
      --model $LLM_MODEL \
      --api_key $API_KEY \
      --output_dir baseline_outputs/gpt-4o-mini/PEW
done
