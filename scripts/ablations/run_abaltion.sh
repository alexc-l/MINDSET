#!/bin/bash
shopt -s nullglob
MODEL="Qwen3-14B"
THEORY="bigfive"

for dir in debug_cache/Qwen3-14B/WVS/*; do
    [[ -r "$dir" ]] || continue
  result=$(basename "$dir")
  echo $result
  python scripts/ablations/extract_stage_from_cache.py \
    --cache_dir "$dir/$THEORY/" \
    --stage synthesis \
    --extract_key conclusion \
    --output_file results/$MODEL/WVS/aba/syn_"$result".csv

done