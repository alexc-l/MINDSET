export CUDA_VISIBLE_DEVICES=0,1,2,3

python scripts/reasoning_eval.py baseline_outputs/qwen3-8b-baselines/hwang_2023 \
    --sample-size 200 \
    --model cache/glm-z1 \
    --batch-size 128 \
    --output eval/hwang_2023