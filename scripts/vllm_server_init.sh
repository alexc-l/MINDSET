#export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

vllm serve cache/Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu_memory_utilization 0.85 \
  --enable-chunked-prefill  \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 16384 \
  --max-num-batched-tokens 32768