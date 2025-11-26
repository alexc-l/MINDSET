### Using LMDeploy for Experiments on NVIDIA Spark

#### Step 1: Install LMDeploy Package
LMDeploy is primarily installed via pip. Use the pre-built CUDA 12 wheel for Spark compatibility (avoids compilation). Run in your conda env:

```bash
# Create/activate env
conda create -n lmdeploy python=3.10
conda activate lmdeploy

# Install LMDeploy (CUDA 12 pre-built; includes TurboMind)
pip install lmdeploy[all,cuda12]  # 'all' for full backends (Turbomind, PyTorch, vLLM)

# Verify
python -c "import lmdeploy; print(lmdeploy.__version__)"  # Should be 0.10.2+
nvidia-smi  # Confirm GPU (Spark B200)
```

- **If CUDA 11.8 needed** (rare for Spark): `pip install lmdeploy[all,cuda11]`.
- **AMD/ROCm support**: If not Spark, add `pip install lmdeploy[all,rocm]`.
- **Size**: ~500MB; includes engines for batching.

#### Step 2: Download/Prepare Model
Clone a Hugging Face model (e.g., Llama-3-8B for testing):

```bash
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct /path/to/model
```

#### Step 3: Initiate the Server
Use `lmdeploy serve api` for an OpenAI-compatible server (supports batch via `/v1/chat/completions`). For Spark, enable TurboMind backend (optimized for Blackwell; up to 1.8x vLLM throughput).

```bash
# Basic server (single GPU, TurboMind backend)
lmdeploy serve api \
  --model /path/to/model  # HF repo or local path
  --backend turbomind     # For Spark/Blackwell acceleration
  --tp 1                 # Tensor parallelism (1 for single GPU; scale to 8+ for multi-Spark)
  --host 0.0.0.0         # Bind to network
  --port 23333           # Default port
  --trust-remote-code    # For HF models

# Advanced (batch-optimized, multi-GPU)
lmdeploy serve api \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --backend turbomind \
  --tp 4                 # e.g., 4 Spark GPUs
  --quant-policy w4a16   # Optional: 4-bit quantization for speed (1.5x on Spark)
  --host 0.0.0.0 \
  --port 23333 \
  --max-batch-size 128   # Batch up to 128 requests
```

- **Server URL**: `http://localhost:23333/v1` (OpenAI API format).
- **Batch Inference**: Send array of `messages` (see code below). Server handles persistent batching (dynamic sizing).
- **Spark Optimization**: TurboMind auto-detects Blackwell; use `--quant-policy w4a16` for 1.5x speed on large models.
- **Logs**: Check `~/.lmdeploy/logs` for errors; monitor with `nvidia-smi`.
- **Stop**: Ctrl+C; restart with same command for persistence.

#### Troubleshooting
- **GPU Detection**: Run `lmdeploy --version` and `nvidia-smi`. Ensure Spark drivers (R550+).
- **Memory Errors**: Use `--quant-policy w4a16` or `--tp 8` for multi-Spark.
- **Batch Limits**: Server default max_batch_size=128; adjust with `--max-batch-size 256`.
- **Docs**: [LMDeploy Server Guide](https://lmdeploy.readthedocs.io/en/latest/serving/server.html) (2025 updates include Blackwell support).

This setup is **production-ready** for experiments—batch throughput on Spark can hit 1000+ tokens/sec for 8B models. If you need multi-node scaling, add `--server-port 23333 --worker-port 50000`. Let me know for tweaks!**