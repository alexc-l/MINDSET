import os
from typing import Dict, Any, List

import yaml
import json
import re
import logging

from openai import OpenAI

log = logging.getLogger(__name__)


def load_prompt(name: str) -> str:
    path = os.path.join("prompts", f"{name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt {path} not found")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# --- Load config ---
def load_stage_config(config_path: str = "config/stages.yaml") -> List[Dict]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get("stages", [])



def parse_messy_json(raw: str, fallback: Dict = None) -> Dict:
    """Robustly extract and parse JSON from LLM output."""
    if not raw:
        return fallback or {}

    # Remove code fences
    raw = re.sub(r'^```(?:json)?\s*|```$', '', raw.strip(), flags=re.MULTILINE)
    raw = raw.strip().strip('"\'')
    raw = re.sub(r'\s+', ' ', raw)

    # Extract JSON object/array
    match = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if not match:
        log.warning(f"No JSON found in output: {raw[:100]}...")
        return fallback or {}

    candidate = match.group(1)
    # Fix trailing commas
    candidate = re.sub(r',\s*([\]}])', r'\1', candidate)

    try:
        parsed = json.loads(candidate)
        return parsed
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse failed: {e} | Raw: {candidate[:200]}...")
        return fallback or {}

# === CONFIG: Choose your fallback strategy ===
_FALLBACK_STRATEGY = "remote_glm4_flash"   # Options: "local", "remote_glm4_flash", "remote_qwen"

# Remote fallback (ZhipuAI GLM-4-Flash)
_REMOTE_CLIENT = None
def get_remote_client():
    global _REMOTE_CLIENT
    if _REMOTE_CLIENT is None:
        _REMOTE_CLIENT = OpenAI(
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key=os.getenv("ZHIPUAI_API_KEY")
        )
    return _REMOTE_CLIENT

# Local fallback (e.g., smaller Qwen3-4B or GLM-4-Flash local)
_LOCAL_FALLBACK_CLIENT = None
def get_local_fallback_client():
    global _LOCAL_FALLBACK_CLIENT
    if _LOCAL_FALLBACK_CLIENT is None:
        _LOCAL_FALLBACK_CLIENT = OpenAI(
            base_url="http://localhost:8001/v1",  # your fallback vLLM server
            api_key="EMPTY"
        )
    return _LOCAL_FALLBACK_CLIENT

def parse_messy_json_with_fallback(
    raw: str,
    request: str,
    max_reasoning_tokens: int = 1500,
    fallback: Dict = None
) -> Dict:
    """Try normal parse → if fail → use fast fallback model (local or remote)"""
    if not raw:
        return fallback or {}

    # Quick heuristic: too long → skip normal parse
    if len(raw) > max_reasoning_tokens * 7:
        log.info(f"Output too long ({len(raw)} chars) → direct fallback")
        return _run_fallback_extraction(raw, request)

    # Normal parse attempt
    cleaned = re.sub(r'^```(?:json)?\s*|```$', '', raw.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip().strip('"\'')
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)

    match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
    if not match:
        log.warning("No JSON found → fallback")
        return _run_fallback_extraction(raw, request)

    try:
        parsed = json.loads(match.group(1))
        log.debug("Primary parse succeeded")
        return parsed
    except json.JSONDecodeError as e:
        log.warning(f"Primary parse failed: {e} → using fallback model")
        return _run_fallback_extraction(raw, request)


def _run_fallback_extraction(raw_output: str, request: str) -> Dict:
    """Extract JSON using fast fallback model (local or remote)"""
    try:

        messages = [
            {"role": "system", "content": "You are a precise JSON extractor. Extract ONLY the final JSON answer from the following verbose reasoning according to the request. No thinking."},
            {"role": "user", "content": f"Request: {request}\nRaw output:\n{raw_output}"}
        ]

        # Choose client based on strategy
        if _FALLBACK_STRATEGY == "remote_glm4_flash":
            client = get_remote_client()
            model = "glm-4-flash"
        elif _FALLBACK_STRATEGY == "local":
            client = get_local_fallback_client()
            model = "qwen3-4b"  # or your local fast model
        else:
            raise ValueError(f"Unknown fallback strategy: {_FALLBACK_STRATEGY}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"} if _FALLBACK_STRATEGY == "remote_glm4_flash" else None
        )

        result = response.choices[0].message.content.strip()
        log.info(f"Fallback extraction succeeded using {model}")

        # Final parse (should almost always work)
        return parse_messy_json(result, fallback={})

    except Exception as e:
        log.error(f"Fallback extraction completely failed: {e}")
        return {}