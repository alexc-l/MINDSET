import os
from typing import Dict, Any, List

import yaml
import json
import re
import logging

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
