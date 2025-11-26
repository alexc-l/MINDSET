# utils/chat_templates.py
import json
from pathlib import Path
from typing import List, Dict, Any

# ------------------------------------------------------------------
# 1. Tiny built-in registry (no external deps)
# ------------------------------------------------------------------
# Format: {model_name_or_prefix: jinja2 template string}
# Add more as needed – these cover 99% of current models
BUILTIN_TEMPLATES = {
    # Llama-3 / Llama-3.1 / Llama-3.2
    "llama-3": """<|begin_of_text|>{% for message in messages %}{% if message.role == "system" %}<|start_header_id|>system<|end_header_id|>

{{ message.content }}<|eot_id|>{% elif message.role == "user" %}<|start_header_id|>user<|end_header_id|>

{{ message.content }}<|eot_id|>{% elif message.role == "assistant" %}<|start_header_id|>assistant<|end_header_id|>

{{ message.content }}<|eot_id|>{% endif %}{% endfor %}<|start_header_id|>assistant<|end_header_id|>
""",
    # Qwen2 / Qwen2.5
    "qwen": """<|im_start|>system
{{ messages[0].content }}<|im_end|>
{% for msg in messages[1:] %}{% if msg.role == "user" %}<|im_start|>user
{{ msg.content }}<|im_end|>
{% elif msg.role == "assistant" %}<|im_start|>assistant
{{ msg.content }}<|im_end|>
{% endif %}{% endfor %}<|im_start|>assistant
""",
    # Mistral / Mixtral / Zephyr
    "mistral": """<s>{% for message in messages %}{% if message.role == "system" %}{{ message.content }}{% elif message.role == "user" %}{{ "<|user|>\n" + message.content + "<|end|>\n<|assistant|>" }}{% elif message.role == "assistant" %}{{ message.content + "</s>" }}{% endif %}{% endfor %}""",
    # Gemma-2
    "gemma": """<start_of_turn>user
{{ messages[0].content }}<end_of_turn>
{% for msg in messages[1:] %}{% if msg.role == "user" %}<start_of_turn>user
{{ msg.content }}<end_of_turn>
{% elif msg.role == "assistant" %}<start_of_turn>model
{{ msg.content }}<end_of_turn>
{% endif %}{% endfor %}<start_of_turn>model
""",
    # Phi-3
    "phi-3": """<|system|>{{ messages[0].content }}<|end|>
{% for msg in messages[1:] %}{% if msg.role == "user" %}<|user|>{{ msg.content }}<|end|>
{% elif msg.role == "assistant" %}<|assistant|>{{ msg.content }}<|end|>
{% endif %}{% endfor %}<|assistant|>
""",
    # Generic OpenAI-style (gpt-4, claude, etc.)
    "openai": """{% for message in messages %}{{ message.role }}: {{ message.content }}{% if not loop.last %}

{% endif %}{% endfor %}assistant: """
}

# ------------------------------------------------------------------
# 2. Optional: fallback to transformers (if installed)
# ------------------------------------------------------------------
try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except Exception:  # ImportError or any issue
    _HAS_TRANSFORMERS = False
    AutoTokenizer = None

# ------------------------------------------------------------------
# 3. Public function
# ------------------------------------------------------------------
def apply_chat_template(
    messages: List[Dict[str, str]],
    model_name: str,
    add_generation_prompt: bool = True,
) -> str:
    """
    Returns a correctly formatted prompt for the given model.
    Order of precedence:
      1. Exact match in BUILTIN_TEMPLATES
      2. Prefix match (e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct" → "llama-3")
      3. HuggingFace transformers (if installed)
      4. Fallback to simple OpenAI style
    """
    model_key = model_name.lower()

    # 1. Exact match
    if model_key in BUILTIN_TEMPLATES:
        template_str = BUILTIN_TEMPLATES[model_key]
    else:
        # 2. Prefix match
        template_str = next(
            (tmpl for key, tmpl in BUILTIN_TEMPLATES.items() if model_key.startswith(key)),
            None,
        )

        # 3. Transformers fallback
        if template_str is None and _HAS_TRANSFORMERS and AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True, use_fast=True
                )
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception as e:
                print(f"[chat_template] transformers failed for {model_name}: {e}")

        # 4. Ultimate fallback
        if template_str is None:
            template_str = BUILTIN_TEMPLATES["openai"]

    # Render Jinja2 template
    from jinja2 import Template
    tmpl = Template(template_str)
    return tmpl.render(messages=messages)