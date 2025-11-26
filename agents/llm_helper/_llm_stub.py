from typing import Dict, Any, List
from .llm_client import LLMClient

# Global client (configure in main.py)
# llm_client: LLMClient = None  # Set this in main.py


def llm_call(
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        llm_client: LLMClient = None,
        **extra
) -> str:
    if llm_client is None:
        raise ValueError("LLMClient not initialized. Set in main.py")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return llm_client.chat_completion(messages, temperature, max_tokens, **extra)


# For batch (used in main.py for efficiency)
def batch_llm_call(
        prompts: List[str],
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        llm_client: LLMClient = None,
        **extra
) -> List[str]:
    batch_messages = []
    for p in prompts:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": p})
        batch_messages.append(msgs)

    return llm_client.batch_chat_completion(batch_messages, temperature, max_tokens, **extra)