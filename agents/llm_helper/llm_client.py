import os
from typing import List, Dict, Any
import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None

try:
    import lmdeploy
    from lmdeploy import pipeline, ChatTemplateConfig
except ImportError:
    lmdeploy = None

log = logging.getLogger(__name__)

import logging

class LLMClient:
    """
    Unified client for LLM calls across providers.
    Supports closed-source: OpenAI, Anthropic, Google (Gemini), ZhipuAI.
    Supports open-source via LMDeploy for acceleration and batching.
    """
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = None,
        base_url: str = None,
        **kwargs  # e.g., for LMDeploy: engine='turbomind', tp=1 for acceleration
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv(f"{self.provider.upper()}_API_KEY")
        self.base_url = base_url
        self.kwargs = kwargs

        if self.provider == 'openai':
            if OpenAI is None:
                raise ImportError("Install openai: pip install openai")
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        elif self.provider == 'anthropic':
            if Anthropic is None:
                raise ImportError("Install anthropic: pip install anthropic")
            self.client = Anthropic(api_key=self.api_key)

        elif self.provider == 'google':
            if genai is None:
                raise ImportError("Install google-generativeai: pip install google-generativeai")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)

        elif self.provider == 'zhipuai':
            if ZhipuAI is None:
                raise ImportError("Install zhipuai: pip install zhipuai")
            self.client = ZhipuAI(api_key=self.api_key)

        elif self.provider == 'lmdeploy':
            if lmdeploy is None:
                raise ImportError("Install lmdeploy: pip install lmdeploy")
            if self.base_url:
                # Server mode: Use HTTP client
                self._is_server = True
                self.session = requests.Session()
                if self.api_key:
                    self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
                log.info(f"LMDeploy server client initialized at {self.base_url}")
            else:
                # Local mode: Use pipeline
                self._is_server = False
                chat_config = ChatTemplateConfig(model_name=self.model) if 'chat_template' in kwargs else None
                self.client = pipeline(
                    self.model,
                    backend=kwargs.get('backend', 'turbomind'),
                    tp=kwargs.get('tp', 1),
                    model_format=kwargs.get('model_format', 'hf'),
                    chat_template_config=chat_config
                )
            log.info("LMDeploy local pipeline initialized")

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **extra
    ) -> str:
        """Single chat completion call."""
        if self.provider == 'openai':
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra
            )
            return resp.choices[0].message.content

        elif self.provider == 'anthropic':
            # Anthropic uses 'system' separate; adjust messages
            system = next((m['content'] for m in messages if m['role'] == 'system'), None)
            user_messages = [m for m in messages if m['role'] != 'system']
            resp = self.client.messages.create(
                model=self.model,
                messages=user_messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra
            )
            return resp.content[0].text

        elif self.provider == 'google':
            # Simplified: Concat messages into content
            content = '\n'.join([f"{m['role']}: {m['content']}" for m in messages])
            config = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
            resp = self.client.generate_content(content, generation_config=config, **extra)
            return resp.text

        elif self.provider == 'zhipuai':
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra
            )
            return resp.choices[0].message.content

        elif self.provider == 'lmdeploy':
            if self._is_server:
                # Server: OpenAI-compatible HTTP
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **extra
                }
                resp = self.session.post(f"{self.base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                # Local: Pipeline
                responses = self.client.chat(messages, temperature=temperature, max_new_tokens=max_tokens, **extra)
                return responses[0].response.text

        raise NotImplementedError(f"Chat completion not implemented for {self.provider}")

    def batch_chat_completion(
        self,
        batch_messages: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **extra
    ) -> List[str]:
        """Batch reasoning for efficiency, especially with LMDeploy."""
        if self.provider == 'lmdeploy':
            if self._is_server:
                # Server: Batch via single request (OpenAI format: list of messages arrays)
                payload = {
                    "model": self.model,
                    "messages": batch_messages,  # Array of message arrays
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **extra
                }
                resp = self.session.post(f"{self.base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return [choice["message"]["content"] for choice in data["choices"]]
            else:
                # Local: Pipeline batch
                responses = self.client.batch_chat(batch_messages, temperature=temperature, max_new_tokens=max_tokens, **extra)
                return [r.response.text for r in responses]            
        else:
            # Fallback sequential
            return [self.chat_completion(msgs, temperature, max_tokens, **extra) for msgs in batch_messages]
