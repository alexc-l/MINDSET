import os
from typing import List, Dict, Any
import requests
import torch

from agents.llm_helper.chat_templates import apply_chat_template

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
import logging

try:
    from vllm import LLM, SamplingParams  # NEW: For local vLLM
except ImportError:
    LLM = SamplingParams = None

try:
    from transformers import pipeline as hf_pipeline  # NEW: HuggingFace pipeline
except ImportError:
    hf_pipeline = None

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

        if self.provider == 'vllm':
            if LLM is None:
                raise ImportError("Install vLLM: pip install vllm")
            if self.base_url:
                # Server mode: OpenAI client
                self._is_server = True
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                log.info(f"vLLM server client at {self.base_url}")
            else:
                # Local mode: vLLM engine
                self._is_server = False
                self.client = LLM(
                    model=self.model,
                    tensor_parallel_size=kwargs.get('tp', 1),  # e.g., 4 for multi-GPU
                    quantization=kwargs.get('quantization', 'nvfp4'),  # Blackwell opt
                    max_model_len=kwargs.get('max_model_len', 4096),
                    **kwargs
                )
                log.info("vLLM local engine initialized")
        elif self.provider == 'lmdeploy':
            if lmdeploy is None:
                raise ImportError("Install lmdeploy: pip install lmdeploy")
            if self.base_url:  # Server mode
                self._is_server = True
                self.session = requests.Session()
                self.session.headers.update({"Content-Type": "application/json"})
                if self.api_key:
                    self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            else:  # Local fallback
                self._is_server = False
        elif self.provider == 'huggingface':
            if hf_pipeline is None:
                raise ImportError("Install transformers: pip install transformers")
            self.client = hf_pipeline(
                "text-generation",
                model=self.model,
                device=kwargs.get('device', 0 if torch.cuda.is_available() else -1),
                **kwargs
            )
            self._is_server = False  # Local only

        elif self.provider == 'ollama':
            # Ollama: Assume local server running at base_url (default http://localhost:11434)
            if self.base_url is None:
                self.base_url = "http://localhost:11434"
            self._is_server = True
            self.session = requests.Session()
            if self.api_key:
                self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **extra
    ) -> str:
        """Single chat completion call."""
        prompt = self._format_prompt(messages)

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

        if self.provider == 'lmdeploy':
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
        if self.provider == 'vllm':
            if self._is_server:
                # Server: OpenAI API
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra
                )
                # exit()
                return resp.choices[0].message.content
            else:
                # Local: vLLM engine
                prompts = [self._format_prompt(m) for m in messages]  # Simple chat format
                sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, **extra)
                outputs = self.client.generate(prompts, sampling_params)
                return outputs[0].outputs[0].text

        if self.provider == 'huggingface':
            outputs = self.client(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1,
                **extra
            )
            generated = outputs[0]['generated_text'][len(prompt):].strip()
            return generated

        if self.provider == 'ollama':
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False,
                **extra
            }
            resp = self.session.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["response"].strip()

        raise NotImplementedError(f"Chat completion not implemented for {self.provider}")

    def batch_chat_completion(
        self,
        batch_messages: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **extra
    ) -> List[str]:
        """Batch reasoning for efficiency, especially with LMDeploy."""
        prompts = [self._format_prompt(msgs) for msgs in batch_messages]

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
        if self.provider == 'vllm':
            if self._is_server:
                # Server: Batch via OpenAI API (array of messages)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=batch_messages,  # [[msg1], [msg2], ...]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra
                )
                return [choice.message.content for choice in resp.choices]
            else:
                # Local: vLLM batch generation
                prompts = [[self._format_prompt(m) for m in msgs] for msgs in batch_messages]
                sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, **extra)
                outputs = self.client.generate(prompts, sampling_params)
                return [out.outputs[0].text for out in outputs]
        if self.provider == 'huggingface':
            outputs = self.client(
                prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1,
                **extra
            )
            return [out[0]['generated_text'][len(prompt):].strip() for out, prompt in zip(outputs, prompts)]

        if self.provider == 'ollama':
            # Ollama: Sequential fallback (no native batch)
            results = []
            for prompt in prompts:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False,
                    **extra
                }
                resp = self.session.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                results.append(resp.json()["response"].strip())
            return results
        else:
            # Fallback sequential
            return [self.chat_completion(msgs, temperature, max_tokens, **extra) for msgs in batch_messages]

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Model-aware prompt formatting.
        """
        return apply_chat_template(
            messages=messages,
            model_name=self.model,
            add_generation_prompt=True,
        )