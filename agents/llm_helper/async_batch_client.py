# llm_client/async_batch_client.py
import os
import json
import time
import tempfile
import requests
import logging
from typing import List, Dict, Optional
from pathlib import Path

log = logging.getLogger(__name__)

class AsyncBatchClient:
    """
    Unified asynchronous batch client for ZhipuAI, OpenAI, and Anthropic.
    Supports JSONL upload, job creation, polling, download, and reordering by custom_id.
    """
    def __init__(self, provider: str, model: str, api_key: str = None, base_url: Optional[str] = None):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        if not self.api_key:
            raise ValueError(f"{provider.upper()}_API_KEY not set")
        
        self.base_url = base_url or self._get_default_base_url(provider)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        self._adapter = self._get_adapter(provider)
        log.info(f"{provider} async batch client initialized (model: {model})")

    def _get_default_base_url(self, provider: str) -> str:
        if provider == "openai":
            return "https://api.openai.com/v1"
        elif provider == "anthropic":
            return "https://api.anthropic.com/v1"
        elif provider == "zhipuai":
            return "https://open.bigmodel.cn/api/paas/v4"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _get_adapter(self, provider: str):
        """Provider-specific adapter for endpoints and payloads."""
        if provider == "openai":
            return {
                "upload_endpoint": "/files",
                "create_endpoint": "/batches",
                "status_endpoint": lambda batch_id: f"/batches/{batch_id}",
                "download_endpoint": lambda file_id: f"/files/{file_id}/content",
                "upload_params": {"purpose": "batch"},
                "create_payload": lambda file_id: {
                    "input_file_id": file_id,
                    "endpoint": "/chat/completions",
                    "completion_window": "24h"
                },
                "status_complete": "completed",
                "response_path": lambda data: data["choices"][0]["message"]["content"],
                "custom_id_path": lambda data: data["custom_id"]
            }
        elif provider == "anthropic":
            return {
                "upload_endpoint": "/files",  # Same as OpenAI
                "create_endpoint": "/messages/batches",
                "status_endpoint": lambda batch_id: f"/messages/batches/{batch_id}",
                "download_endpoint": lambda file_id: f"/files/{file_id}/content",
                "upload_params": {"purpose": "batch"},
                "create_payload": lambda file_id: {
                    "requests": [  # Anthropic expects "requests" array with file reference
                        {"custom_id": "batch_file", "input_file_id": file_id}
                    ],
                    "endpoint": "/messages"
                },
                "status_complete": "completed",
                "response_path": lambda data: data["response"]["body"]["choices"][0]["message"]["content"],
                "custom_id_path": lambda data: data["custom_id"]
            }
        elif provider == "zhipuai":
            return {
                "upload_endpoint": "/v1/files",
                "create_endpoint": "/v1/batch",
                "status_endpoint": lambda batch_id: f"/v1/batch/{batch_id}",
                "download_endpoint": lambda file_id: f"/v1/files/{file_id}/content",
                "upload_params": {"purpose": "batch"},
                "create_payload": lambda file_id: {
                    "input_file_id": file_id,
                    "endpoint": "/chat/completions",
                    "completion_window": "24h"
                },
                "status_complete": "completed",
                "response_path": lambda data: data["response"]["choices"][0]["message"]["content"],
                "custom_id_path": lambda data: data["custom_id"]
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def submit_and_wait(
        self,
        batch_messages: List[List[Dict[str, str]]],
        custom_ids: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        chunk_size: int = 40000,  # Safe under limits (OpenAI/Anthropic ~50k, Anthropic 10k)
        poll_interval: int = 60,
        max_polls: int = 300,  # ~5 hours
        **extra
    ) -> List[str]:
        if not batch_messages:
            return []

        # Generate custom_ids if not provided
        if custom_ids is None:
            custom_ids = [f"0000-{i:04d}" for i in range(len(batch_messages))]
        elif len(custom_ids) != len(batch_messages):
            raise ValueError("custom_ids length must match batch_messages")

        all_results = [""] * len(batch_messages)

        # Split into chunks
        for start_idx in range(0, len(batch_messages), chunk_size):
            end_idx = start_idx + chunk_size
            chunk_messages = batch_messages[start_idx:end_idx]
            chunk_custom_ids = custom_ids[start_idx:end_idx]

            log.info(f"Submitting chunk {start_idx // chunk_size + 1}: {len(chunk_messages)} requests")

            # Create JSONL (provider-specific format)
            jsonl_lines = self._adapter["jsonl_creator"](chunk_messages, chunk_custom_ids, self.model, temperature, max_tokens, **extra)

            # Temp JSONL
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
                f.write("\n".join(jsonl_lines))
                jsonl_path = f.name

            try:
                # Upload
                with open(jsonl_path, "rb") as f:
                    upload_resp = self.session.post(
                        f"{self.base_url}{self._adapter['upload_endpoint']}",
                        files={"file": f},
                        data=self._adapter["upload_params"]
                    )
                upload_resp.raise_for_status()
                file_id = upload_resp.json()["id"]
                log.info(f"Uploaded file_id={file_id}")

                # Create job
                batch_payload = self._adapter["create_payload"](file_id)
                batch_resp = self.session.post(
                    f"{self.base_url}{self._adapter['create_endpoint']}",
                    json=batch_payload
                )
                batch_resp.raise_for_status()
                batch_id = batch_resp.json()["id"]
                log.info(f"Created batch_id={batch_id}")

                # Poll
                for attempt in range(max_polls):
                    status_resp = self.session.get(
                        f"{self.base_url}{self._adapter['status_endpoint'](batch_id)}"
                    )
                    status_resp.raise_for_status()
                    status_data = status_resp.json()
                    status = status_data.get("status", "unknown")

                    log.info(f"Batch {batch_id} status: {status} (attempt {attempt + 1}/{max_polls})")

                    if status == self._adapter["status_complete"]:
                        break
                    elif status in ["failed", "cancelled", "expired"]:
                        raise RuntimeError(f"Batch {batch_id} failed: {status}. Response: {status_data}")
                    time.sleep(poll_interval)

                if status != self._adapter["status_complete"]:
                    raise TimeoutError(f"Batch {batch_id} timed out")

                # Download
                result_resp = self.session.get(
                    f"{self.base_url}{self._adapter['download_endpoint'](batch_id)}"  # Or get output_file_id first
                )
                result_resp.raise_for_status()
                result_file_id = result_resp.json()["output_file_id"] if "output_file_id" in result_resp.json() else batch_id

                download_resp = self.session.get(
                    f"{self.base_url}{self._adapter['download_endpoint'](result_file_id)}"
                )
                download_resp.raise_for_status()
                result_lines = download_resp.text.strip().split("\n")

                # Parse and reorder
                for line in result_lines:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    cid = self._adapter["custom_id_path"](data)
                    response_text = self._adapter["response_path"](data)

                    try:
                        orig_idx = custom_ids.index(cid)
                        all_results[orig_idx] = response_text
                    except ValueError:
                        log.warning(f"Unknown custom_id {cid}")

            finally:
                if os.path.exists(jsonl_path):
                    os.unlink(jsonl_path)

        log.info(f"All chunks complete: {len(all_results)} results")
        return all_results