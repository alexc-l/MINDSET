import json
from typing import Dict, Any

from ..llm_helper._llm_stub import llm_call
from .meta_process import MetaProcess
from ..utils import load_prompt


class StressCharaMetaProcess(MetaProcess):
    """MARK 1+2: Generates socio-demographic profile with stress (no MBTI)."""
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> (str, str):
        demographics = extra.get('demographics')
        stage_config = getattr(self, 'stage_config', {})
        constraint_cfg = extra.get('global_constraint_config', {}) # From global
        # === DYNAMIC SNIPPETS ===
        active_snippets = ""
        snippet_paths = extra.get("snippet_paths", {})

        if constraint_cfg.get("use_eq", False):
            active_snippets += load_prompt(snippet_paths.get("eq", "")) + "\n"
        if constraint_cfg.get("use_iq", False):
            active_snippets += load_prompt(snippet_paths.get("iq", "")) + "\n"
        if constraint_cfg.get("use_ses", False):
            active_snippets += load_prompt(snippet_paths.get("ses", "")) + "\n"

        # === MASTER PROMPT ===
        prompt_path = extra.get('prompt_path')  # Injected

        prompt = (load_prompt(prompt_path)
            .replace("{active_snippets}", active_snippets.strip())
            .replace("{demographics}", json.dumps(demographics, ensure_ascii=False))
        )

        assert demographics is not None and prompt_path is not None, f"demographics and prompt path is None!"
        return prompt, llm_call(prompt, llm_client=extra.get('llm_client', None))