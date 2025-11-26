# agents/constraints.py
import json
import os
from typing import Dict, Any

from ..llm_helper._llm_stub import llm_call
from .meta_process import MetaProcess
from ..utils import load_prompt, parse_messy_json, parse_messy_json_with_fallback


class ConstraintMetaProcess(MetaProcess):
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any],
                constraints: Dict[str, Any], include_metadata: bool = False, **extra) -> (str, str):

        prev_output_dict = json.loads(extra.get("prev_output", "{}"))
        prev_predict = prev_output_dict.get("conclusion", "")
        prev_explain = prev_output_dict.get("explanation", "")
        constraint_cfg = extra.get("global_constraint_config", {})  # From global
        snippet_paths = extra.get("snippet_paths", {})

        # === LOAD FLAGS ===
        use_eq = constraint_cfg.get("use_eq", False)
        use_iq = constraint_cfg.get("use_iq", False)
        use_ses = constraint_cfg.get("use_ses", False)

        # === LOAD SNIPPETS (or empty) ===
        eq_snippet = load_prompt(snippet_paths.get("eq", "")) if use_eq else ""
        iq_snippet = load_prompt(snippet_paths.get("iq", "")) if use_iq else ""
        ses_snippet = load_prompt(snippet_paths.get("ses", "")) if use_ses else ""

        # === INJECT VALUES (only if active) ===
        active_constraints = ""
        if use_eq:
            eq_snippet = (eq_snippet
                .replace("{eq_level}", extra.get("eq_level", ""))
            )
            active_constraints += eq_snippet

        if use_iq:
            iq_snippet = (iq_snippet
                .replace("{iq_level}", extra.get("iq_level", ""))
            )
            active_constraints += iq_snippet

        if use_ses:
            ses_snippet = (ses_snippet
                .replace("{ses_level}" ,extra.get("ses_level", ""))
               )
            active_constraints += ses_snippet

        # === LOAD MASTER PROMPT ===
        master_path = extra.get("prompt_path")

        master_prompt = (load_prompt(master_path)
            .replace("{prev_result}", prev_predict)
            .replace("{prev_explanation}", prev_explain)
            .replace("{active_constraints}", active_constraints.strip())
            .replace("{chara_summary}", extra.get("chara_summary", ""))
            .replace("{stress_level}", extra.get("stress_level", ""))
            .replace("{question}", question)
        )

        raw_output = llm_call(master_prompt, llm_client=extra.get('llm_client', None))
        parsed = parse_messy_json_with_fallback(raw_output, master_prompt)
        return master_prompt, json.dumps(parsed)