import hashlib
import importlib
import json
import logging
import os
from typing import Dict, Any, List

import yaml

from .contants import PROCESS_DESC_LOOKUP
from .mindset.meta_process import MetaProcess
from .mindset.personality_theory import PersonalityTheory
from .mindset.process_combination import ProcessCombination

from .utils import load_prompt, parse_messy_json, load_stage_config
from .llm_helper._llm_stub import llm_call  # For convenience
log = logging.getLogger(__name__)


class StressCharaMetaProcess(MetaProcess):
    """MARK 1+2: Generates socio-demographic profile with stress (no MBTI)."""
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> str:
        demographics = extra.get('demographics')
        prompt_path = extra.get('prompt_path')  # Injected

        assert demographics is not None and prompt_path is not None, f"demographics and prompt path is None!"

        prompt = (load_prompt(prompt_path)
            .replace("{demographics}", json.dumps(demographics, ensure_ascii=False))
        )

        return llm_call(prompt, llm_client=extra.get('llm_client', None))

class MBTISelectMetaProcess(MetaProcess):
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any],
                constraints: Dict[str, Any], include_metadata: bool = False, **extra) -> str:
        prev_output_dict = json.loads(extra.get("prev_output"))
        chara_summary = prev_output_dict.get('chara_summary', None)
        rule_mbti = personality_profile.get('rule_mbti', None)
        rule_probs = personality_profile.get('rule_probs', None)
        prompt_path = extra.get('prompt_path')  # Injected

        assert chara_summary is not None and rule_mbti is not None and rule_probs is not None and prompt_path is not None, f"chara_summary, rule_mbti, rule_probs and prompt path is None!"

        prompt = (load_prompt(prompt_path)
            .replace("{rule_mbti}", rule_mbti)
            .replace("{rule_probs}", json.dumps(rule_probs, ensure_ascii=False))
            .replace("{chara_summary}", chara_summary)
        )
        return llm_call(prompt, llm_client=extra.get('llm_client', None))

class GetStackMetaProcess(MetaProcess):
    """Python: Gets full function stack from MBTI type."""
    _STACKS = {
        'INFJ': {'dominant': 'Ni', 'auxiliary': 'Fe', 'tertiary': 'Ti', 'inferior': 'Se'},
        'INTJ': {'dominant': 'Ni', 'auxiliary': 'Te', 'tertiary': 'Fi', 'inferior': 'Se'},
        'INFP': {'dominant': 'Fi', 'auxiliary': 'Ne', 'tertiary': 'Si', 'inferior': 'Te'},
        'INTP': {'dominant': 'Ti', 'auxiliary': 'Ne', 'tertiary': 'Si', 'inferior': 'Fe'},
        'ENFJ': {'dominant': 'Fe', 'auxiliary': 'Ni', 'tertiary': 'Se', 'inferior': 'Ti'},
        'ENTJ': {'dominant': 'Te', 'auxiliary': 'Ni', 'tertiary': 'Se', 'inferior': 'Fi'},
        'ENFP': {'dominant': 'Ne', 'auxiliary': 'Fi', 'tertiary': 'Te', 'inferior': 'Si'},
        'ENTP': {'dominant': 'Ne', 'auxiliary': 'Ti', 'tertiary': 'Fe', 'inferior': 'Si'},
        'ISFJ': {'dominant': 'Si', 'auxiliary': 'Fe', 'tertiary': 'Ti', 'inferior': 'Ne'},
        'ISTJ': {'dominant': 'Si', 'auxiliary': 'Te', 'tertiary': 'Fi', 'inferior': 'Ne'},
        'ISFP': {'dominant': 'Fi', 'auxiliary': 'Se', 'tertiary': 'Ni', 'inferior': 'Te'},
        'ISTP': {'dominant': 'Ti', 'auxiliary': 'Se', 'tertiary': 'Ni', 'inferior': 'Fe'},
        'ESFJ': {'dominant': 'Fe', 'auxiliary': 'Si', 'tertiary': 'Ne', 'inferior': 'Ti'},
        'ESTJ': {'dominant': 'Te', 'auxiliary': 'Si', 'tertiary': 'Ne', 'inferior': 'Fi'},
        'ESFP': {'dominant': 'Se', 'auxiliary': 'Fi', 'tertiary': 'Te', 'inferior': 'Ni'},
        'ESTP': {'dominant': 'Se', 'auxiliary': 'Ti', 'tertiary': 'Fe', 'inferior': 'Ni'},
        # Expand if needed
    }

    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> str:
        prev_output_dict = json.loads(extra.get("prev_output", "{}"))
        mbti_type = prev_output_dict.get('mbti', '')
        if not mbti_type:
            raise ValueError("MBTI type not available")
        stack = self._STACKS.get(mbti_type.upper(), {})
        if not stack:
            raise ValueError(f"Unknown MBTI type: {mbti_type}")
        return json.dumps(stack)

class AssignImpactMetaProcess(MetaProcess):
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> str:
        stack = extra.get('prev_output', '{}')
        stress_level = extra.get('stress_level', 'medium')
        chara_summary = extra.get('chara_summary', '')
        prompt_path = extra.get('prompt_path')  # Injected
        prompt = (load_prompt(prompt_path)
            .replace("{chara_summary}", chara_summary)
            .replace("{stress_level}", stress_level)
            .replace("{stack}", stack)
        )

        return llm_call(prompt, llm_client=extra.get('llm_client', None))

class ReasonMetaProcess(MetaProcess):
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> str:
        impacted_stack = extra.get('prev_output', '{}')
        chara_summary = extra.get('chara_summary', '')
        stress_level = extra.get('stress_level', 'medium')
        prompt_path = extra.get('prompt_path')  # Injected
        # Enrich stack with descriptions (unchanged)
        try:
            stack_data = json.loads(impacted_stack)
        except json.JSONDecodeError:
            stack_data = []

        enriched_stack = []
        for item in stack_data:
            proc_name = item.get("process")
            impact = item.get("stress_impact", "positive")
            desc_entry = PROCESS_DESC_LOOKUP.get(proc_name, {})
            desc = desc_entry.get("Description", "No description available.")
            neg_desc = desc_entry.get("Neg_desc", "No negative description.")
            final_desc = neg_desc if impact == "negative" else desc
            enriched_item = {**item, "process_description": final_desc}
            enriched_stack.append(enriched_item)

        enriched_stack_json = json.dumps(enriched_stack)

        prompt = (load_prompt(prompt_path)
            .replace("{chara_summary}", chara_summary)
            .replace("{stress_level}", stress_level)
            .replace("{question}", question)
            .replace("{options}", options)
            .replace("{stack}", enriched_stack_json)
        )

        return llm_call(prompt, llm_client=extra.get('llm_client', None))

class SynthesisMetaProcess(MetaProcess):
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> str:
        reasoning_results = extra.get('prev_output', '[]')
        chara_summary = extra.get('chara_summary', '')
        prompt_path = extra.get('prompt_path')  # Injected
        prompt = (load_prompt(prompt_path)
            .replace("{chara_summary}", chara_summary)
            .replace("{question}", question)
            .replace("{options}", options)
            .replace("{reasoning_results}", reasoning_results)
        )
        return llm_call(prompt, llm_client=extra.get('llm_client', None))

class MBTICombination(ProcessCombination):
    def combine(self, question: str, options: str, personality_profile: Dict[str, Any],
                constraints: Dict[str, Any], **extra) -> str:

        state = {
            "demographics": extra.get("demographics", {}),
            "cache_dir": extra.get("cache_dir", "cache"),
            "interview_id": extra.get("interview_id", "unknown"),
            "q_id": extra.get("q_id", "unknown"),
        }

        log.info(f"Starting pipeline for ID={state['interview_id']}, Q={state['q_id']}")

        for stage in self.stages:
            stage_config = getattr(stage, '_stage_config', {})
            stage_name = stage_config.get("name", "unknown")

            stage_extra = {**extra, **state}
            output = stage.execute(
                question, options, personality_profile, constraints,
                **stage_extra
            )
            state["prev_output"] = output

            # === CAPTURE BY NAME (not index) ===
            if stage_name == "stress_chara":
                try:
                    data = json.loads(output)
                    state["stress_level"] = data.get("stress_level")
                    state["chara_summary"] = data.get("chara_summary")
                    log.debug(f"[CAPTURE] stress_chara → stress_level={state['stress_level']}")
                except json.JSONDecodeError:
                    log.warning(f"[CAPTURE] Failed to parse stress_chara output")

            elif stage_name == "mbti_select":
                try:
                    data = json.loads(output)
                    mbti = data.get("mbti")
                    if mbti:
                        state["mbti"] = mbti
                        personality_profile["mbti"] = mbti
                        log.debug(f"[CAPTURE] mbti_select → mbti={mbti}")
                except json.JSONDecodeError:
                    log.warning(f"[CAPTURE] Failed to parse mbti_select output")

            # Add more as needed:
            # elif stage_name == "some_other_stage": ...

        log.info(f"Pipeline complete. Final answer length: {len(state['prev_output'])}")
        return state["prev_output"]

class MBTITheory(PersonalityTheory):
    def build_process_combination(self) -> ProcessCombination:
        return MBTICombination([])

    def predict_from_demographics(self, demographics: Dict[str, Any]) -> Dict[str, Any]:
        # Only rule-based (probs and suggested type); LLM in stages
        age = demographics.get('age', 30)
        gender = demographics.get('gender', 'male')
        mbti_probs = {'I': 0.5, 'E': 0.5, 'S': 0.5, 'N': 0.5, 'T': 0.5, 'F': 0.5, 'J': 0.5, 'P': 0.5}
        if age > 30:
            mbti_probs['I'] += 0.3
            mbti_probs['P'] += 0.2
        if gender == 'male':
            mbti_probs['T'] += 0.3
        # Normalize
        for k in mbti_probs:
            mbti_probs[k] = min(1.0, max(0.0, mbti_probs[k]))
        letters = [('I','E'), ('S','N'), ('T','F'), ('J','P')]
        rule_mbti = ''.join(max(pair, key=lambda k: mbti_probs[k]) for pair in letters)
        return {'rule_mbti': rule_mbti, 'rule_probs': mbti_probs}