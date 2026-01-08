# agents/bigfive.py
import json
from typing import Dict, Any, List

from .llm_helper.constant import log_level
from .mindset.meta_process import MetaProcess
from .mindset.process_combination import ProcessCombination
from .mindset.personality_theory import PersonalityTheory
from .utils import load_prompt, parse_messy_json, parse_messy_json_with_fallback
from .constants import BIGFIVE_DESC_LOOKUP  # Assume similar to PROCESS_DESC_LOOKUP for Big Five traits
from .llm_helper._llm_stub import llm_call  # For convenience
import os
import hashlib
import logging

log = logging.getLogger(__name__)
log.setLevel(log_level)

class BigFiveSelectMetaProcess(MetaProcess):
    """Selects Big Five trait levels based on demo, profile, rules (LLM)."""
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any],
                constraints: Dict[str, Any], include_metadata: bool = False, **extra) -> (str, str):
        chara_summary = extra.get('chara_summary', '')  # From previous
        rule_bigfive = personality_profile.get('rule_bigfive', {})
        rule_probs = personality_profile.get('rule_probs', {})
        prompt_path = extra.get('prompt_path')  # Injected
        prompt = (load_prompt(prompt_path)
            .replace("{chara_summary}", chara_summary)
            .replace("{rule_bigfive}", json.dumps(rule_bigfive))
            .replace("{rule_probs}", json.dumps(rule_probs))
        )
        if "batch_size" in extra.keys():
            return prompt, None

        return prompt, llm_call(prompt, llm_client=extra.get('llm_client', None))

class GetTraitVectorMetaProcess(MetaProcess):
    """Python: Gets full trait vector (levels/scores) from Big Five prediction."""
    _TRAIT_LEVELS = {
        'openness': {'low': 'Closed-minded, practical', 'medium': 'Balanced curiosity', 'high': 'Creative, imaginative'},
        'conscientiousness': {'low': 'Disorganized, impulsive', 'medium': 'Reliable', 'high': 'Organized, dutiful'},
        'extraversion': {'low': 'Introverted, reserved', 'medium': 'Moderate energy', 'high': 'Outgoing, energetic'},
        'agreeableness': {'low': 'Competitive, skeptical', 'medium': 'Cooperative', 'high': 'Compassionate, trusting'},
        'neuroticism': {'low': 'Stable, calm', 'medium': 'Average resilience', 'high': 'Anxious, sensitive'}
    }

    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> (str, str):
        prev_output_dict = json.loads(extra.get('prev_output', '{}'))  # From state, e.g. {"openness": "high", ...}
        bigfive_type = prev_output_dict.get('bigfive', {})
        if not bigfive_type:
            raise ValueError("Big Five traits not available")
        
        trait_vector = {}
        for trait, level in bigfive_type.items():
            if trait in self._TRAIT_LEVELS and level in self._TRAIT_LEVELS[trait]:
                trait_vector[trait] = {
                    "level": level,
                    "description": self._TRAIT_LEVELS[trait][level]
                }
            else:
                log.warning(f"Unknown trait or level: {trait}={level}")

        return "", json.dumps(trait_vector)

class AssignImpactMetaProcess(MetaProcess):
    """Evaluates impact of societal/situational stress on each trait."""
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> (str, str):
        trait_vector = extra.get('prev_output', '{}')
        stress_level = extra.get('stress_level', 'medium')
        chara_summary = extra.get('chara_summary', '')
        prompt_path = extra.get('prompt_path')  # Injected
        prompt = (load_prompt(prompt_path)
            .replace("{chara_summary}", chara_summary)
            .replace("{stress_level}", stress_level)
            .replace("{question}", question)
            .replace("{trait_vector}", trait_vector)
        )
        if "batch_size" in extra.keys():
            return prompt, None
        else:
            output = llm_call(prompt, llm_client=extra.get('llm_client', None))
            output_dict = parse_messy_json_with_fallback(output, prompt)
            return prompt, json.dumps(output_dict)

class ReasonMetaProcess(MetaProcess):
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> (str, str):
        impacted_traits = extra.get('prev_output', '{}')
        chara_summary = extra.get('chara_summary', '')
        stress_level = extra.get('stress_level', 'medium')
        prompt_path = extra.get('prompt_path')  # Injected
        # Enrich with descriptions (similar to MBTI)
        try:
            trait_data = json.loads(impacted_traits)
        except json.JSONDecodeError:
            trait_data = []

        enriched_traits = []
        for item in trait_data:
            trait_name = item.get("trait")
            impact = item.get("stress_impact", "positive")
            level = item.get("level")
            if level == "medium":
                # Medium never goes into "grip" — force adaptive mode
                final_desc = "Shows balanced, moderate expression of this trait with no strong bias."
            else:
                lookup_key = f"{trait_name}"
                # print(lookup_key)
                desc_entry = BIGFIVE_DESC_LOOKUP.get(lookup_key)  # falls back gracefully
                # print(BIGFIVE_DESC_LOOKUP.keys())
                if impact == "negative":
                    final_desc = desc_entry.get("Stressed_desc", desc_entry["Description"])
                else:
                    final_desc = desc_entry["Description"]
            enriched_item = {**item, "trait_description": final_desc}
            enriched_traits.append(enriched_item)

        enriched_traits_json = json.dumps(enriched_traits)

        prompt = (load_prompt(prompt_path)
            .replace("{chara_summary}", chara_summary)
            .replace("{stress_level}", stress_level)
            .replace("{question}", question)
            .replace("{options}", options)
            .replace("{impacted_traits}", enriched_traits_json)
        )
        if "batch_size" in extra.keys():
            return prompt, None

        return prompt, llm_call(prompt, llm_client=extra.get('llm_client', None))

class SynthesisMetaProcess(MetaProcess):
    def execute(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                include_metadata: bool = False, **extra) -> (str, str):
        reasoning_results = extra.get('prev_output', '[]')
        chara_summary = extra.get('chara_summary', '')
        prompt_path = extra.get('prompt_path')  # Injected
        prompt = (load_prompt(prompt_path)
            .replace("{chara_summary}", chara_summary)
            .replace("{question}", question)
            .replace("{options}", options)
            .replace("{reasoning_results}", reasoning_results)
        )
        if "batch_size" in extra.keys():
            return prompt, None

        return prompt, llm_call(prompt, llm_client=extra.get('llm_client', None))

class BigFiveCombination(ProcessCombination):
    def combine(self, question: str, options: str, personality_profile: Dict[str, Any], constraints: Dict[str, Any],
                **extra) -> str:
        state = {'demographics': extra.get('demographics', {})}
        stress_level = None
        bigfive = None

        for i, stage in enumerate(self.stages):
            stage_config = getattr(stage, 'stage_config', {})
            stage_name = stage_config.get("name", "unknown")
            stage_extra = {**extra, **state}
            request, output = stage.execute(question, options, personality_profile, constraints,
                                   **stage_extra)
            state['prev_output'] = output

            # Capture from StressChara (stage 0)
            if stage_name == "stress_chara":
                try:
                    combined_json = json.loads(output)
                    stress_level = combined_json.get('stress_level', 'medium')
                    state['stress_level'] = stress_level
                    state['chara_summary'] = combined_json.get('chara_summary', '')
                except json.JSONDecodeError:
                    pass

            # Capture from BigFiveSelect (stage 1)
            if stage_name == "bigfive_select":
                try:
                    select_json = json.loads(output)
                    bigfive = select_json.get('bigfive', {})
                    state['bigfive'] = bigfive
                    personality_profile['bigfive'] = bigfive  # Update profile if needed
                except json.JSONDecodeError:
                    pass

            if stage_name == "assign_impact":
                try:
                    output_dict = parse_messy_json_with_fallback(output, request)
                    for trait_dict in output_dict:
                        level = trait_dict.get("level", "medium")
                        trait_dict["level"] = level
                    state['prev_output'] = json.dumps(output_dict)
                except json.JSONDecodeError:
                    pass

        return state['prev_output']

class BigFiveTheory(PersonalityTheory):
    def build_process_combination(self) -> ProcessCombination:
        return BigFiveCombination([])

    def predict_from_demographics(self, demographics: Dict[str, Any]) -> Dict[str, Any]:
        # Rule-based suggestions for Big Five traits
        age = demographics.get('age', 30)
        gender = demographics.get('gender', 'male')
        bigfive_probs = {
            'openness': {'low': 0.3, 'medium': 0.4, 'high': 0.3},
            'conscientiousness': {'low': 0.3, 'medium': 0.4, 'high': 0.3},
            'extraversion': {'low': 0.3, 'medium': 0.4, 'high': 0.3},
            'agreeableness': {'low': 0.3, 'medium': 0.4, 'high': 0.3},
            'neuroticism': {'low': 0.3, 'medium': 0.4, 'high': 0.3},
        }
        if age > 30:
            bigfive_probs['conscientiousness']['high'] += 0.2
            bigfive_probs['neuroticism']['low'] += 0.1
        if gender == 'male':
            bigfive_probs['agreeableness']['low'] += 0.1
            bigfive_probs['neuroticism']['high'] -= 0.1
        # Normalize per trait
        for trait in bigfive_probs:
            total = sum(bigfive_probs[trait].values())
            for level in bigfive_probs[trait]:
                bigfive_probs[trait][level] /= total

        rule_bigfive = {trait: max(levels, key=levels.get) for trait, levels in bigfive_probs.items()}
        return {'rule_bigfive': rule_bigfive, 'rule_probs': bigfive_probs}