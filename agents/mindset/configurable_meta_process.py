# New file: agents/configurable_meta_process.py
import importlib
import json
import os
import hashlib
import logging
from typing import Dict, Any
from .meta_process import MetaProcess
from ..utils import parse_messy_json

log = logging.getLogger(__name__)

class ConfigurableMetaProcess(MetaProcess):
    def __init__(self, stage_config: Dict):
        self.stage_config = stage_config
        self.name = stage_config["name"]
        self.include_metadata_default = stage_config.get("include_metadata", False)
        self.prompt_paths = stage_config.get("prompt_paths", {})
        self.cache_enabled = stage_config.get("cache_enabled", True)
        self.question_dependent = stage_config.get("question_dependent", True)
        self.global_constraint_config = stage_config.get("constraint_config", {})

    def _get_cache_key(self, question: str, options: str, personality_profile: Dict, constraints: Dict, include_metadata: bool, **extra) -> str:
        """Build deterministic cache key."""
        key = {
            "question": question if self.question_dependent else None,
            "options": options if self.question_dependent else None,
            "personality_profile": personality_profile,
            "constraints": constraints,
            "include_metadata": include_metadata,
            "demographics": extra.get("demographics", {}),
            "prev_output": extra.get("prev_output", {})
        }
        # Remove None values
        key = {k: v for k, v in key.items() if v is not None}
        return hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def execute(self, question: str, options: str, personality_profile: Dict[str, Any],
                constraints: Dict[str, Any], include_metadata: bool = None, **extra) -> str:
        include_metadata = include_metadata if include_metadata is not None else self.include_metadata_default
        cache_dir = extra.get("cache_dir", "cache")
        interview_id = extra.get("interview_id", "unknown")
        q_id = extra.get("q_id", "global") if not self.question_dependent else extra.get("q_id", "unknown")

        # === LOG: Stage Start ===
        log.info(f"[STAGE {self.name}] → START (id={interview_id}, q={q_id})")
        # End log

        # === CACHE CHECK ===
        if self.cache_enabled:
            os.makedirs(cache_dir, exist_ok=True)
            cache_key = self._get_cache_key(question, options, personality_profile, constraints, include_metadata, **extra)
            cache_file = os.path.join(cache_dir, f"{self.name}_id{interview_id}_q{q_id}_{cache_key}.json")

            if os.path.exists(cache_file):
                # === LOG: Cache Hit ===
                if os.path.exists(cache_file):
                    log.info(f"[STAGE {self.name}] → CACHE HIT → {cache_file}")
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        clean_output = f.read()
                    log.info(f"[STAGE {self.name}] → END (cached)")
                    extra["prev_output"] = clean_output
                    # Inject parsed dict for next stage
                    return clean_output

            # === LOG: Cache Miss → LLM ===
            log.info(f"[STAGE {self.name}] → CACHE MISS → Calling LLM")
            # End log
        else:
            # === LOG: Cache Disabled → LLM ===
            log.info(f"[STAGE {self.name}] → CACHE DISABLED → Calling LLM")
            # End log

        # === DYNAMIC STAGE EXECUTION ===
        module_path, class_name = self.stage_config["class"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        actual_class = getattr(module, class_name)
        instance = actual_class()

        # Inject prompt path
        prompt_path = self.prompt_paths.get("metadata" if include_metadata else "default")
        snippet_paths = self.prompt_paths.get("snippet_paths", None)
        if prompt_path:
            if snippet_paths is not None:
                extra = {**extra, "prompt_path": prompt_path,
                         "snippet_paths": self.prompt_paths.get("snippet_paths"),
                         "global_constraint_config": self.global_constraint_config
                }
            else:
                extra = {**extra, "prompt_path": prompt_path}


        # === LOG: Running Real Stage ===
        log.debug(f"[STAGE {self.name}] → Executing {actual_class.__name__}")
        # End log

        raw_output = instance.execute(question, options, personality_profile, constraints, include_metadata, **extra)

        parsed = parse_messy_json(raw_output, {"error": "Error parsing raw output"})
        clean_output = json.dumps(parsed, ensure_ascii=False)

        # === CACHE CLEAN JSON ===
        if self.cache_enabled:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(clean_output)
            log.info(f"[STAGE {self.name}] → CACHED → {cache_file}")

        # === INJECT PARSED DICT INTO EXTRA FOR NEXT STAGE ===
        extra["prev_output"] = clean_output

        log.info(f"[STAGE {self.name}] → END (LLM)")
        return clean_output  # Return clean JSON string