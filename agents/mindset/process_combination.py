# agents/process_combination.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import yaml
import importlib
import logging
from .meta_process import MetaProcess
from .configurable_meta_process import ConfigurableMetaProcess  # ← Import wrapper
from ..llm_helper.constant import log_level

log = logging.getLogger(__name__)
log.setLevel(log_level)

class ProcessCombination(ABC):
    """
    A full reasoning pipeline for a personality theory.
    """

    def __init__(self, stages: List[MetaProcess]):
        self.stages = stages

    @abstractmethod
    def combine(
        self,
        question: str,
        options: str,
        personality_profile: Dict[str, Any],
        constraints: Dict[str, Any],
        **extra: Any,
    ) -> str:
        """Return final answer."""
        pass

    # ================================
    # CONFIGURATION & FACTORY
    # ================================

    @classmethod
    def build_from_config(
        cls,
        config_path: str,
        global_include_metadata: bool = False,
        stage_overrides: Optional[Dict[str, Any]] = None,
        cache_dir: str = "cache"
    ) -> 'ProcessCombination':
        """
        Factory: Load YAML → instantiate ConfigurableMetaProcess → return pipeline.

        Args:
            config_path: Path to stages YAML
            global_include_metadata: Default for stages without override
            stage_overrides: e.g. {"reason": {"include_metadata": True}}
            cache_dir: Global cache directory (passed to every stage)

        Returns:
            A concrete ProcessCombination (e.g. MBTICombination)
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # === EXTRACT GLOBAL CONSTRAINT CONFIG ===
        global_constraint_config = config.get("constraint_config", {})

        stages_config = config.get("stages", [])
        if not stages_config:
            raise ValueError("No stages defined in config")

        # Resolve concrete combination class
        combo_class_name = config.get("combination_class", "agents.MBTI.MBTICombination")
        module_path, class_name = combo_class_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        combo_class = getattr(module, class_name)

        # Build stage instances (using ConfigurableMetaProcess wrapper)
        stage_instances = []
        for cfg in stages_config:
            if not cfg.get("enabled", True):
                log.info(f"Skipping disabled stage: {cfg.get('name', 'unknown')}")
                continue

            # Resolve real class
            class_path = cfg["class"]
            mod_path, cls_name = class_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            real_cls = getattr(mod, cls_name)

            # Merge metadata
            stage_metadata = cfg.get("include_metadata", global_include_metadata)
            if stage_overrides and cfg["name"] in stage_overrides:
                override = stage_overrides[cfg["name"]]
                if "include_metadata" in override:
                    stage_metadata = override["include_metadata"]

            # Full stage config for wrapper
            wrapper_config = {
                "name": cfg["name"],
                "class": class_path,  # Pass full path to wrapper
                "include_metadata": stage_metadata,
                "prompt_paths": cfg.get("prompt_paths", {}),
                "extra": cfg.get("extra", {}),
                "cache_enabled": cfg.get("cache_enabled", True),
                "question_dependent": cfg.get("question_dependent", True),
                "cache_dir": cache_dir,
                "constraint_config": global_constraint_config
            }

            # Wrap with ConfigurableMetaProcess
            wrapper = ConfigurableMetaProcess(wrapper_config)
            stage_instances.append(wrapper)

        log.info(f"Pipeline built with {len(stage_instances)} stages from {config_path}")
        return combo_class(stage_instances)