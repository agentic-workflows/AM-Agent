"""
Crew Cache Module

Contains caching functions for crew instances to avoid re-initialization.
"""

from langchain_core.language_models import LLM
from typing import Dict

# Import crew classes
from ..crews.decision_crew import DecisionCrew
from ..crews.option_generation_crew import OptionGenerationCrew  
from ..crews.safety_validation_crew import SafetyValidationCrew
from ..crews.citation_classification_crew import CitationClassificationCrew

# Simple session caches (per campaign) to avoid re-initializing the crew for each request
_DECISION_CACHE: Dict[str, DecisionCrew] = {}
_GEN_CACHE: Dict[str, OptionGenerationCrew] = {}
_SAFETY_CACHE: Dict[str, SafetyValidationCrew] = {}
_CITATION_CACHE: Dict[str, CitationClassificationCrew] = {}


def get_decision_crew(campaign_id: str | None, llm: LLM | None) -> DecisionCrew:
    """Get or create a cached DecisionCrew instance."""
    key = campaign_id or "default"
    if key not in _DECISION_CACHE:
        _DECISION_CACHE[key] = DecisionCrew(llm=llm)
    return _DECISION_CACHE[key]


def get_generation_crew(campaign_id: str | None, llm: LLM | None) -> OptionGenerationCrew:
    """Get or create a cached OptionGenerationCrew instance."""
    key = campaign_id or "default"
    if key not in _GEN_CACHE:
        _GEN_CACHE[key] = OptionGenerationCrew(llm=llm)
    return _GEN_CACHE[key]


def get_safety_crew(campaign_id: str | None, llm: LLM | None) -> SafetyValidationCrew:
    """Get or create a cached SafetyValidationCrew instance."""
    key = campaign_id or "default"
    if key not in _SAFETY_CACHE:
        _SAFETY_CACHE[key] = SafetyValidationCrew(llm=llm)
    return _SAFETY_CACHE[key]


def get_citation_crew(campaign_id: str | None, llm: LLM | None) -> CitationClassificationCrew:
    """Get or create a cached CitationClassificationCrew instance."""
    key = campaign_id or "default"
    if key not in _CITATION_CACHE:
        _CITATION_CACHE[key] = CitationClassificationCrew(llm=llm)
    return _CITATION_CACHE[key]


def clear_all_caches() -> None:
    """Clear all crew caches. Useful for testing or memory management."""
    global _DECISION_CACHE, _GEN_CACHE, _SAFETY_CACHE, _CITATION_CACHE
    _DECISION_CACHE.clear()
    _GEN_CACHE.clear()
    _SAFETY_CACHE.clear()
    _CITATION_CACHE.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about cached crew instances."""
    return {
        "decision_crews": len(_DECISION_CACHE),
        "generation_crews": len(_GEN_CACHE),
        "safety_crews": len(_SAFETY_CACHE),
        "citation_crews": len(_CITATION_CACHE),
        "total": len(_DECISION_CACHE) + len(_GEN_CACHE) + len(_SAFETY_CACHE) + len(_CITATION_CACHE)
    }