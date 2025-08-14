"""
Manufacturing Agent Crew Module

Main module that provides backward compatibility and the ManufacturingAgentCrew class.
This module imports and re-exports all the restructured components.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_core.language_models import LLM

# Import all restructured components
from .services import ResearchService, ResearchSummarizer
from .crews import (
    OptionGenerationCrew,
    DecisionCrew,
    SafetyValidationCrew,
    CitationClassificationCrew
)
from .utils import (
    JsonFixerCrew,
    get_decision_crew,
    get_generation_crew,
    get_safety_crew,
    get_citation_crew,
    clear_all_caches,
    get_cache_stats
)

# Re-export everything for backward compatibility
__all__ = [
    # Services
    'ResearchService',
    'ResearchSummarizer',
    
    # Crews
    'OptionGenerationCrew',
    'DecisionCrew',
    'SafetyValidationCrew', 
    'CitationClassificationCrew',
    'ManufacturingAgentCrew',
    
    # Utils
    'JsonFixerCrew',
    'get_decision_crew',
    'get_generation_crew',
    'get_safety_crew',
    'get_citation_crew',
    'clear_all_caches',
    'get_cache_stats'
]


# -----------------------------------------------
# ManufacturingAgentCrew (UNUSED - for main.py compatibility only)
# -----------------------------------------------
@CrewBase
class ManufacturingAgentCrew:
    """
    UNUSED CLASS - Only exists to prevent import errors in main.py
    The actual manufacturing agent logic is in aec_agent_mock.py
    """
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    @agent
    def placeholder_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['option_designer'],
            llm=self.llm
        )

    @task
    def placeholder_task(self) -> Task:
        return Task(
            config=self.tasks_config['generation_task'],
            agent=self.placeholder_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
