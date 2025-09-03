"""
Option Generation Crew Module

Contains the OptionGenerationCrew class for generating candidate control options.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_core.language_models import LLM
from typing import Any, Dict


@CrewBase
class OptionGenerationCrew:
    """Crew that designs candidate control options for a given layer."""

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    @agent
    def option_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['option_designer'],
            verbose=True,
            llm=self.llm,
        )

    @task
    def generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['generation_task'],
            agent=self.option_designer(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    # Public helper
    def generate(self, layer_number: int, planned_controls, number_of_options: int, campaign_id: str | None = None) -> Dict[str, Any]:
        """Run the crew once and return the generated control options plus metadata."""

        inputs = {
            "layer_number": layer_number,
            "planned_controls": planned_controls,
            "number_of_options": number_of_options,
            "campaign_id": campaign_id,
        }

        output = self.crew().kickoff(inputs=inputs)
        raw_text = str(output.raw) if hasattr(output, "raw") else str(output)

        import json as _json

        try:
            control_options = _json.loads(raw_text)
        except Exception:
            # Import locally to avoid circular imports
            from ..utils.json_fixer_crew import JsonFixerCrew
            try:
                fixed = JsonFixerCrew(llm=self.llm).fix(raw_message=raw_text)
                control_options = _json.loads(fixed)
            except Exception:
                control_options = []

        return {
            "control_options": control_options,
        }