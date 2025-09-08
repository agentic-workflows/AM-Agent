"""
Safety Validation Crew Module

Contains the SafetyValidationCrew class for validating manufacturing decisions.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_core.language_models import LLM
from typing import Any, Dict


@CrewBase
class SafetyValidationCrew:
    """Crew that validates decisions made by the decision_maker agent."""

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    @agent
    def safety_validator(self) -> Agent:
        return Agent(
            config=self.agents_config['safety_validator'],
            verbose=True,
            llm=self.llm,
        )

    @task
    def safety_validation_task(self) -> Task:
        return Task(
            config=self.tasks_config['safety_validation_task'],
            agent=self.safety_validator(),
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
    def validate(self, layer_number: int, decision_result: Dict[str, Any], control_options, scores, combined_user_guidance: str = "") -> Dict[str, Any]:
        """Run the crew once and return validation results."""

        max_option_index = len(control_options) - 1 if control_options else 0

        inputs = {
            "layer_number": layer_number,
            "decision_result": decision_result,
            "control_options": control_options,
            "scores": scores,
            "max_option_index": max_option_index,
            "combined_user_guidance": combined_user_guidance,
        }

        output = self.crew().kickoff(inputs=inputs)
        raw_text = str(output.raw) if hasattr(output, "raw") else str(output)

        import json as _json

        try:
            validation_data = _json.loads(raw_text)
            is_valid = bool(validation_data.get("is_valid", False))
            feedback = validation_data.get("feedback", "")
            requires_regeneration = bool(validation_data.get("requires_regeneration", False))
        except Exception:
            # Import locally to avoid circular imports
            from ..utils.json_fixer_crew import JsonFixerCrew
            try:
                fixed = JsonFixerCrew(llm=self.llm).fix(raw_message=raw_text)
                validation_data = _json.loads(fixed)
                is_valid = bool(validation_data.get("is_valid", False))
                feedback = validation_data.get("feedback", "")
                requires_regeneration = bool(validation_data.get("requires_regeneration", False))
            except Exception as exc:
                # Fallback validation logic
                is_valid = False
                feedback = f"Validation failed due to parse error: {exc}"
                requires_regeneration = True

        return {
            "is_valid": is_valid,
            "feedback": feedback,
            "requires_regeneration": requires_regeneration,
            "raw_text": raw_text,
        }