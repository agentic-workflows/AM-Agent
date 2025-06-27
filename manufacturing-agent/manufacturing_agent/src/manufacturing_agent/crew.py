from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_core.language_models import LLM
from typing import Any, Dict


@CrewBase
class OptionGenerationCrew:
    """Crew that designs candidate control options for a given layer."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    @agentdecide
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
            control_options = []

        return {
            "control_options": control_options,
            "response": raw_text,
            "llm": self.llm,
        }


@CrewBase
class DecisionCrew:
    """Crew that selects the best control option given simulation scores."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    @agent
    def decision_maker(self) -> Agent:
        return Agent(
            config=self.agents_config['decision_maker'],
            verbose=True,
            llm=self.llm,
        )

    @task
    def decision_task(self) -> Task:
        return Task(
            config=self.tasks_config['decision_task'],
            agent=self.decision_maker(),
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
    def decide(self, layer_number: int, planned_controls, scores, campaign_id: str | None = None) -> Dict[str, Any]:
        """Run the crew once and return the chosen option index & reasoning."""

        inputs = {
            "layer_number": layer_number,
            "planned_controls": planned_controls,
            "scores": scores,
            "campaign_id": campaign_id,
        }

        output = self.crew().kickoff(inputs=inputs)
        raw_text = str(output.raw) if hasattr(output, "raw") else str(output)

        import json as _json

        try:
            data = _json.loads(raw_text)
            best_option = int(data["best_option"])
            reasoning = data.get("reasoning", "")
            # Post-validation: ensure the index is within the valid range 
            # Potentially need to rerun?
            n_opts = len(scores.get("scores", []))
            if n_opts and (best_option < 0 or best_option >= n_opts):
                best_option = int(min(range(n_opts), key=scores["scores"].__getitem__))
                reasoning += " (adjusted to valid lowest-score option)"
        except Exception as exc:
            best_option = int(min(range(len(scores["scores"])), key=scores["scores"].__getitem__))
            reasoning = f"Fallback due to parse error: {exc}"

        return {
            "best_option": best_option,
            "reasoning": reasoning,
            "raw_text": raw_text,
        }


# Simple session caches (per campaign) to avoid re-initializing the crew for each request
_DECISION_CACHE: dict[str, DecisionCrew] = {}
_GEN_CACHE: dict[str, OptionGenerationCrew] = {}


def get_decision_crew(campaign_id: str | None, llm: LLM | None):
    key = campaign_id or "default"
    if key not in _DECISION_CACHE:
        _DECISION_CACHE[key] = DecisionCrew(llm=llm)
    return _DECISION_CACHE[key]


def get_generation_crew(campaign_id: str | None, llm: LLM | None):
    key = campaign_id or "default"
    if key not in _GEN_CACHE:
        _GEN_CACHE[key] = OptionGenerationCrew(llm=llm)
    return _GEN_CACHE[key] 