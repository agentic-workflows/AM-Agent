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
            from manufacturing_agent.crew import JsonFixerCrew  # local import to avoid circular refs
            try:
                fixed = JsonFixerCrew(llm=self.llm).fix(raw_message=raw_text)
                control_options = _json.loads(fixed)
            except Exception:
                control_options = []

        return {
            "control_options": control_options,
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
    def decide(self, layer_number: int, control_options, planned_controls, scores) -> Dict[str, Any]:
        """Run the crew once and return the chosen option index & reasoning."""

        inputs = {
            "layer_number": layer_number,
            "control_options": control_options,
            "planned_controls": planned_controls,
            "scores": scores,
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
            n_opts = len(scores)
            if n_opts and (best_option < 0 or best_option >= n_opts):
                best_option = int(min(range(n_opts), key=scores.__getitem__))
                reasoning += " (adjusted to valid lowest-score option)"
        except Exception:
            from manufacturing_agent.crew import JsonFixerCrew
            try:
                fixed = JsonFixerCrew(llm=self.llm).fix(raw_message=raw_text)
                data = _json.loads(fixed)
                best_option = int(data["best_option"])
                reasoning = data.get("reasoning", "")
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



# JSON Fixer Crew: validates and repairs JSON strings returned by other LLMs

@CrewBase
class JsonFixerCrew:
    """Crew that extracts and fixes JSON from a raw LLM message."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    @agent
    def json_fixer(self) -> Agent:
        return Agent(
            config=self.agents_config.get('json_fixer', {
                'role': 'JSON Extractor',
                'goal': 'Return valid JSON',
                'backstory': ''
            }),
            verbose=False,
            llm=self.llm,
        )

    @task
    def fixer_task(self) -> Task:
        prompt_template = (
            "You are a JSON extractor and fixer.\n"
            "You are given a raw message that may include explanations, markdown fences, or partial JSON.\n"
            "Your task:\n"
            "  1. Check if the message contains a JSON object or array.\n"
            "  2. If it does, extract and fix the JSON if needed.\n"
            "  3. Ensure all keys and string values are properly quoted.\n"
            "  4. Return only valid, parseable JSON — no markdown, no explanations.\n\n"
            "THE OUTPUT MUST BE A VALID JSON ONLY. DO NOT SAY ANYTHING ELSE.\n\n"
            "Raw message:\n"
            "{raw_message}\n"
        )
        return Task(
            description=prompt_template,
            agent=self.json_fixer(),
        )

    @crew
    def crew(self) -> Crew:
        """Define the execution crew for the JSON fixer."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )

    def fix(self, raw_message: str) -> str:
        """Return a fixed JSON string."""
        output = self.crew().kickoff(inputs={"raw_message": raw_message})
        return str(output.raw) if hasattr(output, "raw") else str(output)