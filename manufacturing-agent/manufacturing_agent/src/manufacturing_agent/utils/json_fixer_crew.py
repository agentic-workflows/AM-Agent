"""
JSON Fixer Crew Module

Contains the JsonFixerCrew class for extracting and fixing JSON from LLM outputs.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_core.language_models import LLM


@CrewBase
class JsonFixerCrew:
    """Crew that extracts and fixes JSON from a raw LLM message."""

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

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