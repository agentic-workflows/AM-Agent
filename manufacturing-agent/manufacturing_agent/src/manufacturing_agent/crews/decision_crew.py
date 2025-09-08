"""
Decision Crew Module

Contains the DecisionCrew class for selecting the best control option.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_core.language_models import LLM
from typing import Any, Dict


@CrewBase
class DecisionCrew:
    """Crew that selects the best control option given simulation scores."""

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

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
    
    # Public helper - independent decision making
    def decide(self, layer_number: int, control_options, planned_controls, scores, user_message: str = "", combined_user_guidance: str = "", history_json=None) -> Dict[str, Any]:
        """Run the crew independently and return the chosen option index & reasoning."""

        inputs = {
            "layer_number": layer_number,
            "control_options": control_options,
            "planned_controls": planned_controls,
            "scores": scores,
            "user_message": user_message,
            "combined_user_guidance": combined_user_guidance,
            "history_json": history_json or [],
        }

        output = self.crew().kickoff(inputs=inputs)
        raw_text = str(output.raw) if hasattr(output, "raw") else str(output)

        import json as _json

        try:
            data = _json.loads(raw_text)
            best_option = int(data["best_option"])
            reasoning = data.get("reasoning", "")
            # Post-validation: ensure the index is within the valid range
            n_opts = len(scores)
            if n_opts and (best_option < 0 or best_option >= n_opts):
                raise ValueError("Best option index out of range")
        except Exception:
            # Import locally to avoid circular imports
            from ..utils.json_fixer_crew import JsonFixerCrew
            try:
                fixed = JsonFixerCrew(llm=self.llm).fix(raw_message=raw_text)
                data = _json.loads(fixed)
                best_option = int(data["best_option"])
                reasoning = data.get("reasoning", "")
            except Exception as exc:
                raise ValueError(f"Decision parsing failed: {exc}")

        return {
            "best_option": best_option,
            "reasoning": reasoning,
            "raw_text": raw_text,
        }

    def decide_with_feedback(self, layer_number: int, control_options, planned_controls, scores, validation_feedback: str = None, user_message: str = "", combined_user_guidance: str = "", history_json=None) -> Dict[str, Any]:
        """Run the crew with additional feedback from safety validation."""
        
        # Create a modified task description that includes the validation feedback
        base_inputs = {
            "layer_number": layer_number,
            "control_options": control_options,
            "planned_controls": planned_controls,
            "scores": scores,
            "user_message": user_message,
            "combined_user_guidance": combined_user_guidance,
            "history_json": history_json or [],
        }
        
        if validation_feedback:
            # Create an enhanced task that includes the previous feedback
            enhanced_description = f"""
            ROLE: Control Decision Agent
            TASK: Choose the best option index for layer {layer_number} using the provided simulation scores = {scores} and control_options = {control_options}.
            
            PREVIOUS VALIDATION FEEDBACK: {validation_feedback}
            IMPORTANT: The previous decision was rejected for the above reasons. Please carefully review the scores and ensure you select the option with the LOWEST score value.
            
            USER GUIDANCE (REQUIRED): {combined_user_guidance}
            HISTORY (REQUIRED): {history_json}
            DECISION CRITERIA: You MUST incorporate the concatenated user guidance and the full prior layer history when making the decision. Simulation scores remain the primary quantitative signal, but explicit user directives in the provided guidance MUST be taken into account and can take precedence over purely score-based selection when they conflict. Use historical outcomes to avoid repeating poor choices and to favor patterns that previously worked.

            FORMAT CONSTRAINTS: Return ONLY a valid JSON object (no Markdown, no code fences) with exactly two keys:
              • "best_option": integer (0‒N-1 where N = len(control_options))
              • "reasoning":  string
            The object must contain **no additional keys** and be directly parseable by `json.loads()`. Do not prepend or append any explanatory text.
            CONSISTENCY CHECK: Ensure the chosen index is within range; if scores list length is N, best_option ∈ [0, N-1].
            ANTI-HALLUCINATION TIP: Double-check numeric comparisons; never claim a larger number is lower.
            Scoring Hint: A lower score indicates better quality. For example, in [5, 10], option 0 is preferred since 5 < 10.
            ⚠️ Caution: Do NOT hallucinate reasoning. For example, if scores = [2, 3, 5], 2 is the lowest score and should be chosen. Use correct numerical comparisons only.
            After you have chosen the best option, you should double check the scores to make sure you have chosen the correct lowest score option.
            """
            
            enhanced_task = Task(
                description=enhanced_description,
                agent=self.decision_maker(),
                expected_output="A JSON string with the keys `best_option` and `reasoning`."
            )
            
            crew = Crew(
                agents=[self.decision_maker()],
                tasks=[enhanced_task],
                process=Process.sequential,
                verbose=True,
            )
            
            output = crew.kickoff(inputs=base_inputs)
        else:
            # Use the regular method if no feedback
            return self.decide(layer_number, control_options, planned_controls, scores, user_message)
        
        raw_text = str(output.raw) if hasattr(output, "raw") else str(output)

        import json as _json

        try:
            data = _json.loads(raw_text)
            best_option = int(data["best_option"])
            reasoning = data.get("reasoning", "")
            # Post-validation: ensure the index is within the valid range 
            n_opts = len(scores)
            if n_opts and (best_option < 0 or best_option >= n_opts):
                raise ValueError("Best option index out of range")
        except Exception:
            # Import locally to avoid circular imports
            from ..utils.json_fixer_crew import JsonFixerCrew
            try:
                fixed = JsonFixerCrew(llm=self.llm).fix(raw_message=raw_text)
                data = _json.loads(fixed)
                best_option = int(data["best_option"])
                reasoning = data.get("reasoning", "")
            except Exception as exc:
                raise ValueError(f"Decision parsing failed: {exc}")

        return {
            "best_option": best_option,
            "reasoning": reasoning,
            "raw_text": raw_text,
        }