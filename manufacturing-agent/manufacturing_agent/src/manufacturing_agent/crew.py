from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_core.language_models import LLM
from typing import Any, Dict
import time

import json

# Azure AI Foundry Research Service 
class ResearchService:
    """Azure AI Foundry Deep Research service for manufacturing insights"""
    
    def __init__(self, foundry_config):
        self.foundry_config = foundry_config
        self.foundry_client = None
        
        # Initialize Azure AI Foundry client
        if foundry_config and foundry_config.get('enabled', False):
            try:
                self._initialize_foundry_client()
                print("Azure AI Foundry Deep Research initialized (West US)")
            except Exception as e:
                print(f"Azure AI Foundry initialization failed: {e}")
                raise
        else:
            print("Azure AI Foundry not configured - Deep Research unavailable")
            raise ValueError("Azure AI Foundry configuration required for Deep Research")
    
    def _initialize_foundry_client(self):
        """Initialize Azure AI Foundry client"""
        try:
            from azure.ai.foundry.agents import Agent
            from azure.ai.foundry.agents.tools import DeepResearchTool
            
            self.foundry_client = Agent(
                model="o3-deep-research",
                tools=[DeepResearchTool()],
                endpoint=self.foundry_config['endpoint'],
                api_key=self.foundry_config['api_key']
            )
        except ImportError:
            print("Azure AI Foundry SDK not installed. Install: pip install azure-ai-foundry[agents]")
            raise
        except Exception as e:
            print(f"Azure AI Foundry client initialization failed: {e}")
            raise
    

    
    def get_research_background(self, layer_number: int, control_options, planned_controls):
        """Azure AI Foundry Deep Research with web search and citations"""
        
        print(f"Conducting Azure AI Foundry Deep Research for layer {layer_number}...")
        print("   - Using o3-deep-research model in West US")
        print("   - Live web search with Bing integration")
        print("   - Real citations and sources")
        
        # Extract parameter ranges
        if control_options:
            powers = [opt.get('power', 0) for opt in control_options]
            dwell_0s = [opt.get('dwell_0', 0) for opt in control_options] 
            dwell_1s = [opt.get('dwell_1', 0) for opt in control_options]
            power_range = f"{min(powers)}-{max(powers)}W"
            dwell0_range = f"{min(dwell_0s)}-{max(dwell_0s)}ms"
            dwell1_range = f"{min(dwell_1s)}-{max(dwell_1s)}ms"
        else:
            power_range = dwell0_range = dwell1_range = "N/A"
        
        research_query = f"""
        Research latest developments in additive manufacturing control parameters for layer {layer_number} powder bed fusion processes.
        
        Parameter Context:
        - Layer number: {layer_number}
        - Laser power range: {power_range}
        - Dwell_0 time range: {dwell0_range}
        - Dwell_1 time range: {dwell1_range}
        
        Research Focus:
        1. Find recent (2023-2025) research papers on laser power optimization for powder bed fusion
        2. Look for latest dwell time studies and thermal management advances  
        3. Search for new parameter interaction findings and optimization strategies
        4. Find current industry best practices and case studies
        5. Identify cutting-edge quality control methods and monitoring approaches
        
        Provide specific recommendations with citations from current literature and web sources.
        Include publication dates, DOI links, and author information where available.
        Focus on actionable insights for manufacturing engineers.
        """
        
        try:
            # Call Azure AI Foundry Deep Research
            response = self.foundry_client.complete(research_query)
            
            # Extract essential results only
            research_result = {
                "research_findings": response.content,
                "citations": getattr(response, 'citations', []),
                "web_sources": getattr(response, 'sources', []),
                "layer_number": layer_number,
                "timestamp": time.time(),
                "parameter_context": {
                    "power_range": power_range,
                    "dwell_0_range": dwell0_range,
                    "dwell_1_range": dwell1_range
                }
            }
            
            print(f"Azure AI Foundry Deep Research completed for layer {layer_number}")
            print(f"   - Research content: {len(response.content)} chars")
            print(f"   - Web citations: {len(getattr(response, 'citations', []))}")
            print(f"   - Web sources: {len(getattr(response, 'sources', []))}")
            
            # Print the research results
            print("=" * 80)
            print(f"AZURE AI FOUNDRY DEEP RESEARCH RESULTS FOR LAYER {layer_number}")
            print("Deep Research with Live Web Search & Citations")
            print("=" * 80)
            print(response.content)
            print("=" * 80)
            print("END OF AZURE AI FOUNDRY DEEP RESEARCH")
            print("=" * 80)
            
            return research_result
            
        except Exception as e:
            print(f"Azure AI Foundry Deep Research failed for layer {layer_number}: {e}")
            # Raise error - research is mandatory
            raise Exception(f"Mandatory Azure AI Foundry Deep Research failed for layer {layer_number}: {str(e)}")


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
    
    # Public helper - requires research context
    def decide(self, layer_number: int, control_options, planned_controls, scores, research_context: Dict) -> Dict[str, Any]:
        """Run the crew with mandatory research context and return the chosen option index & reasoning."""

        inputs = {
            "layer_number": layer_number,
            "control_options": control_options,
            "planned_controls": planned_controls,
            "scores": scores,
            "research_context": research_context,
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

    def decide_with_feedback(self, layer_number: int, control_options, planned_controls, scores, research_context: Dict, validation_feedback: str = None) -> Dict[str, Any]:
        """Run the crew with additional feedback from safety validation."""
        
        # Create a modified task description that includes the validation feedback
        base_inputs = {
            "layer_number": layer_number,
            "control_options": control_options,
            "planned_controls": planned_controls,
            "scores": scores,
        }
        
        if validation_feedback:
            # Create an enhanced task that includes the previous feedback
            enhanced_description = f"""
            ROLE: Control Decision Agent
            TASK: Choose the best option index for layer {layer_number} using the provided simulation scores = {scores} and control_options = {control_options}.
            
            PREVIOUS VALIDATION FEEDBACK: {validation_feedback}
            IMPORTANT: The previous decision was rejected for the above reasons. Please carefully review the scores and ensure you select the option with the LOWEST score value.
            
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
            return self.decide(layer_number, control_options, planned_controls, scores, research_context)
        
        raw_text = str(output.raw) if hasattr(output, "raw") else str(output)

        import json as _json

        try:
            data = _json.loads(raw_text)
            best_option = int(data["best_option"])
            reasoning = data.get("reasoning", "")
            # Post-validation: ensure the index is within the valid range 
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
                best_option = int(min(range(len(scores)), key=scores.__getitem__))
                reasoning = f"Fallback due to parse error: {exc}"

        return {
            "best_option": best_option,
            "reasoning": reasoning,
            "raw_text": raw_text,
        }

    def decide_with_research_context(self, layer_number: int, control_options, planned_controls, scores, research_context: Dict = None) -> Dict[str, Any]:
        """Make decision with research context - DOES NOT modify original prompts"""
        
        # First get the normal decision using existing logic (unchanged)
        base_decision = self.decide(layer_number, control_options, planned_controls, scores)
        
        # If research context available, enhance the reasoning explanation
        if research_context and research_context.get("success", False):
            # Extract key insights from research (first 300 chars)
            research_summary = research_context["research_findings"][:300]
            if len(research_context["research_findings"]) > 300:
                research_summary += "..."
            
            # Enhance reasoning with research context
            enhanced_reasoning = f"{base_decision['reasoning']}\n\nResearch Context: {research_summary}"
            
            return {
                "best_option": base_decision["best_option"],  # Same decision
                "reasoning": enhanced_reasoning,               # Enhanced explanation
                "raw_text": base_decision["raw_text"],
                "research_applied": True,
                "research_citations": len(research_context.get("citations", []))
            }
        else:
            # No research available or research failed - return original decision
            return {
                **base_decision,
                "research_applied": False,
                "research_citations": 0
            }


# Simple session caches (per campaign) to avoid re-initializing the crew for each request
_DECISION_CACHE: dict[str, DecisionCrew] = {}
_GEN_CACHE: dict[str, OptionGenerationCrew] = {}
_SAFETY_CACHE: dict[str, 'SafetyValidationCrew'] = {}


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


def get_safety_crew(campaign_id: str | None, llm: LLM | None):
    key = campaign_id or "default"
    if key not in _SAFETY_CACHE:
        _SAFETY_CACHE[key] = SafetyValidationCrew(llm=llm)
    return _SAFETY_CACHE[key]


@CrewBase
class SafetyValidationCrew:
    """Crew that validates decisions made by the decision_maker agent."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

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
    def validate(self, layer_number: int, decision_result: Dict[str, Any], control_options, scores) -> Dict[str, Any]:
        """Run the crew once and return validation results."""

        max_option_index = len(control_options) - 1 if control_options else 0

        inputs = {
            "layer_number": layer_number,
            "decision_result": decision_result,
            "control_options": control_options,
            "scores": scores,
            "max_option_index": max_option_index,
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
            from manufacturing_agent.crew import JsonFixerCrew
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