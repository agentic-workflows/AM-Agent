import json
import os
import sys
from threading import Thread
from time import sleep
from typing import Dict, List

from flowcept.agents.agent_client import run_tool
from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM, agent_flowcept_task, get_current_context_task
from flowcept.configs import AGENT_HOST, AGENT_PORT
from utils import build_llm

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import uvicorn

from mcp.server.fastmcp import FastMCP
import pathlib
from flowcept.configs import AGENT
# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from aec_agent_context_manager import AdamantineAeCContextManager

try:
    from manufacturing_agent.crew import OptionGenerationCrew, DecisionCrew, SafetyValidationCrew, ResearchService
    from pathlib import Path
    from dotenv import load_dotenv
    import importlib
except ImportError as exc:
    raise ImportError(
        "The 'manufacturing_agent' package is not installed.\n"
        "Run 'pip install -e manufacturing-agent/manufacturing_agent' inside your virtual environment."
    ) from exc

# os.environ["SAMBASTUDIO_URL"] = AGENT.get("llm_server_url")
# os.environ["SAMBASTUDIO_API_KEY"] = AGENT.get("api_key")


agent_controller = AdamantineAeCContextManager()
mcp = FastMCP("AnC_Agent", require_session=True, lifespan=agent_controller.lifespan)


def get_foundry_config():
    """
    Load Azure AI Foundry configuration for deep research
    Required for the research service to function
    """
    try:
        # Check for Azure AI Foundry environment variables
        foundry_endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
        foundry_api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
        foundry_enabled = os.environ.get("AZURE_AI_FOUNDRY_ENABLED", "false").lower() == "true" 
        
        if foundry_enabled and foundry_endpoint and foundry_api_key:
            return {
                "enabled": True,
                "endpoint": foundry_endpoint,
                "api_key": foundry_api_key,
                "region": "West US",  # Required for deep research
                "model": "o3-deep-research",
                "version": "2025-06-26"
            }
        else:
            # Configuration not provided - research will be unavailable
            return {
                "enabled": False,
                "note": "Azure AI Foundry configuration not provided - Deep Research unavailable"
            }
            
    except Exception as e:
        print(f"Error loading Azure AI Foundry config: {e}")
        return {"enabled": False, "error": str(e)}

def build_llm():
    #model_name = AGENT.get("model_name", "o4-mini-2025-04-16")
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # use the correct deployment name 2024-10-21
        api_version="2024-10-21"  # stable API version
    )
    tool_task = get_current_context_task()
    wrapped_llm = FlowceptLLM(llm=llm, campaign_id=tool_task.campaign_id, parent_task_id=tool_task.task_id, workflow_id=tool_task.workflow_id, agent_id=tool_task.agent_id)
    return wrapped_llm

#################################################
# TOOLS
#################################################


@mcp.tool()
@agent_flowcept_task  # Must be in this order. @mcp.tool then @flowcept_task
def generate_options_set(layer: int, planned_controls, number_of_options=4, campaign_id=None):
    llm = build_llm()
    ctx = mcp.get_context()
    history = ctx.request_context.lifespan_context.history
    crew = OptionGenerationCrew(llm=llm)
    result = crew.generate(layer_number=layer, planned_controls=planned_controls, number_of_options=number_of_options, campaign_id=campaign_id)
    return result


@mcp.tool()
@agent_flowcept_task  # Must be in this order. @mcp.tool then @flowcept_task
def choose_option(layer: int, control_options: List[Dict], scores: List, planned_controls: List[Dict], campaign_id: str=None):
    llm = build_llm()
    ctx = mcp.get_context()
    history = ctx.request_context.lifespan_context.history
    
    decision_crew = DecisionCrew(llm=llm)
    safety_crew = SafetyValidationCrew(llm=llm)
    
    # 1. Get research background FIRST (MANDATORY - Azure AI Foundry Deep Research)
    # Load Azure AI Foundry configuration 
    foundry_config = get_foundry_config()
    
    # Initialize Azure AI Foundry research service
    research_service = ResearchService(foundry_config)
    
    # Get research background - this will raise exception if it fails
    research_context = research_service.get_research_background(
        layer_number=layer,
        control_options=control_options,
        planned_controls=planned_controls
    )
    
    max_retries = 2
    safety_feedback = ""
    last_decision = None
    
    for attempt in range(max_retries + 1):
        # Make decision with mandatory research context
        if attempt == 0:
            # First attempt - normal decision with research
            decision = decision_crew.decide(
                layer_number=layer,
                control_options=control_options,
                planned_controls=planned_controls,
                scores=scores,
                research_context=research_context
            )
        else:
            # Retry with validation feedback
            decision = decision_crew.decide_with_feedback(
                layer_number=layer,
                control_options=control_options,
                planned_controls=planned_controls,
                scores=scores,
                research_context=research_context,
                validation_feedback=safety_feedback
            )
        
        # Validate decision with safety crew
        validation = safety_crew.validate(layer_number=layer,
                                         decision_result=decision,
                                         control_options=control_options,
                                         scores=scores)
        
        safety_feedback = validation["feedback"]
        
        # If valid, break out of retry loop
        if validation["is_valid"]:
            break
        
        # Store the last attempt
        last_decision = decision
        
        # If this was the last attempt or regeneration not needed, use mathematical fallback
        if attempt == max_retries or not validation["requires_regeneration"]:
            # Mathematical fallback: choose the option with the lowest score
            correct_option = int(np.argmin(scores))
            decision = {
                "best_option": correct_option,
                "reasoning": f"Safety fallback: Selected option {correct_option} with lowest score {scores[correct_option]} after {attempt + 1} attempts. Previous validation feedback: {safety_feedback}",
                "raw_text": f'{{"best_option": {correct_option}, "reasoning": "Safety fallback selection"}}'
            }
            safety_feedback += f" [Applied mathematical fallback to option {correct_option}]"
            break

    # Final validation to ensure our decision is correct
    final_validation = safety_crew.validate(layer_number=layer,
                                           decision_result=decision,
                                           control_options=control_options,
                                           scores=scores)
    
    # Update safety feedback with final validation
    if not final_validation["is_valid"]:
        safety_feedback += f" [WARNING: Final validation still failed: {final_validation['feedback']}]"

    human_option = int(np.argmin(scores))
    attention_flag = human_option is not None and decision["best_option"] != human_option

    return {
        "option": decision["best_option"],
        "explanation": decision["reasoning"],
        "label": "CrewAI",
        "human_option": human_option,
        "attention": attention_flag,
        "safety_validated": final_validation["is_valid"],
        "safety_feedback": safety_feedback,
        "used_fallback": "Safety fallback" in decision["reasoning"],
        "attempts_made": attempt + 1,
        # Research fields (mandatory Azure AI Foundry Deep Research)
        "research_summary": research_context["research_findings"][:300] + "..." if len(research_context["research_findings"]) > 300 else research_context["research_findings"],
        "research_citations": len(research_context["citations"]),
        "research_parameters": research_context["parameter_context"],
    }

@mcp.tool()
def get_latest(n: int = None) -> str:
    """
    Return the latest task(s) as a JSON string.
    """
    ctx = mcp.get_context()
    tasks = ctx.request_context.lifespan_context.tasks
    if not tasks:
        return "No tasks available."
    if n is None:
        return json.dumps(tasks[-1])
    return json.dumps(tasks[-n])


@mcp.tool()
def check_liveness() -> str:
    """
    Check if the agent is running.
    """

    return f"I'm {mcp.name} and I'm very artificially intelligent!"


@mcp.tool()
def check_llm() -> str:
    """
    Check if the agent can talk to the LLM service.
    """
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-10-21"  
    )
    llm = FlowceptLLM(llm)
    result = llm.invoke("hi!")
    return result


def main():
    """
    Start the MCP server.
    """
    def uvicorn_run():
        uvicorn.run(
            mcp.streamable_http_app, host=AGENT_HOST, port=AGENT_PORT, lifespan="on"
        )
    Thread(target=uvicorn_run).start()
    sleep(2)  # Allow some time for uvicorn to start
    print(run_tool(check_liveness, host=AGENT_HOST, port=AGENT_PORT)[0])


if __name__ == "__main__":
    _pkg_root = pathlib.Path(__file__).resolve().parent
    _dotenv_path = _pkg_root / ".env"
    if _dotenv_path.is_file():
        load_dotenv(dotenv_path=_dotenv_path)
        print(f"Loaded environment variables from {_dotenv_path}")

    pkg_root = Path(importlib.import_module("manufacturing_agent").__file__).resolve().parent
    load_dotenv(pkg_root / ".env") 

    main()
