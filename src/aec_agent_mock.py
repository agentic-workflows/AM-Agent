import json
import os
import sys
from threading import Thread
from time import sleep
from typing import Dict, List, Optional

from flowcept.agents.agent_client import run_tool
from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM, agent_flowcept_task, get_current_context_task
from flowcept.configs import AGENT_HOST, AGENT_PORT
# Using local build_llm function instead of utils.build_llm

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Disable CrewAI/OpenTelemetry telemetry to avoid external calls/timeouts
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import numpy as np
import uvicorn

from mcp.server.fastmcp import FastMCP
import pathlib
from flowcept.configs import AGENT
from langchain_openai import AzureChatOpenAI
from aec_agent_context_manager import AdamantineAeCContextManager

try:
    from manufacturing_agent.crew import OptionGenerationCrew, DecisionCrew, SafetyValidationCrew, CitationClassificationCrew, ResearchService
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


def get_perplexity_config():
    """
    Load Perplexity configuration for deep research
    Required for the research service to function
    """
    try:
        # Check for Perplexity environment variables
        perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
        perplexity_enabled = os.environ.get("PERPLEXITY_ENABLED", "true").lower() == "true" 
        
        if perplexity_enabled and perplexity_api_key:
            return {
                "api_key": perplexity_api_key,
                "model": "sonar-deep-research",
                "endpoint": "https://api.perplexity.ai/chat/completions",
                "temperature": 0.2,
                "timeout": 600  # Keep original timeout to avoid breaking API calls
            }
        else:
            # Configuration not provided - research will be unavailable
            return {
                "api_key": None,
                "note": "Perplexity API key not provided - Deep Research unavailable"
            }
            
    except Exception as e:
        print(f"Error loading Perplexity config: {e}")
        return {"api_key": None, "error": str(e)}

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
def choose_option(layer: int, control_options: Optional[List[Dict]] = None, scores: List = (), planned_controls: Optional[List[Dict]] = None, user_message: str = "", campaign_id: str=None):
    # Tolerate None inputs from the driver
    control_options = control_options or []
    planned_controls = planned_controls or []
    llm = build_llm()
    ctx = mcp.get_context()
    history = ctx.request_context.lifespan_context.history
    
    # Get user message from context if not provided directly
    if not user_message and hasattr(ctx.request_context.lifespan_context, 'user_messages'):
        user_message = ctx.request_context.lifespan_context.user_messages.get(layer, "")
    
    # Store user message in context for this layer
    if hasattr(ctx.request_context.lifespan_context, 'user_messages'):
        ctx.request_context.lifespan_context.user_messages[layer] = user_message
    
    decision_crew = DecisionCrew(llm=llm)
    citation_crew = CitationClassificationCrew(llm=llm)
    safety_crew = SafetyValidationCrew(llm=llm)
    
    # Step 1: Run Decision and Research in Parallel
    print(f"\nStarting Parallel Processing for Layer {layer}...")
    
    # Start Deep Research (runs independently)
    print(f"📚 Launching deep research analysis...")
    
    USE_MOCK_RESEARCH = True  # Set to False to use real Perplexity API
    
    if USE_MOCK_RESEARCH:
        print(f"Using MOCK research service for testing...")
        from mock_research_service import MockResearchService
        research_service = MockResearchService()
    else:
        print(f"Using REAL Perplexity research service...")
        perplexity_config = get_perplexity_config()
        research_service = ResearchService(perplexity_config)
    # Determine if this is the first processed request in the current lifespan
    is_first_request = False
    lifespan_ctx = ctx.request_context.lifespan_context
    if hasattr(lifespan_ctx, 'research_started'):
        research_already_started = getattr(lifespan_ctx, 'research_started')
        is_first_request = not research_already_started
        print(f"🔍 Research status check: research_started={research_already_started}, is_first_request={is_first_request}")
    else:
        is_first_request = True
        print(f"🔍 No research status found - treating as first request: is_first_request={is_first_request}")
        try:
            setattr(lifespan_ctx, 'research_started', False)
        except Exception:
            pass

    research_context = research_service.get_research_background_strict_first_only(
        is_first_request=is_first_request,
        control_options=control_options,
        planned_controls=planned_controls,
        force=False
    )

    # Only mark research as started if it actually succeeded (not skipped and has citations)
    if is_first_request and not research_context.get("skipped", False):
        # Check if research actually produced results
        citations = research_context.get("citations", []) or research_context.get("web_sources", [])
        if citations:  # Only mark as started if we got actual research results
            print(f"✅ Research succeeded with {len(citations)} citations - marking as started")
            try:
                setattr(lifespan_ctx, 'research_started', True)
                # Store the successful research results for reuse by subsequent layers
                setattr(lifespan_ctx, 'cached_research_results', research_context)
            except Exception:
                pass
        else:
            print(f"⚠️ Research attempt failed/empty - will retry on next layer")
    elif not is_first_request:
        # Check if we have cached research results from a previous successful layer
        if hasattr(lifespan_ctx, 'cached_research_results'):
            cached_results = getattr(lifespan_ctx, 'cached_research_results')
            cached_citations = cached_results.get("citations", []) or cached_results.get("web_sources", [])
            if cached_citations:
                print(f"📚 Using cached research results from previous layer ({len(cached_citations)} citations)")
                # Use the cached research results instead of empty results
                research_context = cached_results.copy()  # Copy to avoid mutation
                research_context["from_cache"] = True
                research_context["reused_from_layer"] = True
            else:
                print(f"📚 Cached results exist but are empty - using empty results")
        else:
            print(f"📚 No cached research results available - using empty results")
    
    max_retries = 2
    safety_feedback = ""
    last_decision = None
    final_citation_analysis = None
    
    # Build combined user guidance (strict concatenation up to current layer)
    def _combine_user_messages(user_msgs: Dict[int, str], up_to_layer: int) -> str:
        lines = []
        for i in sorted([k for k in user_msgs.keys() if k <= up_to_layer]):
            msg = (user_msgs.get(i) or "").strip()
            if msg:
                lines.append(msg)
        return "\n".join(lines)

    def _build_history_json(hist: List[Dict], up_to_layer: int):
        result = []
        for item in hist:
            try:
                layer_ix = item.get("layer")
                if layer_ix is not None and layer_ix < up_to_layer:
                    result.append({
                        "layer": layer_ix,
                        "control_options": item.get("control_options", []),
                        "scores": item.get("scores", []),
                        "chosen_option": item.get("chosen_option"),
                        "reasoning": item.get("explanation", ""),
                    })
            except Exception:
                continue
        return result

    user_msgs = getattr(ctx.request_context.lifespan_context, 'user_messages', {}) or {}
    combined_user_guidance = _combine_user_messages(user_msgs, layer)
    history_json = _build_history_json(history or [], layer)

    for attempt in range(max_retries + 1):
        # Step 2: Make Independent Decision (no research input)
        if attempt == 0:
            # First attempt - independent decision
            print(f"🎯 Making independent decision (attempt {attempt + 1})...")
            decision = decision_crew.decide(
                layer_number=layer,
                control_options=control_options,
                planned_controls=planned_controls,
                scores=scores,
                user_message=user_message,
                combined_user_guidance=combined_user_guidance,
                history_json=history_json
            )
        else:
            # Retry with validation feedback
            print(f"🔄 Retrying decision with feedback (attempt {attempt + 1})...")
            decision = decision_crew.decide_with_feedback(
                layer_number=layer,
                control_options=control_options,
                planned_controls=planned_controls,
                scores=scores,
                validation_feedback=safety_feedback,
                user_message=user_message,
                combined_user_guidance=combined_user_guidance,
                history_json=history_json
            )
        
        # Step 3: Analyze Literature Attitude (paper-by-paper evaluation)
        print(f"📊 Evaluating literature attitude against decision...")
        citation_analysis = citation_crew.classify(
            layer_number=layer,
            decision_result=decision,
            research_context=research_context,
            control_options=control_options
        )
        
        # 3. Validate decision with safety crew
        validation = safety_crew.validate(layer_number=layer,
                                         decision_result=decision,
                                         control_options=control_options,
                                         scores=scores,
                                         combined_user_guidance=combined_user_guidance)
        
        safety_feedback = validation["feedback"]
        
        # Store the citation analysis for the current attempt
        final_citation_analysis = citation_analysis
        
        # If valid, break out of retry loop
        if validation["is_valid"]:
            break
        
        # Store the last attempt
        last_decision = decision
        
        # If this was the last attempt or regeneration not needed, stop without mathematical fallback
        if attempt == max_retries or not validation["requires_regeneration"]:
            print("Decision could not be validated after retries; returning failure")
            break

    # Final validation to ensure our decision is correct
    final_validation = safety_crew.validate(layer_number=layer,
                                           decision_result=decision,
                                           control_options=control_options,
                                           scores=scores)
    
    # Update safety feedback with final validation
    if not final_validation["is_valid"]:
        safety_feedback += f" [WARNING: Final validation still failed: {final_validation['feedback']}]"

    human_option = None
    attention_flag = False
    
    # Determine research support boolean based on literature attitude
    literature_attitude = final_citation_analysis["overall_attitude"] if final_citation_analysis else "NO_DATA"
    if literature_attitude == "POSITIVE":
        research_supports_decision = True
    elif literature_attitude == "NEGATIVE":
        research_supports_decision = False
    else:  # NEUTRAL, MIXED, NO_DATA
        research_supports_decision = None
    
    # Additional attention flag for literature support
    literature_attention_flag = research_supports_decision is False
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
        "research_summary": research_context["research_findings"][:300] + "..." if len(research_context["research_findings"]) > 300 else research_context["research_findings"],
        "research_citations": len(research_context["citations"]),
        "research_parameters": research_context["parameter_context"],
        # Literature attitude analysis
        "literature_attitude": literature_attitude,
        "literature_distribution": final_citation_analysis["distribution"] if final_citation_analysis else {"positive": 0, "neutral": 0, "negative": 0},
        "total_papers_analyzed": final_citation_analysis["total_papers"] if final_citation_analysis else 0,
        "paper_by_paper_analysis": final_citation_analysis["paper_analyses"] if final_citation_analysis else [],
        "decision_vs_literature": {
            "decision": final_citation_analysis["decision_summary"] if final_citation_analysis else {},
            "literature_attitude": literature_attitude
        },
        # NEW: Research support boolean and related fields
        "research_supports_decision": research_supports_decision,
        "research_support_confidence": final_citation_analysis.get("confidence", 0.0) if final_citation_analysis else 0.0,
        "literature_attention_flag": literature_attention_flag,
        "user_message": user_message,
        "user_message_provided": bool(user_message.strip())
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
