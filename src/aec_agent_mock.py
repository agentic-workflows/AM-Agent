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
from langchain_openai import ChatOpenAI
from aec_agent_context_manager import AdamantineAeCContextManager

try:
    from manufacturing_agent.crew import OptionGenerationCrew, DecisionCrew
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


def build_llm():
    #model_name = AGENT.get("model_name", "o4-mini-2025-04-16")
    model_name = "gpt-4o"
    llm = ChatOpenAI(model=model_name)
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
    crew = DecisionCrew(llm=llm)
    decision = crew.decide(layer_number=layer,
                           control_options=control_options,
                           planned_controls=planned_controls,
                           scores=scores)

    human_option = int(np.argmin(scores))
    attention_flag = human_option is not None and decision["best_option"] != human_option

    return {
        "option": decision["best_option"],
        "explanation": decision["reasoning"],
        "label": "CrewAI",
        "human_option": human_option,
        "attention": attention_flag,
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
    llm = ChatOpenAI(model="o4-mini-2025-04-16", temperature=1)
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
