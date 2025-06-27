import json
import os
import sys
from typing import Dict, List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import uvicorn
from flowcept.instrumentation.agent_flowcept_task import agent_flowcept_task
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
import pathlib
from flowcept.configs import AGENT
from flowcept.flowceptor.adapters.agents.agents_utils import convert_mcp_to_langchain, build_llm_model, tuples_to_langchain_messages
from flowcept.flowceptor.adapters.agents.flowcept_llm_prov_capture import invoke_llm, add_preamble_to_response
from examples.agents.aec_prompts import choose_option_prompt, generate_options_set_prompt
from examples.agents.aec_agent_context_manager import AdamantineAeCContextManager
from langchain_openai import ChatOpenAI 

# Add manufacturing_agent to path to allow bridge import
MANUFACTURING_AGENT_SRC_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "manufacturing-agent"
    / "manufacturing_agent"
    / "src"
)
sys.path.append(str(MANUFACTURING_AGENT_SRC_PATH))

# Load the .env file from the manufacturing_agent directory
from dotenv import load_dotenv
dotenv_path = MANUFACTURING_AGENT_SRC_PATH / "manufacturing_agent" / ".env"
if dotenv_path.is_file():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"Could not find .env file at {dotenv_path}")


from manufacturing_agent.crew import OptionGenerationCrew, DecisionCrew

os.environ["SAMBASTUDIO_URL"] = AGENT.get("llm_server_url")
os.environ["SAMBASTUDIO_API_KEY"] = AGENT.get("api_key")


agent_controller = AdamantineAeCContextManager()
mcp = FastMCP("AnC_Agent_mock", require_session=True, lifespan=agent_controller.lifespan)



#################################################
# TOOLS
#################################################


@mcp.tool()
@agent_flowcept_task  # Must be in this order. @mcp.tool then @flowcept_task
def generate_options_set(layer: int, planned_controls, number_of_options=4, campaign_id=None):
    model_name = AGENT.get("openai_model", "gpt-4o")
    llm = ChatOpenAI(model=model_name)
    crew = OptionGenerationCrew(llm=llm)
    result = crew.generate(layer_number=layer, planned_controls=planned_controls, number_of_options=number_of_options, campaign_id=campaign_id)
    return result


@mcp.tool()
@agent_flowcept_task  # Must be in this order. @mcp.tool then @flowcept_task
def choose_option(scores: Dict, planned_controls: List[Dict], campaign_id: str=None):
    model_name = AGENT.get("openai_model", "gpt-4o")
    llm = ChatOpenAI(model=model_name)
    crew = DecisionCrew(llm=llm)
    decision = crew.decide(layer_number=scores.get("layer", 0), planned_controls=planned_controls, scores=scores, campaign_id=campaign_id)

    human_option = int(np.argmin(scores["scores"])) if "scores" in scores else None
    attention_flag = human_option is not None and decision["best_option"] != human_option

    print({
        "option": decision["best_option"],
        "explanation": decision["reasoning"],
        "label": "CrewAI",
        "human_option": human_option,
        "attention": attention_flag,
        "response": decision["raw_text"],
        "prompt": decision["prompt_msgs"],
        "llm": llm,
    })

    return {
        "option": decision["best_option"],
        "explanation": decision["reasoning"],
        "label": "CrewAI",
        "human_option": human_option,
        "attention": attention_flag,
        "response": decision["raw_text"],
        "prompt": decision["prompt_msgs"],
        "llm": llm,
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

    return f"I'm {mcp.name} and I'm ready!"


@mcp.tool()
def check_llm() -> str:
    """
    Check if the agent can talk to the LLM service.
    """

    messages = [base.UserMessage(f"Hi, are you working properly?")]

    langchain_messages = convert_mcp_to_langchain(messages)
    response = invoke_llm(langchain_messages)
    result = add_preamble_to_response(response, mcp)

    return result


def main():
    """
    Start the MCP server.
    """
    uvicorn.run(
        mcp.streamable_http_app, host=AGENT.get("mcp_host", "0.0.0.0"), port=AGENT.get("mcp_port", 8000), lifespan="on"
    )


if __name__ == "__main__":
    main()
